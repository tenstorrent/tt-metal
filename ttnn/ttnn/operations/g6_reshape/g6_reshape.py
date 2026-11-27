# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""g6_reshape: torch.reshape on the logical values, laid out again in the input's own layout.

One generic_op dispatch, data-movement kernels only. Output-stationary: each mover owns whole output
atoms, gathers their input face-rows into L1 staging with coalesced reads, permutes them into an
image in output byte order with RISC-V copies, and writes contiguous output runs. TILE inputs take
tile_face_row / tile_element_run (g6_reshape_mover.cpp); ROW_MAJOR inputs take rm_row_gather
(g6_reshape_rm_mover.cpp: atom = one DRAM-alignment chunk of an output page slot, folded bank-affine so a
block stores as one write per bank). Design: op_design.md; L1 accounting: l1_ledger.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

KERNEL = Path(__file__).parent / "kernels" / "g6_reshape_mover.cpp"
RM_KERNEL = Path(__file__).parent / "kernels" / "g6_reshape_rm_mover.cpp"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def tag_dtype(inputs, axes):
    return inputs[0]["dtype"]


def tag_layout(inputs, axes):
    return inputs[0]["layout"]


INPUT_TAGGERS = {
    "dtype": tag_dtype,
    "layout": tag_layout,
}

SUPPORTED = {
    "dtype": [ttnn.bfloat16, ttnn.float32, ttnn.int32, ttnn.uint32],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
}

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# Block knobs (each defined once; CB sizes, trip counts and RT ranges derive from them)
# ---------------------------------------------------------------------------

MAX_CORES = None  # None: the whole compute grid
MOVERS_PER_CORE = 2
MAX_BLOCK_ATOMS = None  # None: the coarsest block that fits (one block per mover unless L1 caps it)
STAGE_DEPTH = 1
IMAGE_DEPTH = 1
# rm_row_gather: a covering read run reads through a gap of up to this many bytes inside one input page
# instead of opening a new run (None: one DRAM alignment word). Bounded by the DRAM alignment so the
# per-piece staging bound below stays 3·A.
RM_READ_GAP_BYTES = None
# rm_row_gather atom fold. "linear": a block is one contiguous stream range (each input byte read once, so a
# single hot input bank serves no stride amplification) with a bank-grouped image (<= NB writes per block).
# "bank": bank-affine fold (one write per (block, bank), but a mover's input elements are NB pages apart).
RM_FOLD = "linear"
# rm_row_gather shared read (Refinement 3): with the linear fold and one block per mover, one mover per core reads
# the whole core's stream range into one staging CB and hands it to its partner through a local semaphore; the
# reading mover alternates by core between the two NoCs. Halves the requests every NoC carries to the input's
# (often single, hot) DRAM bank, whose per-NoC request queue sets the read-await tail. False: each mover reads its
# own block (Refinement 2 behaviour).
RM_SHARED_READ = True
# rm_row_gather bank split (Refinement 3; needs the shared read and one atom per output page): a core's movers
# take alternate bank groups of the core's whole staged range instead of its two stream halves, so each mover
# issues ~NB / movers writes instead of ~NB, with the same atom count.
RM_BANK_SPLIT = True

# BB-05: L1 the model leaves to the op on every core.
L1_OP_BUDGET = 1347776

TILE = 32
FACE = 16
COPY_ENTRY_BYTES = 8
SHARED_READ_SEM = 0  # semaphore id of the rm shared-read handoff
TILE_ENTRY_BYTES = 16
CB_ALIGN = 64

# Mover m's NoC (noc_placement knob): reader/writer-native (RISCV_1 on NoC0, RISCV_0 on NoC1). Refinement 2
# measured the swapped placement ~7% slower on the flagged [1,4096]->[4096,1] int32 and on the tile
# (16,1,5120)->(1,1,16,5120) guard, neutral elsewhere.
MOVER_NOCS = (ttnn.NOC.NOC_0, ttnn.NOC.NOC_1)

# Mover m: CB ids (stage, image, plan) and the RISC-V it runs on (mover 0: RISCV_1, mover 1: RISCV_0).
MOVERS = (
    ((0, 1, 2), ttnn.DataMovementProcessor.RISCV_1),
    ((3, 4, 5), ttnn.DataMovementProcessor.RISCV_0),
)
assert 1 <= MOVERS_PER_CORE <= len(MOVERS) == len(MOVER_NOCS)


def _knobs():
    """Every module-level knob that changes the program; part of the descriptor cache key."""
    return (
        MAX_CORES,
        MOVERS_PER_CORE,
        MAX_BLOCK_ATOMS,
        STAGE_DEPTH,
        IMAGE_DEPTH,
        RM_READ_GAP_BYTES,
        RM_FOLD,
        RM_SHARED_READ,
        RM_BANK_SPLIT,
        MOVER_NOCS,
        L1_OP_BUDGET,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _resolve_shape(numel, shape):
    try:
        entries = [int(s) for s in shape]
    except TypeError as exc:
        raise ValueError(f"g6_reshape: shape must be a list or tuple of ints, got {shape!r}") from exc
    if not 1 <= len(entries) <= 4:
        raise ValueError(f"g6_reshape: output rank must be 1..4, got {len(entries)}")
    if any(s == 0 or s < -1 for s in entries):
        raise ValueError(f"g6_reshape: shape entries must be positive or a single -1, got {entries}")
    inferred = [i for i, s in enumerate(entries) if s == -1]
    if len(inferred) > 1:
        raise ValueError(f"g6_reshape: at most one -1 entry, got {entries}")
    if inferred:
        known = math.prod(s for s in entries if s != -1)
        if numel % known != 0:
            raise ValueError(f"g6_reshape: cannot infer -1 in {entries} for {numel} elements")
        entries[inferred[0]] = numel // known
    if math.prod(entries) != numel:
        raise ValueError(f"g6_reshape: shape {entries} holds {math.prod(entries)} elements, input holds {numel}")
    return tuple(entries)


def validate(input_tensor, shape):
    """Support refusal (SUPPORTED, then EXCLUSIONS), placement guard, then shape checks. Returns the
    resolved output shape."""
    axes = {"dtype": input_tensor.dtype, "layout": input_tensor.layout}
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_reshape: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_reshape: unsupported combination (refinement candidate): {exc}")

    memory_config = input_tensor.memory_config()
    if (
        memory_config.buffer_type != ttnn.BufferType.DRAM
        or memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
    ):
        raise UnsupportedAxisValue(f"g6_reshape: input must be DRAM interleaved, got {memory_config}")

    in_shape = tuple(int(d) for d in input_tensor.shape)
    if not 1 <= len(in_shape) <= 4:
        raise ValueError(f"g6_reshape: input rank must be 1..4, got {len(in_shape)}")
    return _resolve_shape(math.prod(in_shape), shape)


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Geometry:
    B: int
    H: int
    W: int
    Ht: int
    Wt: int

    @classmethod
    def of(cls, shape):
        W = shape[-1]
        H = shape[-2] if len(shape) >= 2 else 1
        B = math.prod(shape[:-2]) if len(shape) > 2 else 1
        return cls(B=B, H=H, W=W, Ht=-(-H // TILE), Wt=-(-W // TILE))

    @property
    def num_pages(self):
        return self.B * self.Ht * self.Wt


class _CoreRangesMixin:
    def core_ranges(self):
        full_rows, rem = divmod(self.num_cores, self.grid_x)
        ranges = []
        if full_rows:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.grid_x - 1, full_rows - 1)))
        if rem:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
        return ttnn.CoreRangeSet(ranges)

    @property
    def stage_bytes(self):
        return STAGE_DEPTH * self.stage_slot_bytes

    @property
    def image_bytes(self):
        return IMAGE_DEPTH * self.image_slot_bytes


@dataclass(frozen=True)
class _Plan(_CoreRangesMixin):
    elem: int
    src: _Geometry
    dst: _Geometry
    regime: str
    atom_rows: int
    atoms_total: int
    block_atoms: int
    src_face_rows_bound: int
    dram_align: int
    stage_slot_bytes: int
    image_slot_bytes: int
    plan_bytes: int
    max_copies: int
    grid_x: int
    num_cores: int
    roles: tuple  # per mover: tuple of ((x, y), atom_begin, atom_count)

    @property
    def l1_bytes_per_core(self):
        return _core_bytes(self.stage_slot_bytes, self.image_slot_bytes, self.plan_bytes)

    @property
    def out_pages(self):
        return self.dst.num_pages


def _round_up(x, m):
    return -(-x // m) * m


def _max_copies(trs, s_bound):
    """Copy entries of a block: <= s_bound pieces per output face-row, two face-rows per TRS."""
    return 2 * trs * s_bound


def _slot_bytes(atoms, atom_rows, s_bound, face_row_bytes, dram_align):
    """Unrounded (staging slot, image slot, plan) bytes of one mover for a block of `atoms` atoms. Staging
    holds each source face-row plus up to `dram_align` of phase pad for the run it opens. An atom never
    straddles an output tile, so the tile table needs at most one entry per atom."""
    trs = atoms * atom_rows
    max_copies = _max_copies(trs, s_bound)
    return (
        max_copies * (face_row_bytes + dram_align),
        2 * trs * face_row_bytes,
        max_copies * COPY_ENTRY_BYTES + atoms * TILE_ENTRY_BYTES,
    )


def _core_bytes(stage_slot, image_slot, plan):
    """L1 per core: every mover's staging and image at their buffer depths, plus its plan."""
    return MOVERS_PER_CORE * (STAGE_DEPTH * stage_slot + IMAGE_DEPTH * image_slot + plan)


def _split_atoms(atoms_total, grid):
    """Atoms [0, atoms_total) over min(max_cores, atoms_total) cores taken row-major from the grid, each
    core's range split over its movers. Returns (num_cores, roles per mover, max atoms of any mover)."""
    max_cores = grid.x * grid.y if MAX_CORES is None else min(MAX_CORES, grid.x * grid.y)
    num_cores = min(max_cores, atoms_total)
    roles = [[] for _ in range(MOVERS_PER_CORE)]
    max_mover_atoms = 0
    for i in range(num_cores):
        begin = i * atoms_total // num_cores
        count = (i + 1) * atoms_total // num_cores - begin
        xy = (i % grid.x, i // grid.x)
        for m in range(MOVERS_PER_CORE):
            share = count // MOVERS_PER_CORE + (1 if m < count % MOVERS_PER_CORE else 0)
            roles[m].append((xy, begin, share))
            begin += share
            max_mover_atoms = max(max_mover_atoms, share)
    return num_cores, tuple(tuple(r) for r in roles), max_mover_atoms


def _fit_block_atoms(max_mover_atoms, slot_bytes_fn):
    """The coarsest block (≤ the largest mover's share, ≤ MAX_BLOCK_ATOMS) whose CB_ALIGN-rounded
    footprint fits L1_OP_BUDGET. slot_bytes_fn(atoms) -> unrounded (stage slot, image slot, plan) bytes,
    linear in atoms. Returns (block_atoms, rounded slot bytes)."""
    bytes_per_atom = _core_bytes(*slot_bytes_fn(1))
    l1_cap_atoms = L1_OP_BUDGET // bytes_per_atom
    if l1_cap_atoms < 1:
        raise ValueError(f"g6_reshape: one atom needs more than {L1_OP_BUDGET} B of L1 per core")
    block_atoms = max(1, min(max_mover_atoms, l1_cap_atoms))
    if MAX_BLOCK_ATOMS is not None:
        block_atoms = max(1, min(block_atoms, MAX_BLOCK_ATOMS))
    while True:
        slots = tuple(_round_up(b, CB_ALIGN) for b in slot_bytes_fn(block_atoms))
        total = _core_bytes(*slots)
        if total <= L1_OP_BUDGET or block_atoms == 1:
            break
        block_atoms -= 1
    if total > L1_OP_BUDGET:
        raise ValueError(f"g6_reshape: {total} B of L1 per core exceeds the {L1_OP_BUDGET} B budget")
    return block_atoms, slots


def _make_plan(in_shape, out_shape, elem, grid, dram_align):
    src, dst = _Geometry.of(in_shape), _Geometry.of(out_shape)
    face_row_bytes = FACE * elem
    aligned = src.W % FACE == 0 and dst.W % FACE == 0
    regime = "tile_face_row" if aligned else "tile_element_run"
    # Input face-rows one output face-row can touch.
    if aligned:
        s_bound = 1
    elif src.W >= FACE:
        s_bound = 3
    else:
        s_bound = -(-(FACE - 1) // src.W) + 1

    # The ownership atom: rows sharing one DRAM-alignment word of a face never split across movers.
    atom_rows = max(1, dram_align // face_row_bytes)
    atoms_total = dst.Wt * dst.B * -(-dst.H // atom_rows)

    num_cores, roles, max_mover_atoms = _split_atoms(atoms_total, grid)

    # Unrounded footprint is linear in atoms, so one atom gives the BB-05 cap on block_atoms.
    block_atoms, (stage_slot, image_slot, plan_bytes) = _fit_block_atoms(
        max_mover_atoms, lambda n: _slot_bytes(n, atom_rows, s_bound, face_row_bytes, dram_align)
    )
    max_copies = _max_copies(block_atoms * atom_rows, s_bound)

    return _Plan(
        elem=elem,
        src=src,
        dst=dst,
        regime=regime,
        atom_rows=atom_rows,
        atoms_total=atoms_total,
        block_atoms=block_atoms,
        src_face_rows_bound=s_bound,
        dram_align=dram_align,
        stage_slot_bytes=stage_slot,
        image_slot_bytes=image_slot,
        plan_bytes=plan_bytes,
        max_copies=max_copies,
        grid_x=grid.x,
        num_cores=num_cores,
        roles=roles,
    )


@dataclass(frozen=True)
class _RmPlan(_CoreRangesMixin):
    """rm_row_gather: page = one logical row; atom = one A-byte chunk of an output page slot
    (align(W_out·e, A) bytes), folded bank-affine on the device (see g6_reshape_rm_mover.cpp)."""

    elem: int
    w_in: int
    rows_in: int
    w_out: int
    rows_out: int
    atoms_per_page: int
    atoms_total: int
    block_atoms: int
    src_face_rows_bound: int  # input pieces one atom can touch (the regime's S)
    dram_align: int
    read_gap: int
    stage_slot_bytes: int
    image_slot_bytes: int
    plan_bytes: int
    max_copies: int
    grid_x: int
    num_cores: int
    roles: tuple
    shared_read: bool  # one covering read per core, shared by its movers (RM_SHARED_READ)
    bank_split: bool  # movers split the core's bank groups (RM_BANK_SPLIT)

    regime = "rm_row_gather"
    atom_rows = 1

    @property
    def out_pages(self):
        return self.rows_out

    @property
    def l1_bytes_per_core(self):
        return _core_bytes(self.stage_slot_bytes, self.image_slot_bytes, self.plan_bytes)

    def mover_runtime_args(self, mover, core_index):
        """[atom_begin, atom_count] (+ [core_begin, core_count, role] under the shared read). Role: 0 waits for
        the partner's read, 1 reads alone (the partner owns no atoms), 2 reads and signals the partner. The
        reader is the mover whose NoC matches the core's checkerboard parity, so readers split evenly over the
        NoCs, unless that mover owns no atoms."""
        (x, y), begin, count = self.roles[mover][core_index]
        if not self.shared_read:
            return [begin, count]
        shares = [self.roles[m][core_index] for m in range(MOVERS_PER_CORE)]
        core_begin = shares[0][1]
        core_count = sum(share[2] for share in shares)
        reader = (x + y) % MOVERS_PER_CORE
        if shares[reader][2] == 0:
            reader = 0
        partner_waits = any(share[2] > 0 for m, share in enumerate(shares) if m != reader)
        role = (2 if partner_waits else 1) if mover == reader else 0
        return [begin, count, core_begin, core_count, role]


def _rm_slot_bytes(atoms, s_bound, dram_align, read_gap, linear_fold):
    """Unrounded (staging slot, image slot, plan) bytes of one rm_row_gather mover for a block of `atoms`.
    A piece is <= A bytes; it opens an A-aligned run (< 3A staged) or extends the open one by at most
    gap + piece + A. The image is A bytes per atom. Staging and image carry A bytes of slack so the kernel
    can start them DRAM-aligned. The linear fold copies plan-free (closed-form sources), so it has no plan."""
    per_piece = 2 * dram_align + max(dram_align, read_gap)
    return (
        atoms * s_bound * per_piece + dram_align,
        atoms * dram_align + dram_align,
        0 if linear_fold else atoms * s_bound * COPY_ENTRY_BYTES,
    )


def _make_rm_plan(in_shape, out_shape, elem, grid, dram_align):
    numel = math.prod(in_shape)
    w_in, w_out = in_shape[-1], out_shape[-1]
    atom_elems = dram_align // elem
    atoms_per_page = -(-(w_out * elem) // dram_align)
    rows_out = numel // w_out
    atoms_total = rows_out * atoms_per_page
    # Input pieces of one atom: atom_elems consecutive stream elements cross at most this many input rows.
    s_bound = min(atom_elems, -(-(atom_elems - 1) // w_in) + 1)
    if RM_FOLD not in ("linear", "bank"):
        raise ValueError(f"g6_reshape: RM_FOLD must be 'linear' or 'bank', got {RM_FOLD!r}")
    read_gap = dram_align if RM_READ_GAP_BYTES is None else RM_READ_GAP_BYTES
    if not 0 <= read_gap <= dram_align:
        raise ValueError(f"g6_reshape: RM_READ_GAP_BYTES must be in [0, {dram_align}], got {read_gap}")

    num_cores, roles, max_mover_atoms = _split_atoms(atoms_total, grid)
    block_atoms, (stage_slot, image_slot, plan_bytes) = _fit_block_atoms(
        max_mover_atoms, lambda n: _rm_slot_bytes(n, s_bound, dram_align, read_gap, RM_FOLD == "linear")
    )
    # The shared read stages the core's whole range (every mover's single block) in one CB.
    shared_read = (
        RM_SHARED_READ
        and RM_FOLD == "linear"
        and MOVERS_PER_CORE > 1
        and STAGE_DEPTH == 1
        and block_atoms >= max_mover_atoms
    )
    return _RmPlan(
        elem=elem,
        w_in=w_in,
        rows_in=numel // w_in,
        w_out=w_out,
        rows_out=rows_out,
        atoms_per_page=atoms_per_page,
        atoms_total=atoms_total,
        block_atoms=block_atoms,
        src_face_rows_bound=s_bound,
        dram_align=dram_align,
        read_gap=read_gap,
        stage_slot_bytes=stage_slot,
        image_slot_bytes=image_slot,
        plan_bytes=plan_bytes,
        max_copies=block_atoms * s_bound,
        grid_x=grid.x,
        num_cores=num_cores,
        roles=roles,
        shared_read=shared_read,
        bank_split=shared_read and RM_BANK_SPLIT and atoms_per_page == 1,
    )


def _plan_for(input_tensor, out_shape):
    device = input_tensor.device()
    in_shape = tuple(int(d) for d in input_tensor.shape)
    if input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT:
        if tuple(int(d) for d in input_tensor.padded_shape) != in_shape:
            raise RuntimeError(f"g6_reshape: ROW_MAJOR input with padding {input_tensor.padded_shape} is not supported")
        plan = _make_rm_plan(
            in_shape,
            out_shape,
            input_tensor.element_size(),
            device.compute_with_storage_grid_size(),
            ttnn.get_dram_alignment(),
        )
        if input_tensor.buffer_num_pages() != plan.rows_in:
            raise RuntimeError(
                f"g6_reshape: input holds {input_tensor.buffer_num_pages()} pages, geometry expects {plan.rows_in}"
            )
        return plan
    plan = _make_plan(
        in_shape,
        out_shape,
        input_tensor.element_size(),
        device.compute_with_storage_grid_size(),
        ttnn.get_dram_alignment(),
    )
    if input_tensor.buffer_num_pages() != plan.src.num_pages:
        raise RuntimeError(
            f"g6_reshape: input holds {input_tensor.buffer_num_pages()} pages, geometry expects {plan.src.num_pages}"
        )
    return plan


# ---------------------------------------------------------------------------
# Program descriptor
# ---------------------------------------------------------------------------


def _tile_compile_args(plan, cb_stage, cb_image, cb_plan, mover):  # mover: unused (same program per mover)
    src, dst = plan.src, plan.dst
    return [
        plan.elem,
        src.W,
        src.H,
        src.Wt,
        src.Ht,
        dst.W,
        dst.H,
        dst.B,
        dst.Wt,
        dst.Ht,
        int(plan.regime == "tile_face_row"),
        plan.atom_rows,
        plan.block_atoms,
        plan.dram_align,
        STAGE_DEPTH,
        IMAGE_DEPTH,
        plan.stage_slot_bytes,
        plan.image_slot_bytes,
        plan.max_copies,
        cb_stage,
        cb_image,
        cb_plan,
    ]


def _rm_compile_args(plan, cb_stage, cb_image, cb_plan, mover):
    # Under the shared read the staging slot is the core's: every mover's slot, pooled in one CB.
    stage_slot_bytes = plan.stage_slot_bytes * (MOVERS_PER_CORE if plan.shared_read else 1)
    return [
        plan.elem,
        plan.w_in,
        plan.w_out,
        plan.rows_out,
        plan.atoms_per_page,
        plan.block_atoms,
        plan.dram_align,
        plan.read_gap,
        STAGE_DEPTH,
        IMAGE_DEPTH,
        stage_slot_bytes,
        plan.image_slot_bytes,
        plan.max_copies,
        cb_stage,
        cb_image,
        cb_plan,
        int(RM_FOLD == "linear"),
        int(plan.shared_read),
        SHARED_READ_SEM,
        int(plan.bank_split),
        mover,
        MOVERS_PER_CORE,
    ]


def _build_descriptor(plan, input_tensor, output_tensor):
    core_ranges = plan.core_ranges()
    in_accessor = ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    out_accessor = ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()
    common_args = [input_tensor.buffer_address(), output_tensor.buffer_address()]
    row_major = plan.regime == "rm_row_gather"
    kernel_source = RM_KERNEL if row_major else KERNEL
    compile_args_fn = _rm_compile_args if row_major else _tile_compile_args

    # rm shared read: every mover stages into mover 0's staging CB, which pools all movers' slots.
    shared_read = row_major and plan.shared_read
    kernels = []
    cbs = []
    semaphores = []
    if shared_read:
        semaphores.append(ttnn.SemaphoreDescriptor(id=SHARED_READ_SEM, core_ranges=core_ranges, initial_value=0))
    for mover in range(MOVERS_PER_CORE):
        (cb_stage, cb_image, cb_plan), processor = MOVERS[mover]
        noc = MOVER_NOCS[mover]
        stage_cb = MOVERS[0][0][0] if shared_read else cb_stage
        stage_bytes = (MOVERS_PER_CORE * plan.stage_bytes if mover == 0 else 0) if shared_read else plan.stage_bytes
        compile_args = [*compile_args_fn(plan, stage_cb, cb_image, cb_plan, mover), *in_accessor, *out_accessor]
        runtime_args = ttnn.RuntimeArgs()
        for i, ((x, y), begin, count) in enumerate(plan.roles[mover]):
            runtime_args[x][y] = plan.mover_runtime_args(mover, i) if row_major else [begin, count]
        config = ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc)
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(kernel_source),
                core_ranges=core_ranges,
                compile_time_args=compile_args,
                runtime_args=runtime_args,
                common_runtime_args=common_args,
                config=config,
            )
        )
        for cb_id, size, fmt in (
            (cb_stage, stage_bytes, input_tensor.dtype),
            (cb_image, plan.image_bytes, input_tensor.dtype),
            (cb_plan, plan.plan_bytes, ttnn.uint32),
        ):
            if size == 0:  # no plan (rm linear fold) / pooled staging (shared read): the kernel never touches it
                continue
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=size,
                    core_ranges=core_ranges,
                    format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=fmt, page_size=size)],
                )
            )
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)


# BB-07: one descriptor per program-changing signature; a hit patches only the buffer addresses.
_DESCRIPTOR_CACHE = {}


def _cache_key(input_tensor, out_shape):
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    return (
        tuple(int(d) for d in input_tensor.shape),
        tuple(int(d) for d in input_tensor.padded_shape),
        out_shape,
        input_tensor.dtype,
        input_tensor.layout,
        str(input_tensor.memory_config()),
        device.id(),
        (grid.x, grid.y),
        _knobs(),
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def g6_reshape(input_tensor: ttnn.Tensor, shape) -> ttnn.Tensor:
    """torch.reshape(input_tensor, shape) on the logical values, in the input's dtype, layout and
    memory configuration, as one device program."""
    out_shape = validate(input_tensor, shape)
    device = input_tensor.device()
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        input_tensor.memory_config(),
    )
    key = _cache_key(input_tensor, out_shape)
    descriptor = _DESCRIPTOR_CACHE.get(key)
    if descriptor is None:
        plan = _plan_for(input_tensor, out_shape)
        if output_tensor.buffer_num_pages() != plan.out_pages:
            raise RuntimeError(
                f"g6_reshape: output holds {output_tensor.buffer_num_pages()} pages, geometry expects {plan.out_pages}"
            )
        if plan.regime == "rm_row_gather" and output_tensor.buffer_aligned_page_size() != (
            plan.atoms_per_page * plan.dram_align
        ):
            raise RuntimeError(
                f"g6_reshape: output page slot is {output_tensor.buffer_aligned_page_size()} B, "
                f"rm_row_gather assumes {plan.atoms_per_page * plan.dram_align} B"
            )
        descriptor = _build_descriptor(plan, input_tensor, output_tensor)
        _DESCRIPTOR_CACHE[key] = descriptor
    else:
        addresses = [input_tensor.buffer_address(), output_tensor.buffer_address()]
        for kernel in descriptor.kernels:
            kernel.common_runtime_args = addresses
    return ttnn.generic_op([input_tensor, output_tensor], descriptor)


def describe_plan(input_tensor: ttnn.Tensor, shape) -> dict:
    """The plan g6_reshape selects for this call (BB-08, BB-21). Validates; never dispatches."""
    out_shape = validate(input_tensor, shape)
    plan = _plan_for(input_tensor, out_shape)
    roles = []
    for mover in range(MOVERS_PER_CORE):
        _, processor = MOVERS[mover]
        noc = MOVER_NOCS[mover]
        for xy, begin, count in plan.roles[mover]:
            roles.append(
                {
                    "core": xy,
                    "mover": mover,
                    "risc": processor.name,
                    "noc": noc.name,
                    "atom_begin": begin,
                    "atom_count": count,
                }
            )
    return {
        "regime": plan.regime,
        "output_shape": out_shape,
        "num_cores": plan.num_cores,
        "core_ranges": plan.core_ranges(),
        "atoms_total": plan.atoms_total,
        "atom_rows": plan.atom_rows,
        "block_atoms": plan.block_atoms,
        "max_blocks_per_mover": -(-max(r["atom_count"] for r in roles) // plan.block_atoms),
        "src_face_rows_bound": plan.src_face_rows_bound,
        "stage_bytes": plan.stage_bytes,
        "image_bytes": plan.image_bytes,
        "plan_bytes": plan.plan_bytes,
        "l1_bytes_per_core": plan.l1_bytes_per_core,
        "shared_read": getattr(plan, "shared_read", False),
        "bank_split": getattr(plan, "bank_split", False),
        "roles": roles,
    }
