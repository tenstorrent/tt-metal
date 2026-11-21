# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for permute (Quasar): three ProgramSpecs, exactly one dispatched per call.

Regimes (op_design.md, "Regimes"), chosen by `select_regime(layout, d)` on the canonical rank-4 permutation:

  page_gather     every output page is one input page (RM with W last; TILE with H, W in place).
                  reader --cb_pages--> writer, implicit sync both ends, plan (R, 0, W) = (1, 0, 1).
  tile_transpose  TILE with H and W swapped: output tile = transpose(one input tile).
                  reader --cb_input_tiles--> NEO transpose --cb_output_tiles--> writer, plan (1, 1, 1).
  staged_gather   everything else: boxes over the four input axes are staged whole in a per-thread scratchpad
                  region and shuffled into output pages by GATHER_THREADS worker harts of one DM kernel.

Every knob of the Blocking Model is a named constant here or derived by `choose_gather_block()`; the kernels
take them as compile-time / runtime arguments and never restate one.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

TILE = 32

# ---------------------------------------------------------------------------
# Knobs (op_design.md "Blocking Model"; worked values for Quasar)
# ---------------------------------------------------------------------------

# Thread plans: (R readers, C NEOs, W writers) per ring regime; staged_gather: worker harts of its one DM kernel.
GATHER_THREADS = 6
THREAD_PLANS = {
    "page_gather": (1, 0, 1),
    "tile_transpose": (1, 1, 1),
    "staged_gather": GATHER_THREADS,
}
FREE_DM_HARTS = 6  # of 8 per cluster: DM0 serves ring interrupts, DM1 the remapper (target page, cluster model)
NEOS = 4

PAGE_RING_DEPTH = 8  # cb_pages entries (page_gather)
TRANSPOSE_BLOCK_TILES = 4  # tiles per DEST window; <= DEST_AUTO_LIMIT (16-bit DEST, half sync: 8), asserted in-kernel
TRANSPOSE_RING_BLOCKS = 2  # windows per NEO in each tile_transpose ring (double buffer)
OUT_SLOTS = 8  # staged_gather output slots per worker thread
RING_CAP_BYTES = 1_048_560  # one ring's capacity on this family (target page, cluster model)
L1_RESERVE_BASE = 262_144  # headroom kept out of the scratchpad budget (op_design.md "Mechanism caps")


def _align_up(x, a):
    return (x + a - 1) // a * a


def _ceil(a, b):
    return -(-a // b)


def _prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


# ---------------------------------------------------------------------------
# Canonical form and regime selection
# ---------------------------------------------------------------------------


def canonical(shape, dims):
    """(s, d): the input shape padded to rank 4 with leading 1s, and the permutation lifted to it."""
    rank = len(shape)
    k = 4 - rank
    s = [1] * k + [int(x) for x in shape]
    d = list(range(k)) + [x + k for x in dims]
    return s, d


def select_regime(layout, d):
    """Pinned (op_design.md "Regime-selection function"): depends on (layout, d) only."""
    if layout == ttnn.ROW_MAJOR_LAYOUT and d[3] == 3:
        return "page_gather"
    if layout == ttnn.TILE_LAYOUT and d[2] == 2 and d[3] == 3:
        return "page_gather"
    if layout == ttnn.TILE_LAYOUT and d[2] == 3 and d[3] == 2:
        return "tile_transpose"
    return "staged_gather"


def split_work(n, grid):
    """(core set, {core: count}, {core: first index}) over the cores that get work, row-wise."""
    _, all_cores, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(grid, n, row_wise=True)
    counts, starts, start = {}, {}, 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            counts[core], starts[core] = per_core, start
            start += per_core
    return all_cores, counts, starts


def _buffer_alignment(tensor):
    if tensor.memory_config().buffer_type == ttnn.BufferType.L1:
        return ttnn.get_l1_alignment()
    return ttnn.get_dram_alignment()


def _element_bytes(dtype):
    return {ttnn.bfloat16: 2, ttnn.float32: 4, ttnn.int32: 4, ttnn.uint32: 4}[dtype]


def _ring_format(dtype):
    # uint32 is not a buffer format on this family: a ring that only moves bytes carries it as int32.
    return ttnn.int32 if dtype == ttnn.uint32 else dtype


def _dm_config(make, explicit_rings=()):
    hw = make()
    if explicit_rings:
        hw.config_2xx = E.DataMovementHardwareConfig.DataMovement2XXConfig(
            disable_dfb_implicit_sync_for=list(explicit_rings)
        )
    return hw


def _tensor_parameters(inp, out):
    return [E.TensorParameter(unique_id="input", spec=inp.spec), E.TensorParameter(unique_id="output", spec=out.spec)]


TENSOR_INDICES = {"input": 0, "output": 1}


def _mixed_radix(index, radices):
    """Digits of `index` over `radices` (most significant first)."""
    digits = [0] * len(radices)
    for k in range(len(radices) - 1, -1, -1):
        digits[k] = index % radices[k]
        index //= radices[k]
    return digits


# ---------------------------------------------------------------------------
# page_gather / tile_transpose: the odometer over the output page grid
# ---------------------------------------------------------------------------


def page_geometry(layout, s, d, regime):
    """(g, st): the output page grid g[0..3] (output page order, last fastest) and, per grid axis, the input
    page stride st[k], so that output page (c0..c3) reads input page sum(c_k * st[k])."""
    t = [s[j] for j in d]
    if layout == ttnn.ROW_MAJOR_LAYOUT:
        in_stride = [s[1] * s[2], s[2], 1]  # input page (row) strides of input axes 0..2; axis 3 is in-page
        return [t[0], t[1], t[2], 1], [in_stride[d[0]], in_stride[d[1]], in_stride[d[2]], 0]
    ht, wt = _ceil(s[2], TILE), _ceil(s[3], TILE)
    in_stride = [s[1] * ht * wt, ht * wt]  # input tile strides of the leading axes 0, 1
    if regime == "page_gather":  # H, W in place: output tile (o0, o1, h, w) <- input tile (.., h, w)
        return [t[0], t[1], ht, wt], [in_stride[d[0]], in_stride[d[1]], wt, 1]
    # tile_transpose: output tile (o0, o1, a, b) = transpose(input tile (.., h = b, w = a))
    return [t[0], t[1], wt, ht], [in_stride[d[0]], in_stride[d[1]], 1, wt]


def _page_run_args(inp, out, s, d, regime):
    layout = inp.layout
    g, st = page_geometry(layout, s, d, regime)
    n_pages = _prod(g)
    assert n_pages == out.buffer_num_pages(), (n_pages, out.buffer_num_pages())
    core_set, counts, starts = split_work(n_pages, inp.device().compute_with_storage_grid_size())
    per_core = {"num_pages": counts, "start_page": starts}
    for k in range(4):
        per_core[f"c{k}"] = {}
    for core, start in starts.items():
        for k, c in enumerate(_mixed_radix(start, g)):
            per_core[f"c{k}"][core] = c
    common = {**{f"g{k}": g[k] for k in range(4)}, **{f"st{k}": st[k] for k in range(4)}}
    return core_set, counts, per_core, common


READER_RT = ["num_pages", "start_page", "c0", "c1", "c2", "c3"]
READER_CRT = ["g0", "g1", "g2", "g3", "st0", "st1", "st2", "st3"]
WRITER_RT = ["num_pages", "start_page"]


def _entry_bytes(tensor):
    # Implicit sync moves a whole entry per page: the tensor's own aligned page slot, so a transfer never
    # spills into the next page and source / destination stay buffer-aligned.
    return _align_up(tensor.buffer_page_size(), _buffer_alignment(tensor))


def _reader_writer(plan_r, plan_w, ring_in, ring_out, rt_common, per_core):
    plan = {"reader_threads": plan_r, "writer_threads": plan_w}
    reader = E.KernelSpec(
        unique_id="reader",
        source=str(KERNEL_DIR / "permute_reader.cpp"),
        hw_config=_dm_config(E.create_reader_datamovement_config),
        num_threads=plan_r,
        dfb_bindings=[E.producer_of(ring_in, "ring")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="input", accessor_name="input")],
        compile_time_args=plan,
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(
            runtime_arg_names=READER_RT, common_runtime_arg_names=READER_CRT
        ),
    )
    writer = E.KernelSpec(
        unique_id="writer",
        source=str(KERNEL_DIR / "permute_writer.cpp"),
        hw_config=_dm_config(E.create_writer_datamovement_config),
        num_threads=plan_w,
        dfb_bindings=[E.consumer_of(ring_out, "ring")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="output", accessor_name="output")],
        compile_time_args=plan,
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=WRITER_RT),
    )
    reader_args = E.KernelRunArgs(
        kernel="reader",
        runtime_arg_values={k: per_core[k] for k in READER_RT},
        common_runtime_arg_values=rt_common,
    )
    writer_args = E.KernelRunArgs(kernel="writer", runtime_arg_values={k: per_core[k] for k in WRITER_RT})
    return reader, writer, reader_args, writer_args


def build_page_gather(inp, out, s, d, *, page_ring_depth=PAGE_RING_DEPTH, plan=None):
    R, _, W = plan or THREAD_PLANS["page_gather"]
    entry = _entry_bytes(inp)
    if entry * page_ring_depth > RING_CAP_BYTES:
        raise NotImplementedError(f"permute: a row of {entry} B x {page_ring_depth} entries exceeds the ring cap")
    core_set, _, per_core, common = _page_run_args(inp, out, s, d, "page_gather")
    ring = E.DataflowBufferSpec(
        unique_id="cb_pages",
        entry_size=entry,
        num_entries=page_ring_depth,
        data_format_metadata=_ring_format(inp.dtype),
    )
    reader, writer, reader_args, writer_args = _reader_writer(R, W, "cb_pages", "cb_pages", common, per_core)
    spec = E.ProgramSpec(
        name="permute_page_gather",
        kernels=[reader, writer],
        dataflow_buffers=[ring],
        tensor_parameters=_tensor_parameters(inp, out),
        work_units=[E.WorkUnitSpec(name="main", kernels=["reader", "writer"], target_nodes=core_set)],
    )
    return spec, E.ProgramRunArgs(kernel_run_args=[reader_args, writer_args]), TENSOR_INDICES


def build_tile_transpose(
    inp, out, s, d, *, block_tiles=TRANSPOSE_BLOCK_TILES, ring_blocks=TRANSPOSE_RING_BLOCKS, plan=None
):
    R, C, W = plan or THREAD_PLANS["tile_transpose"]
    if not (1 <= C <= NEOS and R + W <= FREE_DM_HARTS and C % R == 0 and C % W == 0):
        raise ValueError(f"thread plan {(R, C, W)}: need C <= {NEOS}, R + W <= {FREE_DM_HARTS}, R | C, W | C")
    tile_bytes = inp.buffer_page_size()
    core_set, counts, per_core, common = _page_run_args(inp, out, s, d, "tile_transpose")
    # Capacity is per NEO (STRIDED ring sliced into C sub-rings), a whole number of compute windows.
    entries = ring_blocks * block_tiles * C
    rings = [
        E.DataflowBufferSpec(
            unique_id=name, entry_size=tile_bytes, num_entries=entries, data_format_metadata=_ring_format(inp.dtype)
        )
        for name in ("cb_input_tiles", "cb_output_tiles")
    ]
    reader, writer, reader_args, writer_args = _reader_writer(
        R, W, "cb_input_tiles", "cb_output_tiles", common, per_core
    )
    compute = E.KernelSpec(
        unique_id="compute",
        source=str(KERNEL_DIR / "permute_compute.cpp"),
        # Datacopy multiplies nothing, so LoFi is exact; ttsim aborts with the fidelity unset. 16-bit DEST: a
        # bf16 unpack -> datacopy -> pack is bit-exact (and UnpackToDest of bf16 into 32-bit DEST aborts ttsim).
        hw_config=E.ComputeHardwareConfig(fpu_math_fidelity=ttnn.MathFidelity.LoFi, enable_32_bit_dest=False),
        num_threads=C,
        dfb_bindings=[E.consumer_of("cb_input_tiles", "tiles_in"), E.producer_of("cb_output_tiles", "tiles_out")],
        compile_time_args={"compute_threads": C, "block_tiles": block_tiles},
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["num_pages"]),
    )
    compute_args = E.KernelRunArgs(kernel="compute", runtime_arg_values={"num_pages": counts})
    spec = E.ProgramSpec(
        name="permute_tile_transpose",
        kernels=[reader, compute, writer],
        dataflow_buffers=rings,
        tensor_parameters=_tensor_parameters(inp, out),
        work_units=[E.WorkUnitSpec(name="main", kernels=["reader", "compute", "writer"], target_nodes=core_set)],
    )
    return spec, E.ProgramRunArgs(kernel_run_args=[reader_args, compute_args, writer_args]), TENSOR_INDICES


# ---------------------------------------------------------------------------
# staged_gather
# ---------------------------------------------------------------------------


def _l1_reserve_bytes(tensors):
    """Headroom plus each L1-resident tensor's per-bank share: the scratchpad must not reach into L1 tensors."""
    reserve = L1_RESERVE_BASE
    l1 = [t for t in tensors if t.memory_config().buffer_type == ttnn.BufferType.L1]
    if l1:
        num_banks = ttnn.device.get_memory_view(l1[0].device(), ttnn.BufferType.L1).num_banks
        for t in l1:
            reserve += _ceil(t.buffer_num_pages(), num_banks) * t.buffer_aligned_page_size()
    return reserve


class GatherGeometry:
    """Box quanta, the footprint model and `choose_gather_block()` (op_design.md, Work Distribution)."""

    def __init__(self, layout, s, d, element_bytes, align_in_elems, align_out_elems, slot_align):
        self.tile = layout == ttnn.TILE_LAYOUT
        self.s, self.d = s, d
        self.a, self.b = d[2], d[3]
        self.eb = element_bytes
        self.slot_align = slot_align
        self.tile_bytes = TILE * TILE * element_bytes
        if self.tile:
            self.q = [TILE if j in (2, 3, self.a, self.b) else 1 for j in range(4)]
        else:
            self.q = [1, 1, 1, 1]
            self.q[3] = align_in_elems
            self.q[self.b] = align_out_elems

    def n_blocks(self, e):
        return _prod(_ceil(self.s[j], e[j]) for j in range(4))

    def staging_bytes(self, e):
        if self.tile:
            return e[0] * e[1] * _ceil(e[2], TILE) * _ceil(e[3], TILE) * self.tile_bytes
        return e[0] * e[1] * e[2] * self.row_stride(e)

    def row_stride(self, e):
        return _align_up(e[3] * self.eb, self.slot_align)

    def slot_bytes(self, e):
        return self.tile_bytes if self.tile else e[self.b] * self.eb

    def per_thread_bytes(self, e, out_slots):
        return _align_up(self.staging_bytes(e), self.slot_align) + out_slots * _align_up(
            self.slot_bytes(e), self.slot_align
        )

    def start(self):
        if self.tile:
            return [min(self.s[j], self.q[j]) for j in range(4)]
        e = [1, 1, 1, 1]
        e[3], e[self.b] = self.s[3], self.s[self.b]
        return e

    def _chunk(self, length, parts, q):
        return max(q, _align_up(_ceil(length, parts), q))

    def choose(self, workers, budget_per_thread, out_slots):
        s, q = self.s, self.q
        e = self.start()
        # 2. Fit: RM halves the longer of e3 / e_b, rounded up to its quantum, until the box fits.
        while not self.tile and self.per_thread_bytes(e, out_slots) > budget_per_thread:
            j = 3 if e[3] >= e[self.b] else self.b
            if e[j] <= q[j]:
                j = self.b if j == 3 else 3
                if e[j] <= q[j]:
                    break
            e[j] = self._chunk(e[j], 2, q[j])
        # 3. Occupancy: RM chunks along 3 / b until every worker can have a box.
        if not self.tile:
            for _ in range(4):
                changed = False
                order = [3, self.b] if e[3] >= e[self.b] else [self.b, 3]
                for j in order:
                    nb = self.n_blocks(e)
                    if nb >= workers or e[j] <= q[j]:
                        continue
                    new = self._chunk(e[j], _ceil(workers, nb), q[j])
                    if new < e[j]:
                        e[j], changed = new, True
                if not changed:
                    break
        # 4. Coarsen: while there are >= 2 boxes per worker, grow the free axis innermost in output order.
        while self.n_blocks(e) >= 2 * workers:
            grown = False
            for j in [self.d[k] for k in (3, 2, 1, 0)]:
                if q[j] != 1 or e[j] >= s[j]:
                    continue
                trial = list(e)
                trial[j] = min(s[j], 2 * e[j])
                if self.per_thread_bytes(trial, out_slots) <= budget_per_thread:
                    e, grown = trial, True
                    break
            if not grown and self.tile:
                for j in [self.d[k] for k in (3, 2, 1, 0)]:
                    if q[j] == 1 or e[j] >= s[j]:
                        continue
                    trial = list(e)
                    trial[j] = min(s[j], e[j] + TILE)
                    if self.per_thread_bytes(trial, out_slots) <= budget_per_thread:
                        e, grown = trial, True
                        break
            if not grown:
                break
        return e


def build_staged_gather(inp, out, s, d, *, gather_threads=GATHER_THREADS, out_slots=OUT_SLOTS, extent=None):
    device = inp.device()
    eb = _element_bytes(inp.dtype)
    slot_align = max(ttnn.get_l1_alignment(), ttnn.get_dram_alignment())
    align_elems = _buffer_alignment(inp) // eb  # input and output share a buffer type (R7)
    geo = GatherGeometry(inp.layout, s, d, eb, align_elems, _buffer_alignment(out) // eb, slot_align)

    grid = device.compute_with_storage_grid_size()
    n_cores = grid.x * grid.y
    budget = ttnn.get_max_worker_l1_unreserved_size() - _l1_reserve_bytes([inp, out])
    min_bytes = geo.per_thread_bytes(geo.start() if geo.tile else _min_rm_box(geo), out_slots)
    threads = min(gather_threads, budget // min_bytes) if budget > 0 else 0
    if threads < 1:
        raise NotImplementedError(f"permute: the smallest box ({min_bytes} B) does not fit L1 ({budget} B)")
    budget_per_thread = (budget // threads) // slot_align * slot_align
    e = list(extent) if extent is not None else geo.choose(n_cores * threads, budget_per_thread, out_slots)
    assert geo.per_thread_bytes(e, out_slots) <= budget_per_thread, (e, budget_per_thread)

    nb = [_ceil(s[j], e[j]) for j in range(4)]
    n_blocks = _prod(nb)
    core_set, counts, starts = split_work(n_blocks, grid)

    t = [s[j] for j in d]
    if geo.tile:
        ht, wt = _ceil(s[2], TILE), _ceil(s[3], TILE)
        in_ps = [s[1] * ht * wt, ht * wt, wt, 1]  # input page strides of axes 0, 1 and of a tile step on 2, 3
        out_grid = [t[1], _ceil(t[2], TILE), _ceil(t[3], TILE)]
        nh, nw = _ceil(e[2], TILE), _ceil(e[3], TILE)
        tstride = [e[1] * nh * nw, nh * nw, nw, 1]  # staging tile strides per input axis
        stg = [x * TILE * TILE for x in tstride]  # element offset of one step (axes 0, 1) / one tile (2, 3)
    else:
        in_ps = [s[1] * s[2], s[2], 1, 0]
        out_grid = [t[1], t[2], 1]
        row = geo.row_stride(e) // eb
        stg = [e[1] * e[2] * row, e[2] * row, row, 1]  # staging element strides per input axis
    per_thread = geo.per_thread_bytes(e, out_slots)
    staging = _align_up(geo.staging_bytes(e), slot_align)
    common = {
        **{f"s{j}": s[j] for j in range(4)},
        **{f"e{j}": e[j] for j in range(4)},
        **{f"nb{j}": nb[j] for j in range(4)},
        **{f"d{k}": d[k] for k in range(4)},
        **{f"ps{j}": in_ps[j] for j in range(4)},
        **{f"stg{j}": stg[j] for j in range(4)},
        "og1": out_grid[0],
        "og2": out_grid[1],
        "og3": out_grid[2],
        "per_thread_bytes": per_thread,
        "staging_bytes": staging,
        "slot_stride": _align_up(geo.slot_bytes(e), slot_align),
        "row_stride_bytes": 0 if geo.tile else geo.row_stride(e),
        # RM sub-page transfer sizes round up to this; it stays inside the page's own aligned slot.
        "xfer_align": ttnn.get_l1_alignment(),
    }
    scratch = E.ScratchpadSpec(unique_id="scratch_gather", size_per_node=threads * per_thread)
    kernel = E.KernelSpec(
        unique_id="gather",
        source=str(KERNEL_DIR / "permute_gather.cpp"),
        hw_config=_dm_config(E.create_reader_datamovement_config),
        num_threads=threads,
        scratchpad_bindings=[E.ScratchpadBinding(scratchpad_spec_name="scratch_gather", accessor_name="gather")],
        tensor_bindings=[
            E.TensorBinding(tensor_parameter_name="input", accessor_name="input"),
            E.TensorBinding(tensor_parameter_name="output", accessor_name="output"),
        ],
        compile_time_args={
            "gather_threads": threads,
            "out_slots": out_slots,
            "element_bytes": eb,
            "tile_layout": int(geo.tile),
        },
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(
            runtime_arg_names=["num_blocks", "start_block"], common_runtime_arg_names=list(common)
        ),
    )
    spec = E.ProgramSpec(
        name="permute_staged_gather",
        kernels=[kernel],
        scratchpads=[scratch],
        tensor_parameters=_tensor_parameters(inp, out),
        work_units=[E.WorkUnitSpec(name="main", kernels=["gather"], target_nodes=core_set)],
    )
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(
                kernel="gather",
                runtime_arg_values={"num_blocks": counts, "start_block": starts},
                common_runtime_arg_values=common,
            )
        ]
    )
    return spec, run_args, TENSOR_INDICES


def _min_rm_box(geo):
    e = geo.start()
    for j in (3, geo.b):
        e[j] = min(e[j], geo.q[j])
    return e


BUILDERS = {"page_gather": build_page_gather, "tile_transpose": build_tile_transpose, "staged_gather": build_staged_gather}


def create_program_artifacts(inp, out, dims, **knobs):
    """(spec, run_args, tensor_indices) of the one program this call dispatches. `dims` is normalized."""
    s, d = canonical(inp.shape, dims)
    regime = select_regime(inp.layout, d)
    return BUILDERS[regime](inp, out, s, d, **knobs)
