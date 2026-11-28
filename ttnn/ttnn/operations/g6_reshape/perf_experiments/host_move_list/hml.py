# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""host_move_list: g6_reshape TILE cells with every transfer precomputed on the HOST.

The host walks the output (per output face-row -> input pieces), assigns pieces to movers, coalesces
reads per input page, places each read run in staging at the DRAM-read phase (L1 == src mod 64), and
decides per output run whether it can be written straight from staging (zero-copy: staging == dst
mod 16, the DRAM write alignment) or must be copied into an image first. The result is per-core
runtime args; the device kernel does no walk arithmetic at all.

Runtime args depend only on the geometry (page ids or bank id + bank-local offsets), the two buffer
base addresses are common runtime args, so a cached descriptor stays correct when only the addresses
are patched (same contract as g6_reshape._DESCRIPTOR_CACHE). The one address assumption the plan
makes is base % 64 == 0 (checked on every call).

Variants (VARIANTS below):
  addr="page"  device: TensorAccessor.get_noc_addr(page, off)
  addr="bank"  device: get_noc_addr_from_bank_id(bank, base + bank_local_off) (no div/mod at all)
  split=True   a core's pieces are dealt over BOTH movers (64 B output-word groups), not the baseline's
               atom ranges (baseline gives mover 1 nothing when a core owns one atom)
  gap          read-run coalescing gap inside one input page (bytes read through)
  zones        DeviceZoneScopedN zones on/off
  ct=True      "ideal floor": one kernel per core, every transfer a compile-time constant (test only)
  null=True    empty kernel on the same cores/RISCs (launch floor)
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field
from pathlib import Path

import ttnn

BASE = importlib.import_module("ttnn.operations.g6_reshape.g6_reshape")  # read-only use: plan/ownership

HERE = Path(__file__).parent
KERNEL = HERE / "kernels" / "hml_mover.cpp"
KERNEL_CT = HERE / "kernels" / "hml_mover_ct.cpp"
KERNEL_NULL = HERE / "kernels" / "hml_null.cpp"

TILE, FACE = 32, 16
READ_ALIGN = 64  # NOC_DRAM_READ_ALIGNMENT_BYTES (BH)
WRITE_ALIGN = 16  # NOC_DRAM_WRITE_ALIGNMENT_BYTES (BH)
WORDS_PER_XFER = 3
MAX_RT_WORDS = 4096  # max_runtime_args_tensix (unique + common)
HEADER_WORDS = 4
COMMON_WORDS = 2

MOVER_CFG = (
    ((0, 1), ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
    ((2, 3), ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
)


@dataclass(frozen=True)
class Variant:
    addr: str = "bank"
    split: bool = False
    gap: int = 64
    zones: bool = True
    ct: bool = False
    null: bool = False
    policy: str = "auto"  # auto | img


VARIANTS = {
    "hml_page": Variant(addr="page"),
    "hml_bank": Variant(addr="bank"),
    "hml_bank_split": Variant(addr="bank", split=True),
    "hml_bank_1read": Variant(addr="bank", gap=1024),
    "hml_bank_split_nz": Variant(addr="bank", split=True, zones=False),
    "hml_bank_nz": Variant(addr="bank", zones=False),
    "floor_ct": Variant(addr="bank", zones=False, ct=True),
    "floor_ct_split": Variant(addr="bank", split=True, zones=False, ct=True),
    "null": Variant(null=True),
    "hml_bank_img": Variant(addr="bank", policy="img"),
    "hml_bank_zc": Variant(addr="bank", policy="zc"),
    "hml_bank_split_zc": Variant(addr="bank", split=True, policy="zc"),
    "hml_bank_zc_1read": Variant(addr="bank", policy="zc", gap=1024),
}


# ---------------------------------------------------------------------------
# Host walk: output face-rows -> pieces
# ---------------------------------------------------------------------------


@dataclass
class Piece:
    in_page: int
    in_off: int
    out_page: int
    out_off: int
    nbytes: int
    tail: int = 0  # padding bytes after the piece up to the face-row end (written, don't-care)


def _atom_pieces(plan, a):
    """Pieces of atom a (same atom numbering as the baseline op)."""
    src, dst, e = plan.src, plan.dst, plan.elem
    frb = FACE * e
    fb = FACE * frb
    atoms_per_image = -(-dst.H // plan.atom_rows)
    atoms_per_col = dst.B * atoms_per_image
    tc = a // atoms_per_col
    q = a % atoms_per_col
    b = q // atoms_per_image
    hg = q % atoms_per_image
    out = []
    col_left = dst.W - TILE * tc
    halves = 2 if col_left > FACE else 1
    for h in range(hg * plan.atom_rows, min(dst.H, (hg + 1) * plan.atom_rows)):
        out_page = (b * dst.Ht + h // TILE) * dst.Wt + tc
        R = b * dst.H + h
        for half in range(halves):
            n = min(FACE, col_left - FACE * half)
            out_fr = (2 * ((h % TILE) // FACE) + half) * fb + (h % FACE) * frb
            s = R * dst.W + TILE * tc + FACE * half
            k = 0
            row_pieces = []
            while k < n:
                r, c = divmod(s + k, src.W)
                rb, rh = divmod(r, src.H)
                ln = min(n - k, FACE - c % FACE, src.W - c)
                in_page = (rb * src.Ht + rh // TILE) * src.Wt + c // TILE
                in_off = (2 * ((rh % TILE) // FACE) + (c % TILE) // FACE) * fb + (rh % FACE) * frb + (c % FACE) * e
                row_pieces.append(Piece(in_page, in_off, out_page, out_fr + k * e, ln * e))
                k += ln
            row_pieces[-1].tail = (FACE - n) * e
            out.extend(row_pieces)
    return out


# ---------------------------------------------------------------------------
# Per-mover transfer program
# ---------------------------------------------------------------------------


@dataclass
class MoverProg:
    reads: list = field(default_factory=list)  # (page, off, len, stage_off)
    zc: list = field(default_factory=list)  # (page, off, len, stage_off)
    copies: list = field(default_factory=list)  # (stage_off, img_off, len)
    imgw: list = field(default_factory=list)  # (page, off, len, img_off)
    stage_bytes: int = 0
    image_bytes: int = 0

    @property
    def n_xfers(self):
        return len(self.reads) + len(self.zc) + len(self.copies) + len(self.imgw)


def _mover_prog(pieces, gap, policy):
    prog = MoverProg()
    if not pieces:
        return prog
    # Reads: per input page, merge intervals within `gap` bytes, place at the read phase.
    by_page = {}
    for p in pieces:
        by_page.setdefault(p.in_page, []).append(p)
    cursor = 0
    stage_of = {}
    for page in sorted(by_page):
        ivs = sorted(by_page[page], key=lambda p: p.in_off)
        runs = []
        for p in ivs:
            if runs and p.in_off <= runs[-1][1] + gap:
                runs[-1][1] = max(runs[-1][1], p.in_off + p.nbytes)
                runs[-1][2].append(p)
            else:
                runs.append([p.in_off, p.in_off + p.nbytes, [p]])
        for start, end, ps in runs:
            stage = cursor + ((start - cursor) % READ_ALIGN)
            prog.reads.append((page, start, end - start, stage))
            for p in ps:
                stage_of[id(p)] = stage + (p.in_off - start)
            cursor = stage + (end - start)
    prog.stage_bytes = cursor + READ_ALIGN  # slack: zero-copy tails may read past a run
    # Writes: output runs (contiguous in one output page incl. tails).
    by_out = {}
    for p in pieces:
        by_out.setdefault(p.out_page, []).append(p)
    img_cursor = 0
    for page in sorted(by_out):
        ps = sorted(by_out[page], key=lambda p: p.out_off)
        oruns = []
        for p in ps:
            if oruns and p.out_off == oruns[-1][-1].out_off + oruns[-1][-1].nbytes + oruns[-1][-1].tail:
                oruns[-1].append(p)
            else:
                oruns.append([p])
        for run in oruns:
            # Sub-runs contiguous in staging as well.
            subs = []
            for p in run:
                st = stage_of[id(p)]
                if subs and st == subs[-1][1]:
                    subs[-1][1] = st + p.nbytes + p.tail
                    subs[-1][3] += p.nbytes + p.tail
                else:
                    subs.append([st, st + p.nbytes + p.tail, p.out_off, p.nbytes + p.tail])
            phase_ok = all((s[0] - s[2]) % WRITE_ALIGN == 0 for s in subs)
            if phase_ok and (policy == "zc" or (policy == "auto" and len(subs) == 1)):
                for st, _, off, ln in subs:
                    prog.zc.append((page, off, ln, st))
                continue
            off0 = run[0].out_off
            img = img_cursor + ((off0 - img_cursor) % WRITE_ALIGN)
            total = run[-1].out_off + run[-1].nbytes + run[-1].tail - off0
            for p in run:
                prog.copies.append((stage_of[id(p)], img + p.out_off - off0, p.nbytes))
            prog.imgw.append((page, off0, total, img))
            img_cursor = img + total
    prog.image_bytes = img_cursor
    return prog


def _split(per_mover):
    """Deal a core's pieces over both movers in 64 B output-word groups (no two movers write one word)."""
    groups = {}
    for p in per_mover[0] + per_mover[1]:
        groups.setdefault((p.out_page, p.out_off // READ_ALIGN), []).append(p)
    keys = sorted(groups)
    half = (len(keys) + 1) // 2
    return [[p for k in keys[:half] for p in groups[k]], [p for k in keys[half:] for p in groups[k]]]


@dataclass
class HmlPlan:
    plan: object
    variant: Variant
    cores: list  # [(x, y)]
    progs: dict  # (mover, (x, y)) -> MoverProg
    num_banks: int
    in_page_bytes: int
    out_page_bytes: int

    def max_rt_words(self):
        return max(HEADER_WORDS + WORDS_PER_XFER * p.n_xfers for p in self.progs.values()) + COMMON_WORDS


def make_hml_plan(input_tensor, out_shape, variant):
    device = input_tensor.device()
    in_shape = tuple(int(d) for d in input_tensor.shape)
    grid = device.compute_with_storage_grid_size()
    plan = BASE._make_plan(in_shape, out_shape, input_tensor.element_size(), grid, ttnn.get_dram_alignment())
    nmov = len(plan.roles)
    cores = [xy for (xy, _, _) in plan.roles[0]]
    progs = {}
    for ci, xy in enumerate(cores):
        per_mover = []
        for m in range(nmov):
            _, begin, count = plan.roles[m][ci]
            ps = []
            for a in range(begin, begin + count):
                ps.extend(_atom_pieces(plan, a))
            per_mover.append(ps)
        if variant.split and nmov == 2:
            per_mover = _split(per_mover)
        for m in range(nmov):
            progs[(m, xy)] = _mover_prog(per_mover[m], variant.gap, variant.policy)
    nb = ttnn.device.get_memory_view(device, ttnn.BufferType.DRAM).num_banks
    tile_bytes = TILE * TILE * input_tensor.element_size()
    return HmlPlan(plan, variant, cores, progs, nb, tile_bytes, tile_bytes)


# ---------------------------------------------------------------------------
# Encoding + descriptor
# ---------------------------------------------------------------------------


def _enc_dram(hp, page, off, ln, l1_off, page_bytes):
    if hp.variant.addr == "page":
        assert off < (1 << 16) and ln < (1 << 16)
        return [page, off | (ln << 16), l1_off]
    bank, slot = page % hp.num_banks, page // hp.num_banks
    assert ln < (1 << 24)
    return [bank | (ln << 8), slot * page_bytes + off, l1_off]


def encode(hp, prog):
    args = [len(prog.reads), len(prog.zc), len(prog.copies), len(prog.imgw)]
    for page, off, ln, st in prog.reads:
        args += _enc_dram(hp, page, off, ln, st, hp.in_page_bytes)
    for page, off, ln, st in prog.zc:
        args += _enc_dram(hp, page, off, ln, st, hp.out_page_bytes)
    for st, img, ln in prog.copies:
        args += [st, img, ln]
    for page, off, ln, img in prog.imgw:
        args += _enc_dram(hp, page, off, ln, img, hp.out_page_bytes)
    return args


def _core_range_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])


def build_descriptor(hp, input_tensor, output_tensor):
    v = hp.variant
    crs = hp.plan.core_ranges()
    common = [input_tensor.buffer_address(), output_tensor.buffer_address()]
    in_acc = ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    out_acc = ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()
    nmov = len(hp.plan.roles)
    kernels, cbs = [], []
    defines = [("HML_ZONES", "1" if v.zones else "0")]
    for m in range(nmov):
        (cb_stage, cb_image), proc, noc = MOVER_CFG[m]
        cfg = ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)
        if v.null:
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_NULL),
                    core_ranges=crs,
                    compile_time_args=[],
                    runtime_args=ttnn.RuntimeArgs(),
                    common_runtime_args=common,
                    config=cfg,
                )
            )
            continue
        stage_bytes = max(64, max(hp.progs[(m, xy)].stage_bytes for xy in hp.cores)) + READ_ALIGN
        image_bytes = max(64, max(hp.progs[(m, xy)].image_bytes for xy in hp.cores)) + READ_ALIGN
        ct_head = [hp.plan.elem, int(v.addr == "bank"), cb_stage, cb_image, hp.in_page_bytes]
        if v.ct:
            for xy in hp.cores:
                args = encode(hp, hp.progs[(m, xy)])
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=str(KERNEL_CT),
                        core_ranges=_core_range_set([xy]),
                        compile_time_args=[*ct_head, *args],
                        runtime_args=ttnn.RuntimeArgs(),
                        common_runtime_args=common,
                        config=cfg,
                        defines=defines,
                    )
                )
        else:
            rt = ttnn.RuntimeArgs()
            for x, y in hp.cores:
                rt[x][y] = encode(hp, hp.progs[(m, (x, y))])
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL),
                    core_ranges=crs,
                    compile_time_args=[*ct_head, *in_acc, *out_acc],
                    runtime_args=rt,
                    common_runtime_args=common,
                    config=cfg,
                    defines=defines,
                )
            )
        for cb_id, size in ((cb_stage, stage_bytes), (cb_image, image_bytes)):
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=size,
                    core_ranges=crs,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=input_tensor.dtype, page_size=size)
                    ],
                )
            )
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


_CACHE = {}
LAST_PLAN = {}


def hml_reshape(input_tensor, shape, variant="hml_bank"):
    v = VARIANTS[variant] if isinstance(variant, str) else variant
    out_shape = BASE.validate(input_tensor, shape)
    assert input_tensor.layout == ttnn.TILE_LAYOUT, "host_move_list bench covers TILE cells only"
    device = input_tensor.device()
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape), input_tensor.dtype, input_tensor.layout, device, input_tensor.memory_config()
    )
    a_in, a_out = input_tensor.buffer_address(), output.buffer_address()
    # The plan's only address assumption (read phase is address-independent).
    assert a_in % READ_ALIGN == 0 and a_out % READ_ALIGN == 0, (a_in, a_out)
    key = (tuple(input_tensor.shape), out_shape, input_tensor.dtype, v)
    desc = _CACHE.get(key)
    if desc is None:
        hp = make_hml_plan(input_tensor, out_shape, v)
        words = hp.max_rt_words()
        if words > MAX_RT_WORDS:
            raise RuntimeError(f"host_move_list: {words} RT words per core > {MAX_RT_WORDS}")
        LAST_PLAN[key] = hp
        desc = build_descriptor(hp, input_tensor, output)
        _CACHE[key] = desc
    else:
        for k in desc.kernels:
            k.common_runtime_args = [a_in, a_out]
    return ttnn.generic_op([input_tensor, output], desc)


def plan_stats(in_shape, out_shape, elem, variant, grid=(13, 10), dram_align=64, num_banks=8):
    """Host-only RT-word / transfer statistics (no device): for the feasibility sweep."""

    class G:
        x, y = grid

    v = VARIANTS[variant] if isinstance(variant, str) else variant
    plan = BASE._make_plan(tuple(in_shape), tuple(out_shape), elem, G, dram_align)
    nmov = len(plan.roles)
    worst = 0
    worst_x = None
    for ci in range(plan.num_cores):
        per_mover = []
        for m in range(nmov):
            _, begin, count = plan.roles[m][ci]
            ps = []
            for a in range(begin, begin + count):
                ps.extend(_atom_pieces(plan, a))
            per_mover.append(ps)
        if v.split and nmov == 2:
            per_mover = _split(per_mover)
        for m in range(nmov):
            pr = _mover_prog(per_mover[m], v.gap, v.policy)
            w = HEADER_WORDS + WORDS_PER_XFER * pr.n_xfers + COMMON_WORDS
            if w > worst:
                worst, worst_x = w, (len(pr.reads), len(pr.zc), len(pr.copies), len(pr.imgw))
        if ci > 3 and worst > MAX_RT_WORDS:
            break
    return dict(
        cores=plan.num_cores, atoms=plan.atoms_total, regime=plan.regime, worst_words=worst, worst_xfers=worst_x
    )
