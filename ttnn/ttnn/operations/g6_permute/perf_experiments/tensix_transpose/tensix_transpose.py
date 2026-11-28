# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""tensix_transpose experiment: regime nhwc_rm_transpose with the [w, c] -> [c, w] transpose on the
Tensix engine instead of the writer RISC-V.

ROW_MAJOR bfloat16 x [N, H, W, C], dims (0, 3, 1, 2): out[n, c, h, w] = x[n, h, w, c].
Unit (n, h, wb) = input pages [(n * H + h) * W + wb * block_w, + width), block_w = 32 * kw (or the
whole row W when W <= 32 * kw). The reader lands page k of a unit in slot k of Ct * 64 B
(Ct = ceil(C / 32)), so each 32 slots are a 32-row RM block of Ct tiles. The compute tilizes,
transposes and pack-untilizes: for each c-tile it emits 32 rows c of kwu * 32 values w. The writer
sends each row c < C as the partial output row (n, c, h), values [wb * block_w, + width).

Variants (Cfg.variant): two_pass | fused | fused_raw (see kernels/tt_compute.cpp).
CB format (Cfg.fmt): uint16 (opaque 16-bit move) | bf16 (Float16_b through the FP datapath).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import ttnn
from fractions import Fraction

from ttnn.operations.g6_permute import g6_permute_blocks_rm_gather as rm_gather
from ttnn.operations.g6_permute.g6_permute_program_descriptor import (
    L1_OP_BUDGET_BYTES,
    _div_up,
    out_l1_bytes_per_core,
    split_units,
    worker_grid,
)

KDIR = Path(__file__).resolve().parent / "kernels"
DIMS = (0, 3, 1, 2)
CB_IN, CB_ROWS, CB_TILES = 0, 1, 2
TILE = 32
TILE_BYTES = TILE * TILE * 2
# Per-core cost weights (cycles) of _choose_block_w: one page read issued, one partial-row write issued.
READ_CYCLES = 37
WRITE_CYCLES = 42

FMT = {"uint16": ttnn.uint16, "bf16": ttnn.bfloat16}
VARIANT_DEFINE = {
    "two_pass": "TTX_TWO_PASS",
    "fused": "TTX_FUSED",
    "fused_raw": "TTX_FUSED_RAW",
    "fused_d2b": "TTX_FUSED_D2B",
}


@dataclass
class Cfg:
    variant: str = "fused_raw"
    fmt: str = "uint16"
    kw: int | None = None  # w-tiles per unit; None: _choose_block_w
    in_units: int = 2  # cb_in depth, in units (kw w-tile pages each)
    rows_pages: int = 2  # cb_rows depth, in c-tile pages
    push_tiles: int = 1  # w-tiles per reader barrier + push
    first_push_tiles: int = 1  # w-tiles of the reader's first push
    use_trid: bool = False  # reader pushes a group after issuing the next one (transaction-id barriers)
    zones: bool = False
    # own (tt_reader.cpp) | op (the op's shared g6_permute_blocks_reader.cpp, DRAM knobs) | auto (op for
    # DRAM input, own for L1 input: the measured best of each)
    reader: str = "own"
    ablate: tuple = ()  # perf ablation defines (TTX_NO_READS, TTX_NO_WRITES): output is garbage
    max_cores: int | None = None

    def key(self):
        return (
            self.variant,
            self.fmt,
            self.kw,
            self.in_units,
            self.rows_pages,
            self.push_tiles,
            self.first_push_tiles,
            self.use_trid,
            self.zones,
            self.ablate,
            self.reader,
            self.max_cores,
        )


@dataclass
class Plan:
    shape: tuple
    ct: int
    slot_bytes: int
    kw: int
    kw_tail: int
    block_w: int
    tail_w: int
    units_per_row: int
    num_units: int
    num_cores: int
    units_g1: int
    units_g2: int
    all_cores: object = None
    assignment: list = field(default_factory=list)


def _unit_geometry(W, block_w):
    """(kw, block_w, units_per_row, tail_w, kw_tail) for units of block_w input pages: a multiple of 32
    (kw = block_w / 32 w-tiles), or a sub-tile width of 8 or 16 (one w-tile whose rows past block_w are
    never written out); the whole row W when W <= block_w."""
    if W <= block_w:
        kw = _div_up(W, TILE)
        return kw, W, 1, W, kw
    upr = _div_up(W, block_w)
    tail_w = W - (upr - 1) * block_w
    return _div_up(block_w, TILE), block_w, upr, tail_w, _div_up(tail_w, TILE)


# Candidate unit widths in input pages. 8 and 16 spread small tensors over more cores (a partial row
# then starts 16 B-aligned, the NoC write granule of L1 and DRAM).
BLOCK_W_CHOICES = (8, 16, 32, 64, 128, 256)
# Per-core cost weights (cycles): one tile through the compute (fused_d2b, measured compute-only); one
# unit's fixed reader cost (reserve, push, barrier tail), which keeps sub-tile units for small tensors.
TILE_CYCLES = 130
UNIT_CYCLES = 200


def _choose_block_w(N, H, W, C, num_cores, kw_max=8):
    """The unit width minimizing the busiest core's max(read issue, write issue, compute); ties go wider."""
    best = None
    ct = _div_up(C, TILE)
    for block_w in (b for b in BLOCK_W_CHOICES if _div_up(b, TILE) <= kw_max):
        kw, bw, upr, _, _ = _unit_geometry(W, block_w)
        units = N * H * upr
        per_core = _div_up(units, min(units, num_cores))
        cost = max(
            per_core * (bw * READ_CYCLES + UNIT_CYCLES), per_core * C * WRITE_CYCLES, per_core * kw * ct * TILE_CYCLES
        )
        if best is None or cost <= best[0]:
            best = (cost, bw)
        if bw < block_w:
            break
    return best[1]


def build_plan(x, cfg: Cfg) -> Plan:
    N, H, W, C = (int(d) for d in x.shape)
    ct = _div_up(C, TILE)
    grid = worker_grid(x.device())
    if cfg.max_cores is not None:
        cols = min(grid.x, cfg.max_cores)
        grid = ttnn.CoreCoord(cols, max(1, cfg.max_cores // cols))
    # fused_d2b keeps a scratch tile after the unit's last DEST tile: at most 7 of the 8.
    kw_max = 4 if cfg.variant == "fused_d2b" else 8
    block_w = TILE * cfg.kw if cfg.kw is not None else _choose_block_w(N, H, W, C, grid.x * grid.y, kw_max)
    kw, block_w, upr, tail_w, kw_tail = _unit_geometry(W, block_w)
    num_units = N * H * upr
    num_cores, all_cores, g1, g2, assignment = split_units(grid, num_units)
    return Plan(
        shape=(N, H, W, C),
        ct=ct,
        slot_bytes=ct * 64,
        kw=kw,
        kw_tail=kw_tail,
        block_w=block_w,
        tail_w=tail_w,
        units_per_row=upr,
        num_units=num_units,
        num_cores=num_cores,
        units_g1=g1,
        units_g2=g2,
        all_cores=all_cores,
        assignment=assignment,
    )


def l1_bytes(plan: Plan, cfg: Cfg):
    in_bytes = cfg.in_units * plan.kw * TILE * plan.slot_bytes
    rows_bytes = cfg.rows_pages * plan.kw * TILE_BYTES
    tiles_bytes = plan.kw * plan.ct * TILE_BYTES if cfg.variant == "two_pass" else 0
    return in_bytes + rows_bytes + tiles_bytes


def create_program_descriptor(plan: Plan, cfg: Cfg, x, out) -> ttnn.ProgramDescriptor:
    N, H, W, C = plan.shape
    dtype = FMT[cfg.fmt]
    in_page_bytes = x.buffer_page_size()
    out_page_bytes = out.buffer_page_size()
    assert in_page_bytes == 2 * C and out_page_bytes == 2 * W

    out_l1 = out_l1_bytes_per_core(
        out.memory_config().buffer_type, out.buffer_num_pages(), out.buffer_aligned_page_size(), x.device()
    )
    assert l1_bytes(plan, cfg) + out_l1 <= L1_OP_BUDGET_BYTES, "L1 budget"

    def cb(index, pages, page_bytes):
        return ttnn.CBDescriptor(
            total_size=pages * page_bytes,
            core_ranges=plan.all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_bytes)],
        )

    op_reader = use_op_reader(plan, cfg, x)
    in_page_tiles = plan.kw if op_reader else 1
    cbs = [
        cb(CB_IN, cfg.in_units * plan.kw // in_page_tiles, in_page_tiles * TILE * plan.slot_bytes),
        cb(CB_ROWS, cfg.rows_pages, plan.kw * TILE_BYTES),
    ]
    if cfg.variant == "two_pass":
        cbs.append(cb(CB_TILES, plan.kw * plan.ct, TILE_BYTES))

    defines = [(VARIANT_DEFINE[cfg.variant], "1")]
    if cfg.zones:
        defines.append(("TTX_ZONES", "1"))
    defines += [(d, "1") for d in cfg.ablate]

    reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for core, start, count in plan.assignment:
        nh, wb = divmod(start, plan.units_per_row)
        n, h = divmod(nh, H)
        first_page = nh * W + wb * plan.block_w
        reader_rt[core.x][core.y] = [first_page, count, wb]
        compute_rt[core.x][core.y] = [count, wb]
        writer_rt[core.x][core.y] = [count, wb, h, n * C * H + h]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "tt_reader.cpp"),
        core_ranges=plan.all_cores,
        compile_time_args=[
            CB_IN,
            plan.slot_bytes,
            in_page_bytes,
            plan.kw,
            plan.units_per_row,
            plan.block_w,
            plan.tail_w,
            cfg.push_tiles,
            cfg.first_push_tiles,
            int(cfg.use_trid),
            cfg.in_units * plan.kw,
            *ttnn.TensorAccessorArgs(x).get_compile_time_args(),
        ],
        runtime_args=reader_rt,
        common_runtime_args=[x.buffer_address()],
        defines=defines,
        config=ttnn.ReaderConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "tt_compute.cpp"),
        core_ranges=plan.all_cores,
        compile_time_args=[CB_IN, CB_ROWS, CB_TILES, plan.ct, plan.kw, plan.kw_tail, plan.units_per_row, in_page_tiles],
        runtime_args=compute_rt,
        defines=defines,
        # Values only move: fidelity and approximation are moot. 16-bit half-sync DEST holds the
        # unit's kw <= 8 tiles.
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=False, dst_full_sync_en=False, math_approx_mode=False),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "tt_writer.cpp"),
        core_ranges=plan.all_cores,
        compile_time_args=[
            CB_ROWS,
            C,
            plan.ct,
            H,
            plan.units_per_row,
            plan.block_w,
            plan.tail_w,
            plan.kw,
            plan.kw_tail,
            out_page_bytes,
            *ttnn.TensorAccessorArgs(out).get_compile_time_args(),
        ],
        runtime_args=writer_rt,
        common_runtime_args=[out.buffer_address()],
        defines=defines,
        config=ttnn.WriterConfigDescriptor(),
    )
    if op_reader:
        rp = _op_read_plan(plan, x)
        starts = []
        for core, start, count in plan.assignment:
            nh, wb = divmod(start, plan.units_per_row)
            starts.append((core, nh * W + wb * plan.block_w, count, wb))
        reader = rm_gather.reader_descriptor(rp, x, 1, plan.block_w, plan.units_per_row, plan.tail_w, starts)
        writer.config = rm_gather.dm_config(
            rp, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1, ttnn.WriterConfigDescriptor
        )
    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)


def use_op_reader(plan: Plan, cfg: Cfg, x):
    """The op's reader lands pages at their bank stride: usable when that equals the tilize row."""
    wanted = cfg.reader == "op" or (cfg.reader == "auto" and x.memory_config().buffer_type == ttnn.BufferType.DRAM)
    return wanted and x.buffer_aligned_page_size() == plan.slot_bytes


@dataclass
class _OpReadPlan:
    in_page_bytes: int
    slot_bytes: int
    read_alt_share: Fraction
    num_banks: int
    bank_major: bool
    chunk_pages: int
    bank_stagger: bool
    all_cores: object


def _op_read_plan(plan: Plan, x):
    """The op's nhwc_rm_transpose read knobs (2-NoC split and bank-major walk for DRAM input)."""
    from ttnn.operations.g6_permute import g6_permute_nhwc_rm_transpose as nhwc

    alt, bank_major, num_banks, chunk = rm_gather.read_plan(
        x, plan.block_w, nhwc.READ_ALT_NOC_SHARE, nhwc.BANK_MAJOR, nhwc.BANK_RUN_ROUNDS
    )
    return _OpReadPlan(
        x.buffer_page_size(), plan.slot_bytes, alt, num_banks, bank_major, chunk, nhwc.BANK_STAGGER, plan.all_cores
    )


_CACHE: dict = {}


def permute(x, cfg: Cfg):
    """out = x.permute(0, 3, 1, 2) through the Tensix-transpose program (cached per shape and cfg)."""
    N, H, W, C = (int(d) for d in x.shape)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([N, C, H, W]), x.dtype, x.layout, x.device(), x.memory_config())
    key = (tuple(x.shape), x.memory_config().buffer_type, cfg.key())
    entry = _CACHE.get(key)
    if entry is None:
        plan = build_plan(x, cfg)
        desc = create_program_descriptor(plan, cfg, x, out)
        _CACHE[key] = (plan, desc)
    else:
        plan, desc = entry
        desc.kernels[0].common_runtime_args = [x.buffer_address()]
        desc.kernels[1].common_runtime_args = [out.buffer_address()]
    return ttnn.generic_op([x, out], desc)
