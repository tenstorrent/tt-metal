# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""riscv_transpose bake-off: ProgramDescriptors of nhwc_rm_transpose variants (who transposes, and how).

The plan (units, block split, core split) is the op's own nhwc_rm_transpose.build_plan, with its module
knobs BLOCK_PAGES / INPUT_DEPTH set per variant. Variants:
  op      the op's descriptor and kernels, untouched (the honest baseline).
  solo    the op's structure (NCRISC reads, BRISC transposes and writes), transpose impl chosen
          (rvt_transpose.hpp): IMPL_BASE batch 1 is the op's own writer code.
  shared  NCRISC reads; the transpose of each unit is split by row items over TRISC 0/1/2 and optionally
          BRISC (weights per participant); BRISC writes. rows_depth rows generations.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path

import ttnn
from ttnn.operations.g6_permute import g6_permute_nhwc_rm_transpose as nhwc
from ttnn.operations.g6_permute.g6_permute_blocks_rm_gather import (
    CB_INPUT_SLOTS,
    CB_OUTPUT_ROWS,
    dm_config,
    reader_descriptor,
)

KDIR = Path(__file__).resolve().parent / "kernels"
DIMS = (0, 3, 1, 2)
# Counter CBs (only their tiles_received stream registers are used): free counter, done counters of
# TRISC 0, 1, 2.
CB_FREE = 8
CB_DONE0 = 9
# S1 read split: BRISC's read counter.
CB_BRD = 13
COUNTER_CBS = [CB_FREE, CB_DONE0, CB_DONE0 + 1, CB_DONE0 + 2, CB_DONE0 + 3, CB_BRD]

IMPL_BASE, IMPL_LINE, IMPL_HALF, IMPL_BLK, IMPL_BLKV, IMPL_BLKS = range(6)
OptLevel = ttnn._ttnn.program_spec.KernelBuildOptLevel


@dataclass
class Variant:
    mode: str  # "op" | "solo" | "shared"
    impl: int = IMPL_BASE
    batch: int = 1
    block_pages: int = 256
    input_depth: int = 2
    rows_depth: int = 1
    # shared: weight per participant ("T0", "T1", "T2", "B")
    shares: dict = field(default_factory=dict)
    dm_opt: object = None
    # S1: BRISC reads the last width * num / den pages of every unit (None: the reader reads all).
    brisc_read: tuple = None
    # S1: block 0's own split (pipeline fill), None: brisc_read.
    first_read: tuple = None
    # shared: helpers post per row item and the writer streams rows as they complete.
    stream: bool = False


@contextlib.contextmanager
def _knobs(variant: Variant, block_w=None):
    saved = (nhwc.BLOCK_PAGES, nhwc.INPUT_DEPTH, nhwc.BLOCK_W)
    nhwc.BLOCK_PAGES, nhwc.INPUT_DEPTH = variant.block_pages, variant.input_depth
    nhwc.BLOCK_W = block_w
    try:
        yield
    finally:
        nhwc.BLOCK_PAGES, nhwc.INPUT_DEPTH, nhwc.BLOCK_W = saved


def build_plan(variant: Variant, input_tensor, block_w=None):
    """The op's plan with the variant's block knobs; block_w forces the unit width (the op's BLOCK_W)."""
    with _knobs(variant, block_w):
        return nhwc.build_plan(input_tensor, DIMS)


def share_bounds(variant: Variant, C):
    """{participant: (lo, hi)} row-item bounds of the shared split."""
    items = (C + 1) // 2
    order = ["T0", "T1", "T2", "N", "B"]
    total = sum(variant.shares.get(p, 0) for p in order)
    out, cum = {}, 0
    for p in order:
        lo = items * cum // total
        cum += variant.shares.get(p, 0)
        out[p] = (lo, items * cum // total)
    return out


def _with_opt(kd, opt):
    """The same kernel built at optimization level `opt` (opt_level is a constructor-only field)."""
    return ttnn.KernelDescriptor(
        kernel_source=kd.kernel_source,
        core_ranges=kd.core_ranges,
        compile_time_args=kd.compile_time_args,
        defines=kd.defines,
        runtime_args=kd.runtime_args,
        common_runtime_args=kd.common_runtime_args,
        opt_level=opt,
        config=kd.config,
    )


def create(variant: Variant, plan, input_tensor, output_tensor) -> ttnn.ProgramDescriptor:
    if variant.mode == "op":
        return nhwc.create_program_descriptor(plan, input_tensor, output_tensor)
    _, H, W, C = plan.shape
    unit_slot_bytes = plan.block_w * plan.slot_bytes
    unit_row_bytes = C * plan.row_bytes
    rows_depth = variant.rows_depth if variant.mode == "shared" else 1

    def cb(index, total, page):
        return ttnn.CBDescriptor(
            total_size=total,
            core_ranges=plan.all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=input_tensor.dtype, page_size=page)
            ],
        )

    cbs = [
        cb(CB_INPUT_SLOTS, plan.input_depth * plan.block_units * unit_slot_bytes, unit_slot_bytes),
        cb(CB_OUTPUT_ROWS, rows_depth * plan.block_units * unit_row_bytes, unit_row_bytes),
    ]
    if variant.mode == "shared":
        cbs += [cb(i, 16, 16) for i in COUNTER_CBS]

    starts = [(core, *nhwc._unit_start(plan, start), count) for core, start, count in plan.assignment]
    reader = reader_descriptor(
        plan,
        input_tensor,
        plan.block_units,
        plan.block_w,
        plan.units_per_row,
        plan.tail_w,
        [(core, first_page, count, cursor[0]) for core, first_page, cursor, count in starts],
    )
    reader.kernel_source = str(KDIR / "rvt_reader.cpp")
    if variant.brisc_read is not None:
        num, den = variant.brisc_read
        reader.defines = [("RVT_SPLIT_NUM", str(num)), ("RVT_SPLIT_DEN", str(den)), ("RVT_CB_BRD", str(CB_BRD))]
        if variant.first_read is not None:
            reader.defines = reader.defines + [
                ("RVT_SPLIT0_NUM", str(variant.first_read[0])),
                ("RVT_SPLIT0_DEN", str(variant.first_read[1])),
            ]

    has_trisc = variant.mode == "shared" and any(variant.shares.get(t, 0) for t in ("T0", "T1", "T2"))
    if variant.mode == "shared":
        bounds = share_bounds(variant, C)
        own_lo, own_hi = bounds["B"]
        n_helpers = 3
        helper_mask = (7 if has_trisc else 0) | (8 if variant.shares.get("N", 0) else 0)
        if variant.shares.get("N", 0):
            assert variant.brisc_read is None, "NCRISC share uses the op's read path"
            reader.defines = list(reader.defines) + [
                ("RVT_NC_LO", str(bounds["N"][0])),
                ("RVT_NC_HI", str(bounds["N"][1])),
                ("RVT_NC_C", str(C)),
                ("RVT_NC_ROW_BYTES", str(plan.row_bytes)),
                ("RVT_NC_ROWS_DEPTH", str(rows_depth)),
                ("RVT_NC_INPUT_DEPTH", str(plan.input_depth)),
                ("RVT_CB_ROWS", str(CB_OUTPUT_ROWS)),
                ("RVT_CB_FREE", str(CB_FREE)),
                ("RVT_CB_DONE_N", str(CB_DONE0 + 3)),
            ]
    else:
        own_lo = own_hi = 0
        n_helpers = 0

    writer_ct = [
        CB_INPUT_SLOTS,
        CB_OUTPUT_ROWS,
        plan.block_units,
        H,
        C,
        plan.units_per_row,
        plan.block_w,
        plan.tail_w,
        plan.slot_bytes,
        plan.row_bytes,
        plan.out_page_bytes,
        variant.impl,
        variant.batch,
        plan.input_depth,
        rows_depth,
        CB_FREE,
        CB_DONE0,
        n_helpers,
        own_lo,
        own_hi,
        *ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args(),
    ]
    writer_defines = [("RVT_HELPER_MASK", str(helper_mask))] if variant.mode == "shared" else []
    writer_common = [output_tensor.buffer_address()]
    if variant.brisc_read is not None:
        num, den = variant.brisc_read
        writer_ct += ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
        writer_defines = [
            ("RVT_BRISC_READ_NUM", str(num)),
            ("RVT_BRISC_READ_DEN", str(den)),
            ("RVT_CB_BRD", str(CB_BRD)),
            ("RVT_IN_PAGE_BYTES", str(plan.in_page_bytes)),
        ]
        writer_common.append(input_tensor.buffer_address())
        if variant.first_read is not None:
            writer_defines += [
                ("RVT_SPLIT0_NUM", str(variant.first_read[0])),
                ("RVT_SPLIT0_DEN", str(variant.first_read[1])),
            ]
    if variant.stream:
        assert variant.mode == "shared" and own_lo == own_hi, "streamed writes: TRISC helpers only"
        writer_defines += [("RVT_STREAM", "1")]
        for t in range(3):
            lo, hi = bounds[f"T{t}"]
            writer_defines += [(f"RVT_LO{t}", str(lo)), (f"RVT_HI{t}", str(hi))]
    writer_rt = ttnn.RuntimeArgs()
    for core, first_page, cursor, count in starts:
        writer_rt[core.x][core.y] = [count, *cursor, first_page]
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "rvt_writer.cpp"),
        core_ranges=plan.all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        common_runtime_args=writer_common,
        defines=writer_defines,
        config=dm_config(plan, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1, ttnn.WriterConfigDescriptor),
    )
    if variant.dm_opt is not None:
        reader, writer = (_with_opt(k, variant.dm_opt) for k in (reader, writer))
    kernels = [reader, writer]

    if has_trisc:
        trisc_ct = [
            CB_INPUT_SLOTS,
            CB_OUTPUT_ROWS,
            CB_FREE,
            CB_DONE0,
            plan.block_units,
            plan.units_per_row,
            plan.block_w,
            plan.tail_w,
            C,
            plan.slot_bytes,
            plan.row_bytes,
            plan.input_depth,
            rows_depth,
            *bounds["T0"],
            *bounds["T1"],
            *bounds["T2"],
        ]
        trisc_rt = ttnn.RuntimeArgs()
        for core, _, cursor, count in starts:
            trisc_rt[core.x][core.y] = [count, cursor[0]]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KDIR / "rvt_trisc.cpp"),
                core_ranges=plan.all_cores,
                compile_time_args=trisc_ct,
                runtime_args=trisc_rt,
                defines=[("RVT_STREAM", "1")] if variant.stream else [],
                config=ttnn.ComputeConfigDescriptor(),
            )
        )
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


def patch(descriptor, variant: Variant, input_tensor, output_tensor):
    """Point a built descriptor at new buffers (reader: input address, writer: output address)."""
    descriptor.kernels[0].common_runtime_args = [input_tensor.buffer_address()]
    if variant.mode != "op" and variant.brisc_read is not None:
        descriptor.kernels[1].common_runtime_args = [output_tensor.buffer_address(), input_tensor.buffer_address()]
    else:
        descriptor.kernels[1].common_runtime_args = [output_tensor.buffer_address()]
