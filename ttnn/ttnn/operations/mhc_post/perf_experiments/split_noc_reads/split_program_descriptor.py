# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""split_noc_reads bench — copy of the op's ProgramDescriptor (baseline, `create_program_descriptor`, kernels
copied verbatim into ./kernels) plus the candidate `create_split_program_descriptor` (reads / writes shared
across both DM RISCs, see kernels/split_dm.cpp). Stubs for ablations: `skip_compute` (eltwise_chain
SKIP_COMPUTE), `skip_expand` (coefficient expansion stores dropped, CB handshake kept).

Original docstring: mhc_post — ProgramDescriptor (regime `flat_stream`, see op_design.md Blocking Model).

Flattened (token-tile row r, column tile c) units, r-major, split contiguously over the full compute
grid (`split_work_to_cores(..., row_wise=True)`). Per core: segments (one token row each), blocks of
`block_col_tiles` columns. Every block knob below is defined once; CB sizes, CT/RT args derive from it.
"""

from __future__ import annotations

import math
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_HW = 32

# ---- CB slots (semantic names) ----
CB_SUBLAYER_TILES = 0  # F block tiles              reader  -> compute
CB_RESIDUAL_TILES = 1  # X block tiles (n streams)  reader  -> compute
CB_COEF_RAW = 2  # raw post / comb tiles of one token row, private scratch of the COEF_EXPANDER kernel
CB_COEF_BCAST = 3  # n * P half-packed column-broadcast fp32 coefficient tiles, COEF_EXPANDER -> compute
CB_OUTPUT_TILES = 16  # X' block tiles (n streams) compute -> writer

# ---- Block-model knobs (single source of truth) ----
BLOCK_TOKEN_TILES = 1  # a block never spans two token rows (realized by segments)
DEPTH_IN = 2  # blocks in flight on cb_sublayer_tiles / cb_residual_tiles
DEPTH_OUT = 2  # blocks in flight on cb_output_tiles
COEF_DEPTH = 2  # token-row coefficient sets in flight on cb_coef_bcast
L1_BUDGET_BYTES = 1 << 20  # CB budget per core
# Block-size policy: block_col_tiles = min(L1 fit, longest segment, MAX_BLOCK_COL_TILES,
# ceil(max units per core / MIN_BLOCKS_PER_CORE)). The coarsest L1 fit leaves a core with 1-2 blocks, so read,
# mix and write barely overlap and every core bursts its whole prefetch at DRAM at once. Measured (Blackhole,
# 110 cores, bf16): cap 8 is the best cap at T640 C7168 / T1280 C4096, and >= 3 blocks per core is best at
# T640 C1792 (10-11 units per core -> B = 4). Deeper DEPTH_IN (3) measured slower.
MAX_BLOCK_COL_TILES = 8  # cap on block_col_tiles (None = no cap)
MIN_BLOCKS_PER_CORE = 3  # blocks the busiest core's range is cut into, at least (1 = no constraint)
# DEST sync mode of the compute kernel (overrides the caller's dst_full_sync_en; an internal schedule
# choice). SyncFull gives 8 fp32 DEST slots, enough for one output column's P coefficient tiles + n+1
# data tiles at n = 4; the kernel derives its window layout from DEST_AUTO_LIMIT. False (SyncHalf, 4
# slots) compiles only for n <= 2 (the kernel static_asserts the fit).
DST_FULL_SYNC = True
# Which DM kernel runs load_coefficients (raw post / comb read + column-broadcast expansion) and is therefore
# the single producer of cb_coef_bcast: "writer" (BRISC, idle until the first output block; keeps the reader a
# pure stream) or "reader" (NCRISC, expansion in the read shadow of block 1).
COEF_EXPANDER = "writer"
NUM_CIRCULAR_BUFFERS = 64  # length of ComputeConfigDescriptor.unpack_to_dest_mode

assert COEF_EXPANDER in ("reader", "writer")
# The writer loads segment s+1's set while compute still holds segment s's: two sets in flight.
assert COEF_EXPANDER == "reader" or COEF_DEPTH >= 2, "writer-side expansion needs COEF_DEPTH >= 2"
assert BLOCK_TOKEN_TILES == 1, "flat_stream realizes block_token_tiles through segments; only 1 is built"


SKIP_NOC = False  # module knob (set by the bench): drop the data NoC transfers, keep CB handshakes


def _stub_defines(skip_compute, skip_expand):
    compute_defines = [("CKL_ELTWISE_CHAIN_SKIP_COMPUTE", "1")] if skip_compute else []
    dm_defines = [("MHC_SKIP_EXPAND", "1")] if skip_expand else []
    if SKIP_NOC:
        dm_defines.append(("MHC_SKIP_NOC", "1"))
    return compute_defines, dm_defines


def _tensor_token_tiles(shape) -> int:
    lead = 1
    for d in shape[:-2]:
        lead *= d
    return lead * math.ceil(shape[-2] / TILE_HW)


def _work_assignment(grid_size, total_units):
    """Contiguous flattened split: [(core, start_unit, num_units), ...] in split order."""
    (
        _,
        all_cores,
        core_group_1,
        core_group_2,
        units_per_core_g1,
        units_per_core_g2,
    ) = ttnn.split_work_to_cores(grid_size, total_units, row_wise=True)
    assignment = []
    start = 0
    for group, per_core in ((core_group_1, units_per_core_g1), (core_group_2, units_per_core_g2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, start, per_core))
            start += per_core
    assert start == total_units
    return all_cores, assignment


def _max_segment_col_tiles(assignment, col_tiles_per_row) -> int:
    longest = 0
    for _, start, count in assignment:
        u, left = start, count
        while left:
            seg = min(col_tiles_per_row - u % col_tiles_per_row, left)
            longest = max(longest, seg)
            u += seg
            left -= seg
    return longest


def _block_col_tiles_fit(
    n, sublayer_tile_bytes, residual_tile_bytes, coef_tile_bytes, num_coef_tiles, num_raw_tiles
) -> int:
    """Closed form from l1_ledger.md: coarsest block that fits L1_BUDGET_BYTES."""
    coef_bytes = COEF_DEPTH * num_coef_tiles * coef_tile_bytes + num_raw_tiles * coef_tile_bytes
    per_col = DEPTH_IN * (sublayer_tile_bytes + n * residual_tile_bytes) + DEPTH_OUT * n * residual_tile_bytes
    return (L1_BUDGET_BYTES - coef_bytes) // per_col


def _cb(index, dtype, page_bytes, num_pages, cores):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_bytes,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_bytes)],
    )


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    residual: ttnn.Tensor,
    post: ttnn.Tensor,
    comb: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    compute_kernel_config: ttnn.ComputeConfigDescriptor,
    skip_compute: bool = False,
    skip_expand: bool = False,
) -> ttnn.ProgramDescriptor:
    compute_defines, dm_defines = _stub_defines(skip_compute, skip_expand)
    device = input_tensor.device()

    n = post.shape[-1]
    col_tiles_per_row = input_tensor.shape[-1] // TILE_HW  # Ct
    token_tiles = _tensor_token_tiles(list(input_tensor.padded_shape))
    total_units = token_tiles * col_tiles_per_row
    post_tiles_per_row = math.ceil(n / TILE_HW)
    comb_tiles_per_row = math.ceil(n * n / TILE_HW)
    num_raw_tiles = post_tiles_per_row + comb_tiles_per_row
    coef_tiles_per_stream = math.ceil((n + 1) / 2)  # two coefficient terms per tile (mhc_post_common.hpp)
    num_coef_tiles = n * coef_tiles_per_stream

    sublayer_page = input_tensor.buffer_page_size()
    residual_page = residual.buffer_page_size()
    output_page = output_tensor.buffer_page_size()
    coef_page = post.buffer_page_size()
    assert comb.buffer_page_size() == coef_page
    assert output_page == residual_page

    # ---- work split + block size ----
    grid_size = device.compute_with_storage_grid_size()
    all_cores, assignment = _work_assignment(grid_size, total_units)
    block_col_tiles_fit = _block_col_tiles_fit(
        n, sublayer_page, residual_page, coef_page, num_coef_tiles, num_raw_tiles
    )
    assert block_col_tiles_fit >= 1, "mhc_post: coefficient set + one column block does not fit L1_BUDGET_BYTES"
    block_col_tiles = min(block_col_tiles_fit, _max_segment_col_tiles(assignment, col_tiles_per_row))
    max_units_per_core = max(count for _, _, count in assignment)
    block_col_tiles = min(block_col_tiles, math.ceil(max_units_per_core / MIN_BLOCKS_PER_CORE))
    if MAX_BLOCK_COL_TILES is not None:
        block_col_tiles = min(block_col_tiles, MAX_BLOCK_COL_TILES)
    block_col_tiles = max(1, block_col_tiles)

    # ---- circular buffers ----
    cbs = [
        _cb(CB_SUBLAYER_TILES, input_tensor.dtype, sublayer_page, DEPTH_IN * block_col_tiles, all_cores),
        _cb(CB_RESIDUAL_TILES, residual.dtype, residual_page, DEPTH_IN * n * block_col_tiles, all_cores),
        _cb(CB_COEF_RAW, post.dtype, coef_page, num_raw_tiles, all_cores),
        _cb(CB_COEF_BCAST, post.dtype, coef_page, COEF_DEPTH * num_coef_tiles, all_cores),
        _cb(CB_OUTPUT_TILES, output_tensor.dtype, output_page, DEPTH_OUT * n * block_col_tiles, all_cores),
    ]

    # ---- reader ----
    reader_ct = [
        n,
        col_tiles_per_row,
        block_col_tiles,
        post_tiles_per_row,
        comb_tiles_per_row,
        sublayer_page,
        residual_page,
        coef_page,
        TILE_HW,
        CB_SUBLAYER_TILES,
        CB_RESIDUAL_TILES,
        CB_COEF_RAW,
        CB_COEF_BCAST,
        coef_tiles_per_stream,
        int(COEF_EXPANDER == "reader"),
    ]
    for t in (input_tensor, residual, post, comb):
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    # ---- writer ----
    writer_ct = [
        n,
        col_tiles_per_row,
        block_col_tiles,
        output_page,
        CB_OUTPUT_TILES,
        post_tiles_per_row,
        comb_tiles_per_row,
        coef_page,
        TILE_HW,
        CB_COEF_RAW,
        CB_COEF_BCAST,
        coef_tiles_per_stream,
        int(COEF_EXPANDER == "writer"),
    ]
    for t in (output_tensor, post, comb):
        writer_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    # ---- compute ----
    compute_ct = [
        n,
        col_tiles_per_row,
        block_col_tiles,
        CB_SUBLAYER_TILES,
        CB_RESIDUAL_TILES,
        CB_COEF_BCAST,
        CB_OUTPUT_TILES,
        coef_tiles_per_stream,
    ]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    f_addr, x_addr = input_tensor.buffer_address(), residual.buffer_address()
    p_addr, m_addr = post.buffer_address(), comb.buffer_address()
    o_addr = output_tensor.buffer_address()
    for core, start, count in assignment:
        reader_rt[core.x][core.y] = [f_addr, x_addr, p_addr, m_addr, start, count]
        writer_rt[core.x][core.y] = [o_addr, start, count, p_addr, m_addr]
        compute_rt[core.x][core.y] = [start, count]

    # UnpackToDestFp32 on every Float32 CB compute reads with copy_tile (derived from the CB format).
    unpack_modes = [ttnn.UnpackToDestMode.Default] * NUM_CIRCULAR_BUFFERS
    for index, dtype in (
        (CB_SUBLAYER_TILES, input_tensor.dtype),
        (CB_RESIDUAL_TILES, residual.dtype),
        (CB_COEF_BCAST, post.dtype),
    ):
        if dtype == ttnn.float32:
            unpack_modes[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute_cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=compute_kernel_config.math_fidelity,
        fp32_dest_acc_en=compute_kernel_config.fp32_dest_acc_en,
        math_approx_mode=compute_kernel_config.math_approx_mode,
    )
    compute_cfg.dst_full_sync_en = DST_FULL_SYNC
    compute_cfg.unpack_to_dest_mode = unpack_modes

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "mhc_post_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        defines=dm_defines,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "mhc_post_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        defines=dm_defines,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "mhc_post_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        defines=compute_defines,
        config=compute_cfg,
    )
    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)


# =====================================================================================================
# Candidate: split_noc_reads
# =====================================================================================================
CB_RESIDUAL_B = 4  # X streams [NA, n)       BRISC-side reader -> compute
CB_OUTPUT_B = 17  # X' streams [MA, n)       compute -> NCRISC-side writer


class SplitConfig:
    """How the DM traffic is shared between the two DM RISCs.

    f_on        : "ncrisc" | "brisc"   which RISC reads F
    x_brisc     : number of X streams (the LAST ones, [n - x_brisc, n)) read by BRISC; the rest by NCRISC
    out_ncrisc  : number of X' streams (the LAST ones) written by NCRISC; the rest by BRISC
    ncrisc_noc / brisc_noc : NoC index of each RISC's kernel (dedicated-NoC mode)
    """

    # Alternate-NoC routing (DM_DYNAMIC_NOC): each RISC may send some of its traffic on the other NoC.
    #   ncrisc_f_alt / brisc_f_alt : F reads on the other NoC
    #   ncrisc_x_alt / brisc_x_alt : the LAST k of that RISC's X streams on the other NoC
    #   ncrisc_w_alt / brisc_w_alt : the LAST k of that RISC's output streams on the other NoC
    def __init__(
        self,
        f_on="ncrisc",
        x_brisc=0,
        out_ncrisc=0,
        ncrisc_noc=0,
        brisc_noc=1,
        dynamic=False,
        ncrisc_f_alt=False,
        brisc_f_alt=False,
        ncrisc_x_alt=0,
        brisc_x_alt=0,
        ncrisc_w_alt=0,
        brisc_w_alt=0,
        sched=0,
    ):
        self.sched = sched  # read+write RISC schedule: 0 lag-1, 1 event loop (reads first), 2 (writes first)
        assert f_on in ("ncrisc", "brisc")
        self.f_on, self.x_brisc, self.out_ncrisc = f_on, x_brisc, out_ncrisc
        self.ncrisc_noc, self.brisc_noc = ncrisc_noc, brisc_noc
        self.ncrisc_f_alt, self.brisc_f_alt = ncrisc_f_alt, brisc_f_alt
        self.ncrisc_x_alt, self.brisc_x_alt = ncrisc_x_alt, brisc_x_alt
        self.ncrisc_w_alt, self.brisc_w_alt = ncrisc_w_alt, brisc_w_alt
        uses_alt = ncrisc_f_alt or brisc_f_alt or ncrisc_x_alt or brisc_x_alt or ncrisc_w_alt or brisc_w_alt
        self.dynamic = dynamic or bool(uses_alt)

    def __repr__(self):
        return f"SplitConfig({self.__dict__})"


def create_split_program_descriptor(
    input_tensor,
    residual,
    post,
    comb,
    output_tensor,
    compute_kernel_config,
    split: SplitConfig,
    skip_compute: bool = False,
    skip_expand: bool = False,
) -> ttnn.ProgramDescriptor:
    compute_defines, dm_defines = _stub_defines(skip_compute, skip_expand)
    device = input_tensor.device()

    n = post.shape[-1]
    x_brisc = min(split.x_brisc, n)
    out_ncrisc = min(split.out_ncrisc, n)
    NA = n - x_brisc  # X streams in CB_RESIDUAL_TILES (NCRISC)
    MA = n - out_ncrisc  # X' streams in CB_OUTPUT_TILES (BRISC)
    col_tiles_per_row = input_tensor.shape[-1] // TILE_HW
    token_tiles = _tensor_token_tiles(list(input_tensor.padded_shape))
    total_units = token_tiles * col_tiles_per_row
    post_tiles_per_row = math.ceil(n / TILE_HW)
    comb_tiles_per_row = math.ceil(n * n / TILE_HW)
    num_raw_tiles = post_tiles_per_row + comb_tiles_per_row
    coef_tiles_per_stream = math.ceil((n + 1) / 2)
    num_coef_tiles = n * coef_tiles_per_stream

    sublayer_page = input_tensor.buffer_page_size()
    residual_page = residual.buffer_page_size()
    output_page = output_tensor.buffer_page_size()
    coef_page = post.buffer_page_size()
    assert output_page == residual_page

    # ---- work split + block size: identical policy to the baseline ----
    grid_size = device.compute_with_storage_grid_size()
    all_cores, assignment = _work_assignment(grid_size, total_units)
    block_col_tiles_fit = _block_col_tiles_fit(
        n, sublayer_page, residual_page, coef_page, num_coef_tiles, num_raw_tiles
    )
    assert block_col_tiles_fit >= 1
    block_col_tiles = min(block_col_tiles_fit, _max_segment_col_tiles(assignment, col_tiles_per_row))
    max_units_per_core = max(count for _, _, count in assignment)
    block_col_tiles = min(block_col_tiles, math.ceil(max_units_per_core / MIN_BLOCKS_PER_CORE))
    if MAX_BLOCK_COL_TILES is not None:
        block_col_tiles = min(block_col_tiles, MAX_BLOCK_COL_TILES)
    block_col_tiles = max(1, block_col_tiles)
    B = block_col_tiles

    cbs = [
        _cb(CB_SUBLAYER_TILES, input_tensor.dtype, sublayer_page, DEPTH_IN * B, all_cores),
        _cb(CB_COEF_RAW, post.dtype, coef_page, num_raw_tiles, all_cores),
        _cb(CB_COEF_BCAST, post.dtype, coef_page, COEF_DEPTH * num_coef_tiles, all_cores),
    ]
    if NA > 0:
        cbs.append(_cb(CB_RESIDUAL_TILES, residual.dtype, residual_page, DEPTH_IN * NA * B, all_cores))
    if NA < n:
        cbs.append(_cb(CB_RESIDUAL_B, residual.dtype, residual_page, DEPTH_IN * (n - NA) * B, all_cores))
    if MA > 0:
        cbs.append(_cb(CB_OUTPUT_TILES, output_tensor.dtype, output_page, DEPTH_OUT * MA * B, all_cores))
    if MA < n:
        cbs.append(_cb(CB_OUTPUT_B, output_tensor.dtype, output_page, DEPTH_OUT * (n - MA) * B, all_cores))

    accessor_ct = []
    for t in (input_tensor, residual, output_tensor, post, comb):
        accessor_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    def dm_ct(read_f, x_lo, x_hi, w_lo, w_hi, expand, cb_x, cb_out, f_alt, x_alt, w_alt):
        x_alt_from = x_hi - min(x_alt, x_hi - x_lo)
        w_alt_from = w_hi - min(w_alt, w_hi - w_lo)
        return [
            n,
            col_tiles_per_row,
            B,
            int(read_f),
            x_lo,
            x_hi,
            w_lo,
            w_hi,
            int(expand),
            CB_SUBLAYER_TILES,
            cb_x,
            cb_out,
            CB_COEF_RAW,
            CB_COEF_BCAST,
            post_tiles_per_row,
            comb_tiles_per_row,
            sublayer_page,
            residual_page,
            coef_page,
            coef_tiles_per_stream,
            int(f_alt),
            x_alt_from,
            w_alt_from,
            split.sched,
        ] + accessor_ct

    # BRISC keeps the coefficient expansion (it is the op's COEF_EXPANDER=writer) as long as it writes.
    brisc_expands = MA > 0
    s = split
    ncrisc_ct = dm_ct(
        s.f_on == "ncrisc",
        0,
        NA,
        MA,
        n,
        not brisc_expands,
        CB_RESIDUAL_TILES,
        CB_OUTPUT_B,
        s.ncrisc_f_alt,
        s.ncrisc_x_alt,
        s.ncrisc_w_alt,
    )
    brisc_ct = dm_ct(
        s.f_on == "brisc",
        NA,
        n,
        0,
        MA,
        brisc_expands,
        CB_RESIDUAL_B,
        CB_OUTPUT_TILES,
        s.brisc_f_alt,
        s.brisc_x_alt,
        s.brisc_w_alt,
    )
    noc_mode = ttnn.NOC_MODE.DM_DYNAMIC_NOC if s.dynamic else ttnn.NOC_MODE.DM_DEDICATED_NOC

    compute_ct = [
        n,
        col_tiles_per_row,
        B,
        CB_SUBLAYER_TILES,
        CB_RESIDUAL_TILES,
        CB_COEF_BCAST,
        CB_OUTPUT_TILES,
        coef_tiles_per_stream,
        CB_RESIDUAL_B,
        NA,
        CB_OUTPUT_B,
        MA,
    ]

    dm_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    addrs = [
        input_tensor.buffer_address(),
        residual.buffer_address(),
        output_tensor.buffer_address(),
        post.buffer_address(),
        comb.buffer_address(),
    ]
    for core, start, count in assignment:
        dm_rt[core.x][core.y] = addrs + [start, count]
        compute_rt[core.x][core.y] = [start, count]

    unpack_modes = [ttnn.UnpackToDestMode.Default] * NUM_CIRCULAR_BUFFERS
    for index, dtype in (
        (CB_SUBLAYER_TILES, input_tensor.dtype),
        (CB_RESIDUAL_TILES, residual.dtype),
        (CB_RESIDUAL_B, residual.dtype),
        (CB_COEF_BCAST, post.dtype),
    ):
        if dtype == ttnn.float32:
            unpack_modes[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute_cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=compute_kernel_config.math_fidelity,
        fp32_dest_acc_en=compute_kernel_config.fp32_dest_acc_en,
        math_approx_mode=compute_kernel_config.math_approx_mode,
    )
    compute_cfg.dst_full_sync_en = DST_FULL_SYNC
    compute_cfg.unpack_to_dest_mode = unpack_modes

    noc = {0: ttnn.NOC.NOC_0, 1: ttnn.NOC.NOC_1}
    dm_src = str(KERNEL_DIR / "split_dm.cpp")
    ncrisc = ttnn.KernelDescriptor(
        kernel_source=dm_src,
        core_ranges=all_cores,
        compile_time_args=ncrisc_ct,
        runtime_args=dm_rt,
        defines=dm_defines,
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_1, noc=noc[split.ncrisc_noc], noc_mode=noc_mode
        ),
    )
    brisc = ttnn.KernelDescriptor(
        kernel_source=dm_src,
        core_ranges=all_cores,
        compile_time_args=brisc_ct,
        runtime_args=dm_rt,
        defines=dm_defines,
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0, noc=noc[split.brisc_noc], noc_mode=noc_mode
        ),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "split_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        defines=compute_defines,
        config=compute_cfg,
    )
    return ttnn.ProgramDescriptor(kernels=[ncrisc, brisc, compute], semaphores=[], cbs=cbs)


# =====================================================================================================
# Candidate: helper DMA (CB topology + compute kernel identical to the op; kernels/helper_dm.cpp)
# =====================================================================================================
class HelperConfig:
    """x_help: last k X streams read by BRISC into the reader's reserved window; f_help: F read by BRISC;
    w_help: last m output streams written by NCRISC from the writer's front window;
    brisc_help_alt / ncrisc_help_alt: that RISC's HELP traffic on the other NoC (dynamic-NoC mode)."""

    def __init__(
        self,
        x_help=0,
        f_help=False,
        w_help=0,
        brisc_help_alt=False,
        ncrisc_help_alt=False,
        dynamic=False,
        help_from=0,
        coef_incremental=False,
    ):
        self.help_from, self.coef_incremental = help_from, coef_incremental
        self.x_help, self.f_help, self.w_help = x_help, f_help, w_help
        self.brisc_help_alt, self.ncrisc_help_alt = brisc_help_alt, ncrisc_help_alt
        self.dynamic = dynamic or brisc_help_alt or ncrisc_help_alt

    def __repr__(self):
        return f"HelperConfig({self.__dict__})"


def create_helper_program_descriptor(
    input_tensor,
    residual,
    post,
    comb,
    output_tensor,
    compute_kernel_config,
    helper: HelperConfig,
    skip_compute=False,
    skip_expand=False,
):
    # Start from the baseline descriptor (same CBs, same compute kernel + args, same work split / block size),
    # then swap the two DM kernels for helper_dm.cpp.
    base = create_program_descriptor(
        input_tensor, residual, post, comb, output_tensor, compute_kernel_config, skip_compute, skip_expand
    )
    reader_k, writer_k, compute_k = base.kernels
    _, dm_defines = _stub_defines(skip_compute, skip_expand)
    n = post.shape[-1]
    B = reader_k.compile_time_args[2]
    col_tiles_per_row = reader_k.compile_time_args[1]
    post_tiles_per_row = math.ceil(n / TILE_HW)
    comb_tiles_per_row = math.ceil(n * n / TILE_HW)
    coef_tiles_per_stream = math.ceil((n + 1) / 2)
    x_help_lo = n - min(helper.x_help, n)
    w_help_lo = n - min(helper.w_help, n)
    all_cores = reader_k.core_ranges

    accessor_ct = []
    for t in (input_tensor, residual, output_tensor, post, comb):
        accessor_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    SEM = [0, 1, 2, 3]  # rd_go, rd_done, wr_go, wr_done

    def ct(role, help_alt):
        return (
            [
                n,
                col_tiles_per_row,
                B,
                role,
                x_help_lo,
                int(helper.f_help),
                w_help_lo,
                int(help_alt),
                DEPTH_IN,
                DEPTH_OUT,
                CB_SUBLAYER_TILES,
                CB_RESIDUAL_TILES,
                CB_OUTPUT_TILES,
                CB_COEF_RAW,
                CB_COEF_BCAST,
                post_tiles_per_row,
                comb_tiles_per_row,
                input_tensor.buffer_page_size(),
                residual.buffer_page_size(),
                post.buffer_page_size(),
                coef_tiles_per_stream,
            ]
            + SEM
            + [helper.help_from, int(helper.coef_incremental)]
            + accessor_ct
        )

    rt = ttnn.RuntimeArgs()
    addrs = [
        input_tensor.buffer_address(),
        residual.buffer_address(),
        output_tensor.buffer_address(),
        post.buffer_address(),
        comb.buffer_address(),
    ]
    _, assignment = _work_assignment(
        input_tensor.device().compute_with_storage_grid_size(),
        _tensor_token_tiles(list(input_tensor.padded_shape)) * col_tiles_per_row,
    )
    for core, start, count in assignment:
        rt[core.x][core.y] = addrs + [start, count]
    mode = ttnn.NOC_MODE.DM_DYNAMIC_NOC if helper.dynamic else ttnn.NOC_MODE.DM_DEDICATED_NOC
    src = str(KERNEL_DIR / "helper_dm.cpp")
    ncrisc = ttnn.KernelDescriptor(
        kernel_source=src,
        core_ranges=all_cores,
        compile_time_args=ct(0, helper.ncrisc_help_alt),
        runtime_args=rt,
        defines=dm_defines,
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0, noc_mode=mode
        ),
    )
    brisc = ttnn.KernelDescriptor(
        kernel_source=src,
        core_ranges=all_cores,
        compile_time_args=ct(1, helper.brisc_help_alt),
        runtime_args=rt,
        defines=dm_defines,
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1, noc_mode=mode
        ),
    )
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=all_cores, initial_value=0) for i in SEM]
    return ttnn.ProgramDescriptor(kernels=[ncrisc, brisc, compute_k], semaphores=sems, cbs=list(base.cbs))
