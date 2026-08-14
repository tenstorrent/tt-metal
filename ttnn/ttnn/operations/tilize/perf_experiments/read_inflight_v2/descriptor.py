# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""read_inflight_v2 — the isolated pipeline every arm is measured in.

Concept under test: the reader's READ SCHEDULE (how many groups of reads are in
flight over rotating transaction ids, and how deep the CB is behind them) plus
the orthogonal sub-question of the READ TRANSACTION SIZE.

Held constant across arms so the delta is attributable to the reader alone:

  * the compute kernel (one library tilize call),
  * the writer kernel (drain for a resident shard, whole-page writes otherwise),
  * the output CB geometry (``cb_depth_out * WT_CHUNK`` pages — deliberately NOT
    a function of the reader's depth),
  * the blocking (WT_CHUNK, blocks per core, core set) — except in the explicit
    L1-tight cell, where changing the blocking IS the thing being priced,
  * dtype / layout / ComputeConfig (no precision knob is touched anywhere).
"""

import pathlib

import ttnn

KERNEL_DIR = pathlib.Path(__file__).parent / "experiment_kernels"

CB_IN = 0
CB_OUT = 16
TILE_H = 32
TILE_W = 32

# The op's own L1 envelope for the two streaming CBs (tilize_program_descriptor
# CB_L1_BUDGET). Reproduced here so every arm can be priced against it.
CB_L1_BUDGET = 1_048_576
FAST_TILIZE_MAX_W = 255

VARIANT_HELPER = 0
VARIANT_TRID = 2
VARIANT_AHEAD = 3

P_ACCESSOR = 0
P_LOCAL_SHARD = 1

_ELEM = {ttnn.bfloat16: 2, ttnn.float32: 4, ttnn.uint8: 1, ttnn.uint16: 2, ttnn.uint32: 4}


def elem_bytes(dtype):
    return _ELEM[dtype]


def wt_cap(cb_depth, in_tile_bytes, out_tile_bytes):
    """The op's own L1 ceiling on WT_CHUNK (tilize_program_descriptor.wt_cap).

    Reproduced verbatim because raising ``cb_depth`` shrinks it — that coupling
    is the L1 interaction this experiment has to price.
    """
    per_chunk_tile = cb_depth * (in_tile_bytes + out_tile_bytes)
    if per_chunk_tile == 0:
        return FAST_TILIZE_MAX_W
    return max(1, min(FAST_TILIZE_MAX_W, CB_L1_BUDGET // per_chunk_tile))


class Plan:
    """Per-core work assignment + CB geometry, shared by every arm."""

    def __init__(self, *, cores, wt_chunk, wt, blocks_per_core, starts, out_placement, elem, tile_bytes):
        self.cores = cores
        self.wt_chunk = wt_chunk
        self.wt = wt
        self.blocks_per_core = blocks_per_core
        self.starts = starts  # per core: (start_stick, col_off, tile_row0, tile_col0)
        self.out_placement = out_placement
        self.elem = elem
        self.tile_bytes = tile_bytes

    @property
    def row_bytes(self):
        return self.wt_chunk * TILE_W * self.elem

    def in_cb_bytes(self, nt_blk, cb_depth):
        return cb_depth * nt_blk * self.wt_chunk * self.tile_bytes

    def out_cb_bytes(self, cb_depth_out):
        if self.out_placement == P_LOCAL_SHARD:
            return 0  # aliased on the resident shard — costs no extra L1
        return cb_depth_out * self.wt_chunk * self.tile_bytes


def plan_height_sharded(shape, dtype, num_cores, *, src_row_stride=None):
    """ROW_MAJOR source -> HEIGHT-sharded L1 TILE destination (the crossover).

    Each core owns its own shard's tile region and reads the contiguous run of
    source sticks that backs it — the op's R_ALIGNED / W_REGION / n_chunks == 1
    path, i.e. ONE helper call per core.
    """
    h, w = shape[-2], shape[-1]
    assert h % (num_cores * TILE_H) == 0 and w % TILE_W == 0
    shard_h = h // num_cores
    shard_ht, shard_wt = shard_h // TILE_H, w // TILE_W
    elem = _ELEM[dtype]
    cores = [ttnn.CoreCoord(i % 8, i // 8) for i in range(num_cores)]
    starts = [(i * shard_ht * TILE_H, 0, i * shard_ht, 0) for i in range(num_cores)]
    return Plan(
        cores=cores,
        wt_chunk=shard_wt,
        wt=shard_wt,
        blocks_per_core=shard_ht,
        starts=starts,
        out_placement=P_LOCAL_SHARD,
        elem=elem,
        tile_bytes=TILE_H * TILE_W * elem,
    )


def plan_interleaved(shape, dtype, grid, n_chunks):
    """Interleaved source -> interleaved TILE destination (the op's W_BLOCKS split).

    Blocks are W-chunk-major (``wc = b // NT_H``, ``row = b % NT_H``) and each
    core takes a contiguous range, exactly like the op. The assert pins the
    property the reader relies on — a core's whole range sits inside ONE W chunk,
    so its sticks are one contiguous run.
    """
    h, w = shape[-2], shape[-1]
    nt_h, wt = h // TILE_H, w // TILE_W
    assert wt % n_chunks == 0
    wt_chunk = wt // n_chunks
    total_blocks = nt_h * n_chunks
    num_cores = grid[0] * grid[1]
    assert total_blocks % num_cores == 0, f"{total_blocks} blocks over {num_cores} cores"
    per_core = total_blocks // num_cores
    assert nt_h % per_core == 0, "a core's block range must stay inside one W chunk"
    elem = _ELEM[dtype]
    cores, starts = [], []
    for c in range(num_cores):
        cores.append(ttnn.CoreCoord(c % grid[0], c // grid[0]))
        b0 = c * per_core
        wc, row = b0 // nt_h, b0 % nt_h
        starts.append((row * TILE_H, wc * wt_chunk * TILE_W * elem, row, wc * wt_chunk))
    return Plan(
        cores=cores,
        wt_chunk=wt_chunk,
        wt=wt,
        blocks_per_core=per_core,
        starts=starts,
        out_placement=P_ACCESSOR,
        elem=elem,
        tile_bytes=TILE_H * TILE_W * elem,
    )


def _compute_config(in_dtype, out_dtype):
    """The op's OWN precision contract for this dtype pair — never a perf knob.

    fp32 -> fp32 must be bit-exact, which the op buys with fp32 DEST plus
    UnpackToDestFp32 on the input CB (tilize_program_descriptor `lossless_fp32`).
    Every arm of this bake-off gets the identical config.
    """
    cfg = ttnn.ComputeConfigDescriptor()
    lossless_fp32 = in_dtype == ttnn.float32 and out_dtype == ttnn.float32
    # 8-bit datums need fp32 DEST too (the op's own reason: the tilize LLK's
    # 8-bit path is only validated with DEST accumulation on; with a 16-bit DEST
    # the tile packs as zeros). Copied from tilize_program_descriptor so the
    # bench runs the op's contract for uint8, not a weaker one.
    cfg.fp32_dest_acc_en = lossless_fp32 or in_dtype == ttnn.uint8 or out_dtype == ttnn.uint8
    if lossless_fp32:
        modes = [ttnn.UnpackToDestMode.Default] * 32
        modes[CB_IN] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = modes
    return cfg


def build(input_tensor, output_tensor, plan, *, variant, nt_blk, cb_depth, ahead=1, coalesce=1, cb_depth_out=2):
    """One arm's ProgramDescriptor. variant/nt_blk/cb_depth/ahead/coalesce are the knobs."""
    assert plan.blocks_per_core % nt_blk == 0, "nt_blk must divide blocks/core (every group full size)"
    if variant == VARIANT_HELPER:
        assert nt_blk == 1 and coalesce == 1, "the helper's cadence and transfer size ARE the baseline"
    if variant == VARIANT_TRID:
        assert cb_depth == 2, "B8 double-issue uses exactly two fixed CB slots"
    if variant == VARIANT_AHEAD:
        assert cb_depth >= ahead + 1, "the CB must hold every outstanding group plus the one being issued"

    core_set = ttnn.CoreRangeSet({ttnn.CoreRange(c, c) for c in plan.cores})
    tile_desc = ttnn.TileDescriptor(TILE_H, TILE_W)

    cb_in = ttnn.CBDescriptor(
        total_size=plan.in_cb_bytes(nt_blk, cb_depth),
        core_ranges=core_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_IN, data_format=input_tensor.dtype, page_size=plan.tile_bytes, tile=tile_desc
            )
        ],
    )
    if plan.out_placement == P_LOCAL_SHARD:
        cb_out = ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, output_tensor, core_ranges=core_set)
        cb_out.total_size = plan.blocks_per_core * plan.wt_chunk * plan.tile_bytes
        cb_out.format_descriptors = [
            ttnn.CBFormatDescriptor(
                buffer_index=CB_OUT, data_format=output_tensor.dtype, page_size=plan.tile_bytes, tile=tile_desc
            )
        ]
    else:
        cb_out = ttnn.CBDescriptor(
            total_size=plan.out_cb_bytes(cb_depth_out),
            core_ranges=core_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_OUT, data_format=output_tensor.dtype, page_size=plan.tile_bytes, tile=tile_desc
                )
            ],
        )

    reader_ct = [variant, TILE_H, plan.wt_chunk, nt_blk, plan.elem, ahead, cb_depth, coalesce]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    writer_ct = [plan.out_placement, plan.wt_chunk, plan.wt, plan.tile_bytes]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    src_addr, dst_addr = input_tensor.buffer_address(), output_tensor.buffer_address()
    for core, (start_stick, col_off, tile_row0, tile_col0) in zip(plan.cores, plan.starts):
        reader_rt[core.x][core.y] = [src_addr, start_stick, plan.blocks_per_core, col_off]
        writer_rt[core.x][core.y] = [dst_addr, plan.blocks_per_core, tile_row0, tile_col0]
        compute_rt[core.x][core.y] = [plan.blocks_per_core]

    return ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "reader.cpp"),
                core_ranges=core_set,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "writer.cpp"),
                core_ranges=core_set,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "compute.cpp"),
                core_ranges=core_set,
                compile_time_args=[plan.wt_chunk],
                runtime_args=compute_rt,
                config=_compute_config(input_tensor.dtype, output_tensor.dtype),
            ),
        ],
        semaphores=[],
        cbs=[cb_in, cb_out],
    )
