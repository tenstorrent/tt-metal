# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""read_inflight bake-off — the isolated pipeline the arms are measured in.

Concept under test: BYTES IN FLIGHT PER READ BARRIER on the reader's
DRAM -> L1 crossover. Everything else is held constant across arms:

  * the compute kernel (one library tilize call),
  * the writer kernel (drain for a resident shard, whole-page writes otherwise),
  * the output CB geometry (``cb_depth_out * WT_CHUNK`` pages — deliberately NOT
    a function of NT_BLK, so the only L1 that moves between arms is the reader's
    input CB),
  * the blocking (WT_CHUNK, blocks per core, core set),
  * dtype / layout / ComputeConfig (no precision knob is touched anywhere).

The three arms are described in ``experiment_kernels/reader.cpp``.
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

VARIANT_HELPER = 0
VARIANT_RAW = 1
VARIANT_TRID = 2
VARIANT_AHEAD = 3

P_ACCESSOR = 0
P_LOCAL_SHARD = 1


def _elem_bytes(dtype):
    return {ttnn.bfloat16: 2, ttnn.float32: 4, ttnn.uint8: 1, ttnn.uint16: 2, ttnn.uint32: 4}[dtype]


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


def plan_height_sharded(shape, dtype, num_cores):
    """DRAM interleaved ROW_MAJOR source -> HEIGHT-sharded L1 TILE destination.

    Each core owns its own shard's tile region and reads the contiguous run of
    source sticks that backs it (the op's R_ALIGNED / W_REGION / n_chunks == 1
    path — one helper call per core).
    """
    h, w = shape[-2], shape[-1]
    assert h % (num_cores * TILE_H) == 0 and w % TILE_W == 0
    shard_h = h // num_cores
    shard_ht, shard_wt = shard_h // TILE_H, w // TILE_W
    elem = _elem_bytes(dtype)
    cores = [ttnn.CoreCoord(i, 0) for i in range(num_cores)]
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
    """DRAM interleaved -> DRAM interleaved, the op's W_BLOCKS split.

    Blocks are W-chunk-major (``wc = b // NT_H``, ``row = b % NT_H``) and each
    core takes a contiguous range, exactly like the op. ``n_chunks`` reproduces
    the op's ``derive_blocking`` result for the shape under test; the assert
    below pins the property the run relies on — a core's whole range sits inside
    ONE W chunk, so its sticks are contiguous.
    """
    h, w = shape[-2], shape[-1]
    nt_h, wt = h // TILE_H, w // TILE_W
    assert wt % n_chunks == 0
    wt_chunk = wt // n_chunks
    total_blocks = nt_h * n_chunks
    num_cores = grid[0] * grid[1]
    assert total_blocks % num_cores == 0
    per_core = total_blocks // num_cores
    assert nt_h % per_core == 0, "a core's block range must stay inside one W chunk"
    elem = _elem_bytes(dtype)
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
    if in_dtype == ttnn.float32 and out_dtype == ttnn.float32:
        cfg.fp32_dest_acc_en = True
        modes = [ttnn.UnpackToDestMode.Default] * 32
        modes[CB_IN] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = modes
    return cfg


def build(input_tensor, output_tensor, plan, *, variant, nt_blk, cb_depth, ahead=1, cb_depth_out=2):
    """One arm's ProgramDescriptor. `variant`/`nt_blk`/`cb_depth`/`ahead` are the knobs."""
    assert plan.blocks_per_core % nt_blk == 0, "nt_blk must divide blocks/core (every group full size)"
    if variant == VARIANT_HELPER:
        assert nt_blk == 1, "the helper's barrier cadence IS NT_BLK=1 — that is the point"
    if variant == VARIANT_TRID:
        assert cb_depth == 2, "B8 double-issue uses exactly two fixed CB slots"
    if variant == VARIANT_AHEAD:
        assert cb_depth >= ahead + 1, "the CB must hold every outstanding group plus the one being issued"
        # A core with fewer than `ahead + 1` groups is NOT a correctness problem:
        # the kernel's drain loop pushes whatever is outstanding, so the window
        # simply never fills and the schedule degenerates to the baseline's.

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

    reader_ct = [variant, TILE_H, plan.wt_chunk, nt_blk, plan.elem, ahead, cb_depth]
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
                # Precision contract FIXED and identical on every arm. fp32 in /
                # fp32 out takes the op's own exactness settings (fp32 DEST +
                # lossless unpack) because that is the contract the op ships for
                # that dtype pair — it is NOT a knob this experiment turns.
                config=_compute_config(input_tensor.dtype, output_tensor.dtype),
            ),
        ],
        semaphores=[],
        cbs=[cb_in, cb_out],
    )
