# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_pad program descriptors (one per built regime).

extension_band     : one band-writer kernel role (write-only, in place) per DM RISC lane.
new_storage_tiles  : reader (RISCV_1 / NOC0) + writer (RISCV_0 / NOC1); no compute kernel (pure data move).

Every structural value comes from the host plan (g6_pad.py `make_plan`), the single source of truth for the
block knobs. Per-core RT args hold only range start / count / buffer address / fill bits; the descriptor
builders also return the RT-arg slots that change per call so a cached descriptor can be patched.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

CB_FILL_TILE = 0  # one tile of c (constant NoC-write source, scratch of its writer kernel)
CB_FILL_TILE_LANE1 = 2  # second lane's own fill page (band split across both DM RISCs)
CB_INPUT_TILES = 1  # assembly slots of one output block (new_storage_tiles)

# RT-arg indices that vary per call (patched on a cache hit).
RT_ADDR = 0
RT_FILL_BITS = 3


def _cb(index, total_size, page_size, dtype, cores):
    return ttnn.CBDescriptor(
        total_size=total_size,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
    )


def _core_assignment(plan, device):
    """Contiguous unit ranges in row-wise core order (split_work_to_cores, row_wise=True)."""
    grid = device.compute_with_storage_grid_size()
    ncores = plan["ncores"]
    core_set = ttnn.num_cores_to_corerangeset(ncores, grid, True)
    (n, all_cores, g1, g2, units_g1, units_g2) = ttnn.split_work_to_cores(core_set, plan["n_units"], True)
    cores = ttnn.corerange_to_cores(all_cores, None, True)
    g1_cores = set((c.x, c.y) for c in ttnn.corerange_to_cores(g1, None, True))
    ranges = []
    start = 0
    for c in cores:
        cnt = units_g1 if (c.x, c.y) in g1_cores else units_g2
        ranges.append((c, start, cnt))
        start += cnt
    assert start == plan["n_units"], (start, plan["n_units"])
    return all_cores, ranges, units_g1


def build_extension_descriptor(plan, tensor):
    """extension_band: write c into band rows of the last tile row of every slice, in place."""
    device = tensor.device()
    all_cores, ranges, units_g1 = _core_assignment(plan, device)
    tile_bytes = plan["tile_bytes"]
    lanes = plan["band_lanes"]
    # The per-core range is split between the lanes; the block knob is per lane.
    band_block_tiles = plan["band_block_tiles"] or max(1, -(-units_g1 // lanes))

    segs = list(plan["band_segments"]) + [(0, 0)] * (4 - len(plan["band_segments"]))
    accessor_ct = ttnn.TensorAccessorArgs(tensor).get_compile_time_args()

    cbs = []
    kernels = []
    patches = []  # (kernel index, x, y)
    lane_cfgs = [
        (CB_FILL_TILE, ttnn.WriterConfigDescriptor()),  # RISCV_0 / NOC1
        (CB_FILL_TILE_LANE1, ttnn.ReaderConfigDescriptor()),  # RISCV_1 / NOC0
    ][:lanes]
    for lane, (cb_index, config) in enumerate(lane_cfgs):
        cbs.append(_cb(cb_index, tile_bytes, tile_bytes, tensor.dtype, all_cores))
        ct = [cb_index, plan["elem_bytes"], tile_bytes, plan["in_tiles_h"], plan["in_tiles_w"], band_block_tiles]
        ct.append(len(plan["band_segments"]))
        for off, ln in segs:
            ct += [off, ln]
        ct += accessor_ct
        rt = ttnn.RuntimeArgs()
        for c, start, cnt in ranges:
            first = -(-cnt // lanes) if lanes > 1 else cnt
            lane_start, lane_cnt = (start, first) if lane == 0 else (start + first, cnt - first)
            rt[c.x][c.y] = [tensor.buffer_address(), lane_start, lane_cnt, plan["fill_bits"]]
            patches.append((len(kernels), c.x, c.y))
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "g6_pad_band_writer.cpp"),
                core_ranges=all_cores,
                compile_time_args=ct,
                runtime_args=rt,
                config=config,
            )
        )
    desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    plan["band_block_tiles_used"] = band_block_tiles
    return desc, patches


def build_new_storage_descriptor(plan, input_tensor, output_tensor):
    """new_storage_tiles: assemble every output page once in L1 and write it once."""
    device = input_tensor.device()
    all_cores, ranges, units_g1 = _core_assignment(plan, device)
    tile_bytes = plan["tile_bytes"]

    block_tiles = min(plan["block_tiles"] or units_g1, units_g1, plan["block_tiles_cap"])
    num_blocks_max = -(-units_g1 // block_tiles)
    input_depth_blocks = min(plan["input_depth_blocks"], num_blocks_max)
    plan["block_tiles_used"] = block_tiles
    plan["input_depth_blocks_used"] = input_depth_blocks

    cbs = [
        _cb(CB_FILL_TILE, tile_bytes, tile_bytes, input_tensor.dtype, all_cores),
        _cb(CB_INPUT_TILES, input_depth_blocks * block_tiles * tile_bytes, tile_bytes, input_tensor.dtype, all_cores),
    ]

    geom = [
        plan["out_tiles_h"],
        plan["out_tiles_w"],
        plan["out_lead"][1],
        plan["in_tiles_h"],
        plan["in_tiles_w"],
        plan["in_lead"][0],
        plan["in_lead"][1],
    ]
    reader_ct = [CB_INPUT_TILES, tile_bytes, block_tiles] + geom
    reader_ct += ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    writer_ct = [CB_FILL_TILE, CB_INPUT_TILES, plan["elem_bytes"], tile_bytes, block_tiles] + geom
    writer_ct += [plan["edge_row"], plan["edge_col"]]
    writer_ct += ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    patches_in, patches_out = [], []
    for c, start, cnt in ranges:
        reader_rt[c.x][c.y] = [input_tensor.buffer_address(), start, cnt]
        writer_rt[c.x][c.y] = [output_tensor.buffer_address(), start, cnt, plan["fill_bits"]]
        patches_in.append((0, c.x, c.y))
        patches_out.append((1, c.x, c.y))

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_pad_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_pad_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    desc = ttnn.ProgramDescriptor(kernels=[reader, writer], semaphores=[], cbs=cbs)
    return desc, (patches_in, patches_out)
