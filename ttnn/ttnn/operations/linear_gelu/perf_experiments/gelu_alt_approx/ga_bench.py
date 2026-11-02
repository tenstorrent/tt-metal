# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated pack-thread GELU bench (gelu_alt_approx). One generic_op per call: fp32 input height-sharded
over a core grid (UnpackToDestFp32), math copies blocks to DEST, PACK thread runs the SFPU variant and packs
to a sharded output (bf16 = the op's output, or fp32 for internal-error probes)."""

from pathlib import Path

import ttnn

HERE = Path(__file__).parent
TILE = 32
CB_IN = 0
CB_OUT = 16

VARIANTS = {
    "none": 0,
    "base": 1,
    "t21": 2,
    "mag": 3,
    "hwx": 4,
    "t21n2": 5,
    "magn2": 6,
    "magb": 7,
    "t21s": 8,
    "t21sh": 9,
    "probe_exp": 10,
    "probe_recip": 11,
    "probe_round": 12,
}


def grid(nx, ny):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nx - 1, ny - 1))])


def shard_cfg(cores, tiles_per_core):
    return ttnn.create_sharded_memory_config(
        shape=(tiles_per_core * TILE, TILE),
        core_grid=cores,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def run(x_dev, *, variant, cores, tiles_per_core, block=3, reps=1, out_dtype=ttnn.bfloat16):
    ncores = cores.num_cores()
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([ncores * tiles_per_core * TILE, TILE]),
        out_dtype,
        ttnn.TILE_LAYOUT,
        x_dev.device(),
        shard_cfg(cores, tiles_per_core),
    )
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, math_approx_mode=False
    )
    modes = [ttnn.UnpackToDestMode.Default] * 64
    modes[CB_IN] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    compute = ttnn.KernelDescriptor(
        kernel_source=str(HERE / "bench_compute.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[VARIANTS[variant], tiles_per_core, block, reps],
        runtime_args=[],
        config=cfg,
    )
    sink = ttnn.KernelDescriptor(
        kernel_source=str(HERE / "bench_sink.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[tiles_per_core, reps],
        runtime_args=[],
        config=ttnn.WriterConfigDescriptor(),
    )
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_IN, x_dev),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, out),
    ]
    desc = ttnn.ProgramDescriptor(kernels=[compute, sink], semaphores=[], cbs=cbs)
    return ttnn.generic_op([x_dev, out], desc)
