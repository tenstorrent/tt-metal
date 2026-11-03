# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""sfpu_exp_bf16 bench: single-core ProgramDescriptor reconstruction of the ring-joint SDPA softmax exp phases.

MODE 0 = exp_block (in-place Sq x Sk score block, Sub bcast-col -> ScaledExp -> Pack, Strided), MODE 1 = rescale
exp (Sq tiles, Sub -> ScaledExp -> Pack). All inputs are resident L1 shards (zero NoC traffic), so the device
kernel duration is the compute chain.
"""
import struct
from pathlib import Path

import ttnn

KDIR = Path(__file__).parent / "kernels"
VARIANTS = {
    "base": 0,
    "trim": 1,
    "split": 2,
    "split2": 3,
    "fold2": 4,
    "noexp": 5,
    "split2c": 6,
    "fold2c": 7,
    "split2m": 8,
    "fold2m": 9,
    "lm": 10,
    "drop": 11,
    "base_after": 0,
}
CB_SCORES, CB_MAX_BLK, CB_MAX_ACC, CB_OUT = 0, 1, 2, 16


def bf16_bits(x):
    bits = struct.unpack("<I", struct.pack("<f", float(x)))[0]
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def _core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def shard_cfg(h, w):
    return ttnn.create_sharded_memory_config(
        shape=(h, w),
        core_grid=_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def to_dev(t, device):
    h, w = t.shape[-2], t.shape[-1]
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=shard_cfg(h, w)
    )


def _program(variant, mode, Sq, Sk, scale, cb_tensors, fidelity):
    reader_rt = []
    for cb, t, n in cb_tensors["pushed"]:
        reader_rt += [cb, n]
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "exp_bench_reader.cpp"),
        core_ranges=_core(),
        compile_time_args=[len(cb_tensors["pushed"])],
        runtime_args=[(ttnn.CoreCoord(0, 0), reader_rt)],
        config=ttnn.ReaderConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "exp_bench_compute.cpp"),
        core_ranges=_core(),
        compile_time_args=[VARIANTS[variant], mode, Sq, Sk, bf16_bits(scale)],
        runtime_args=[(ttnn.CoreCoord(0, 0), [])],
        config=ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=False, math_approx_mode=False),
    )
    cbs = [ttnn.cb_descriptor_from_sharded_tensor(cb, t) for cb, t in cb_tensors["all"]]
    return ttnn.ProgramDescriptor(kernels=[reader, compute], semaphores=[], cbs=cbs)


def run_exp_block(scores_dev, max_dev, *, variant, Sq, Sk, scale, fidelity=ttnn.MathFidelity.HiFi2):
    """In place: scores_dev <- exp(scale * (scores - max_bcast_col))."""
    cbt = {
        "pushed": [(CB_MAX_BLK, max_dev, Sq)],
        "all": [(CB_SCORES, scores_dev), (CB_MAX_BLK, max_dev)],
    }
    desc = _program(variant, 0, Sq, Sk, scale, cbt, fidelity)
    ttnn.generic_op([scores_dev, max_dev], desc)
    return scores_dev


def run_rescale(acc_dev, blk_dev, out_dev, *, variant, Sq, scale, fidelity=ttnn.MathFidelity.HiFi2):
    """out_dev <- exp(scale * (acc - blk)), Sq tiles."""
    cbt = {
        "pushed": [(CB_MAX_ACC, acc_dev, Sq), (CB_MAX_BLK, blk_dev, Sq)],
        "all": [(CB_MAX_ACC, acc_dev), (CB_MAX_BLK, blk_dev), (CB_OUT, out_dev)],
    }
    desc = _program(variant, 1, Sq, 1, scale, cbt, fidelity)
    ttnn.generic_op([acc_dev, blk_dev, out_dev], desc)
    return out_dev
