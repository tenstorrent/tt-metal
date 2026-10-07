# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bench: the matmul_reduce_scatter transport-core fp32-DEST add (relay 2-input / final 3-input).

One Tensix core. Producer (NCRISC) re-exposes resident bf16 input CBs in groups of `group` segments,
the add kernel (TRISC, fp32_dest_acc_en=True, HiFi4 — the op's config) writes cb_sum, the consumer
(BRISC) drains cb_sum in groups of `group` segments. No NoC traffic, so the time is add-bound.
All CBs are backed by single-core L1-sharded tensors of capacity 2*group*seg_tiles pages (the op's
capacity); since the input/output CBs advance in lockstep, output page p == sum of input pages p.
"""

from pathlib import Path

import ttnn

TILE = 32
KDIR = Path(__file__).parent / "kernels"
CB_PARTIAL, CB_A, CB_B, CB_SUM, CB_ZERO = 0, 1, 2, 16, 3

# variant -> (kernel file, extra defines)
VARIANTS = {
    "baseline": ("add_baseline.cpp", {}),
    "baseline_skipcompute": ("add_baseline.cpp", {"CKL_ELTWISE_CHAIN_SKIP_COMPUTE": "1"}),
    "raw0": ("add_raw.cpp", {"MODE": "0"}),
    "raw1": ("add_raw.cpp", {"MODE": "1"}),
    "raw2": ("add_raw.cpp", {"MODE": "2"}),
    "raw3": ("add_raw.cpp", {"MODE": "3"}),
    "raw4": ("add_raw.cpp", {"MODE": "4"}),
    "raw5": ("add_raw.cpp", {"MODE": "5"}),
    "fast": ("add_fast.cpp", {"SEGWAIT": "1"}),
    "fast_packblock": ("add_fast.cpp", {"SEGWAIT": "1", "PACKBLOCK": "1"}),
    "final": ("add_final.cpp", {}),
    "fast_pb_split": ("add_fast.cpp", {"SEGWAIT": "1", "PACKBLOCK": "1", "MATHSPLIT": "1"}),
    "fast_pb_zb": ("add_fast.cpp", {"SEGWAIT": "1", "PACKBLOCK": "1", "ZSIDE": "2"}),
    "fast_pb_zalt": ("add_fast.cpp", {"SEGWAIT": "1", "PACKBLOCK": "1", "ZSIDE": "3"}),
    "fast_blockwait": ("add_fast.cpp", {"SEGWAIT": "0"}),
    "fast_nopack": ("add_fast.cpp", {"SEGWAIT": "1", "DIAG_NOPACK": "1"}),
    "raw0_nopack": ("add_raw.cpp", {"MODE": "0", "DIAG_NOPACK": "1"}),
    "raw4_nopack": ("add_raw.cpp", {"MODE": "4", "DIAG_NOPACK": "1"}),
}


def register(name, kfile, defines=None):
    VARIANTS[name] = (kfile, dict(defines or {}))


def core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def mem_config(cap):
    return ttnn.create_sharded_memory_config(
        shape=(cap * TILE, TILE),
        core_grid=core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _rt(args):
    rt = ttnn.RuntimeArgs()
    rt[0][0] = list(args)
    return rt


def capacity(seg_tiles, group):
    return 2 * group * seg_tiles


def run(inputs, out, *, variant, n_in, seg_tiles, num_segs, group=4, add_block=None, num_copy_segs=0):
    """inputs: list of 2 or 3 sharded bf16 tensors (partial, a[, b]); out: sharded bf16 tensor."""
    kfile, defines = VARIANTS[variant]
    add_block = min(4, seg_tiles) if add_block is None else add_block
    if n_in == 22:  # line-end final: partial + arrival B only (has_a=0, has_b=1)
        cbs_in, has_a, has_b, n_dm = [CB_PARTIAL, CB_B], 0, 1, 2
    else:
        cbs_in = [CB_PARTIAL, CB_A, CB_B][:n_in]
        has_a, has_b, n_dm = 1, (1 if n_in == 3 else 0), n_in
    cr = core()
    producer = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "producer.cpp"),
        core_ranges=cr,
        compile_time_args=[n_dm] + cbs_in + [CB_B] * (3 - len(cbs_in)) + [seg_tiles, group],
        runtime_args=_rt([num_segs, num_copy_segs]),
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
    )
    consumer = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "consumer.cpp"),
        core_ranges=cr,
        compile_time_args=[CB_SUM, seg_tiles, group],
        runtime_args=_rt([num_segs, num_copy_segs]),
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / kfile),
        core_ranges=cr,
        compile_time_args=[CB_PARTIAL, CB_A, CB_B, CB_SUM, has_a, has_b, add_block, seg_tiles, CB_ZERO],
        runtime_args=_rt([num_copy_segs, num_segs]),
        defines=list(defines.items()),
        config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
    )
    cbs = [ttnn.cb_descriptor_from_sharded_tensor(cb, t) for cb, t in zip(cbs_in, inputs)]
    cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_SUM, out))
    # one-page scratch CB (the zero tile some variants self-produce); unused by the baseline
    cbs.append(
        ttnn.CBDescriptor(
            total_size=2048,
            core_ranges=cr,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_ZERO, data_format=ttnn.bfloat16, page_size=2048)
            ],
        )
    )
    desc = ttnn.ProgramDescriptor(kernels=[producer, consumer, compute], semaphores=[], cbs=cbs)
    return ttnn.generic_op(list(inputs) + [out], desc)
