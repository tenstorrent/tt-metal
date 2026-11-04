# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""pv_rescale_fusion bench: single-core ProgramDescriptor running the op's per-unit compute loop (copy of the op's
compute kernel, kernels/compute.cpp, VARIANT selects the P.V + online-softmax-correction implementation).

Q / K / V / out CBs are backed by resident L1 shards (K / V: two key-block slots, block kb uses slot kb % 2), the
reader only publishes credit, so DEVICE KERNEL DURATION is the compute kernel's key walk.
"""
import struct
from pathlib import Path

import ttnn
from ttnn.operations.ring_joint_scaled_dot_product_attention import (
    ring_joint_scaled_dot_product_attention_program_descriptor as pd,
)

KDIR = Path(__file__).parent / "kernels"
VARIANTS = {"base": 0, "fused": 1, "l1acc": 2, "fused_mmfirst": 3, "fused_l": 4}


def bf16_bits(x):
    bits = struct.unpack("<I", struct.pack("<f", float(x)))[0]
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def _core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def to_dev(t, device, dtype):
    h, w = t.shape[-2], t.shape[-1]
    mc = ttnn.create_sharded_memory_config(
        shape=(h, w),
        core_grid=_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)


def k_slot_layout(torch, K, Sk, DHt):
    """K slot (Sk*32 keys, DH) -> (DHt*32, Sk*32) whose row-major tile (d, t) holds K tile (t, d) untransposed —
    the op's transposed-block CB order (tile (t, d) at CB index d*Sk + t)."""
    return K.reshape(Sk, 32, DHt, 32).permute(2, 1, 0, 3).reshape(DHt * 32, Sk * 32).contiguous()


def run(
    device,
    *,
    variant,
    q_dev,
    k_dev,
    v_dev,
    out_dev,
    Sq,
    Sk,
    DHt,
    DVt,
    segments,
    R,
    mask_cols,
    scale,
    fp32,
    fidelity,
    q_dtype,
    kv_dtype,
):
    acc_dtype = ttnn.float32 if fp32 else ttnn.bfloat16
    plan = pd.compute_cb_plan(Sq, Sk, DHt=DHt, DVt=DVt, q_dtype=q_dtype, kv_dtype=kv_dtype, acc_dtype=acc_dtype)
    backed = {pd.CB_Q_BLOCK: q_dev, pd.CB_K_BLOCK: k_dev, pd.CB_V_BLOCK: v_dev, pd.CB_OUT_TILES: out_dev}
    cbs = []
    for idx, pages, dt in plan:
        if idx in backed:
            cbs.append(ttnn.cb_descriptor_from_sharded_tensor(idx, backed[idx]))
        else:
            page = ttnn.tile_size(dt)
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=pages * page,
                    core_ranges=_core(),
                    format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)],
                )
            )
    n_blocks = sum((v + Sk - 1) // Sk for v, _ in segments)
    c0 = ttnn.CoreCoord(0, 0)
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "reader.cpp"),
        core_ranges=_core(),
        compile_time_args=[pd.CB_Q_BLOCK, pd.CB_K_BLOCK, pd.CB_V_BLOCK, Sq, Sk, DHt, DVt],
        runtime_args=[(c0, [1, n_blocks])],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "writer.cpp"),
        core_ranges=_core(),
        compile_time_args=[pd.CB_OUT_TILES, pd.CB_KEY_MASK, pd.CB_MAX_SCALER, pd.CB_SUM_SCALER, mask_cols],
        runtime_args=[(c0, [1, R * DVt])],
        config=ttnn.WriterConfigDescriptor(),
    )
    ct = [
        pd.CB_Q_BLOCK,
        pd.CB_K_BLOCK,
        pd.CB_V_BLOCK,
        pd.CB_KEY_MASK,
        pd.CB_MAX_SCALER,
        pd.CB_SUM_SCALER,
        pd.CB_SCORES,
        pd.CB_MAX_BLK,
        pd.CB_MAX_ACC,
        pd.CB_SUM_BLK,
        pd.CB_SUM_ACC,
        pd.CB_ROW_FACTOR,
        pd.CB_OUT_BLK,
        pd.CB_OUT_ACC,
        pd.CB_OUT_TILES,
        Sq,
        Sk,
        DHt,
        DVt,
        Sq,  # n_q_t (unused by the walk)
        0,  # Lq_t
        0,  # n_kv_t
        1,  # BH
        1,  # C
        1,  # hq
        1,  # hk
        bf16_bits(scale),
        VARIANTS[variant],
    ]
    whole = R == Sq
    rt = [0, len(segments)]
    for valid, masked in segments:
        rt += [valid, int(masked)]
    rt += [1 if whole else 0, 0 if whole else 1, 0 if whole else R]
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "compute.cpp"),
        core_ranges=_core(),
        compile_time_args=ct,
        runtime_args=[(c0, rt)],
        config=ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=fp32, math_approx_mode=False),
    )
    desc = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    ttnn.generic_op([q_dev, k_dev, v_dev, out_dev], desc)
    return n_blocks
