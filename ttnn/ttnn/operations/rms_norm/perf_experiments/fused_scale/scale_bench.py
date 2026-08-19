# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off for rms_norm's SCALE PASS:  out = (x * 1/rms) * gamma.

CONCEPT ISOLATION (/perf-lab): every operand is resident in L1 on ONE core, the
CBs are backed by the sharded tensors themselves (`cb_descriptor_from_sharded_
tensor`), and there is no reader/writer kernel — so the measured device time is
compute only, with zero NoC traffic.  The scale pass is then repeated
`kernel_iters` times so the per-tile cost of the two multiplies dominates the
launch overhead.  Nothing but the *emission* of the two multiplies changes
between arms.

Geometry mirrors the real op's own plan for the shape under test: `BLOCK_HT`
tile-rows x `wt` tile-columns per call, `DEST_BLOCK` tiles per dst-sync window.
For the focus case (1,1,32,7168) at bf16/HiFi2/fp32_dest_acc_en=False the op's
`blocking_plan` gives Regime B, BLOCK_HT=1, WT_SCALE_BLOCK=112, DEST_BLOCK=8.

PRECISION CONTRACT: the ComputeConfigDescriptor is an input to every arm and is
never touched for speed (math_fidelity / fp32_dest_acc_en / math_approx_mode /
dst_full_sync_en and the dtypes are the caller's).
"""

from __future__ import annotations

from pathlib import Path

import ttnn

TILE = 32
KERNEL_DIR = Path(__file__).parent / "kernels"

CB_X = 0
CB_GAMMA = 1
CB_RMS = 2
CB_NORMED = 3
CB_RMS_FULL = 4
CB_GAMMA_FULL = 5
CB_OUT = 16

# arm name -> (variant id, scratch CBs it needs)
#   normed     : the intermediate the baseline round-trips through (BLOCK_HT*wt pages)
#   rms_full   : materialised col-broadcast 1/rms (BLOCK_HT pages)
#   gamma_full : materialised row-broadcast gamma (wt pages)
VARIANTS = {
    "baseline": (0, ("normed",)),
    "baseline_reversed": (1, ("normed",)),
    "fused_rmsfull": (2, ("rms_full",)),
    "fused_inchain": (3, ("rms_full",)),
    "fused_gammafull": (4, ("gamma_full",)),
    "fused_gammafull_amortized": (5, ("gamma_full",)),
    "raw_llk": (6, ("rms_full",)),
    "fused_sfpu": (7, ()),
    "baseline_subchunk": (8, ("normed_sub",)),
}

# Arms whose chain contains a DestReuseBinary element.
#
# MEASURED HARDWARE/HELPER CONSTRAINT (this bench, Blackhole p150b): at 16-bit
# DEST (fp32_dest_acc_en=False, DEST_AUTO_LIMIT = 8) a DestReuseBinary running
# at block_size == 8 corrupts ROWS 16-31 OF THE LAST TILE of every DEST block
# (pcc 0.9834 vs 0.99998 at block <= 7; the bad tile reads another row's reuse
# operand).  block_size <= 7 is exact.  At fp32 DEST (limit 4) block == limit is
# clean, so the reuse path needs one 16-bit slot of headroom that
# `chain_max_block_v`'s clamp does not reserve.  Capped here, per arm, so the
# bake-off never measures a wrong answer.
REUSE_ARMS = ("fused_rmsfull", "fused_inchain", "fused_gammafull", "fused_gammafull_amortized", "raw_llk")


def dest_limit(compute_kernel_config):
    """Host mirror of ckl::DEST_AUTO_LIMIT (see the op's `_dest_limit`)."""
    base = 16 if bool(getattr(compute_kernel_config, "dst_full_sync_en", False)) else 8
    return base // 2 if bool(getattr(compute_kernel_config, "fp32_dest_acc_en", True)) else base


def dest_block_for(arm, dest_block, compute_kernel_config):
    """The largest block this arm can run CORRECTLY (see REUSE_ARMS)."""
    if arm not in REUSE_ARMS:
        return dest_block
    limit = dest_limit(compute_kernel_config)
    return min(dest_block, limit - 1) if limit > 1 else dest_block


# Intermediate CB format policy, copied from the op (`_interm_dtype`): a
# block-float value parked in L1 gets re-quantised, so a compute-only
# intermediate promotes to bfloat16.
BLOCK_FLOAT = (ttnn.bfloat8_b, ttnn.bfloat4_b)


def interm_dtype(dtype):
    return ttnn.bfloat16 if dtype in BLOCK_FLOAT else dtype


def single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def sharded_config(rows, cols):
    return ttnn.create_sharded_memory_config(
        shape=(rows, cols),
        core_grid=single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _scratch_cb(cb_id, num_pages, dtype):
    page = ttnn.tile_size(dtype)
    return ttnn.CBDescriptor(
        total_size=num_pages * page,
        core_ranges=single_core(),
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=dtype, page_size=page)],
    )


def make_inputs(device, *, wt, block_ht, dtype, gamma_dtype=None, has_gamma=True, seed=0):
    """x / gamma / 1/rms as single-core L1-sharded tensors, plus the torch reference.

    gamma carries data only in tile row 0 and 1/rms only in tile column 0 —
    exactly the tiles the op's two hardware broadcasts read, so a variant that
    materialises a full tile has to do the broadcast itself and cannot cheat off
    a pre-filled tile.
    """
    import torch

    torch.manual_seed(seed)
    gamma_dtype = gamma_dtype or dtype
    h = block_ht * TILE
    w = wt * TILE

    x_t = torch.randn((h, w), dtype=torch.float32)
    rms_col = torch.rand((h, 1), dtype=torch.float32) * 1.5 + 0.25  # 1/rms, strictly positive
    gamma_row = torch.randn((1, w), dtype=torch.float32)

    rms_t = torch.zeros((h, TILE), dtype=torch.float32)
    rms_t[:, 0:1] = rms_col
    gamma_t = torch.zeros((TILE, w), dtype=torch.float32)
    gamma_t[0:1, :] = gamma_row

    x = ttnn.from_torch(
        x_t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=sharded_config(h, w)
    )
    rms = ttnn.from_torch(
        rms_t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=sharded_config(h, TILE)
    )
    gamma = None
    if has_gamma:
        gamma = ttnn.from_torch(
            gamma_t,
            dtype=gamma_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=sharded_config(TILE, w),
        )

    ref = x_t * rms_col
    if has_gamma:
        ref = ref * gamma_row
    return x, gamma, rms, ref


def create_program_descriptor(
    x, gamma, rms, out, *, arm, wt, block_ht, dest_block, kernel_iters, compute_kernel_config, sub_chunk=0
):
    if arm not in VARIANTS:
        raise ValueError(f"arm must be one of {tuple(VARIANTS)}, got {arm!r}")
    variant, scratch = VARIANTS[arm]
    has_gamma = gamma is not None
    t_interm = interm_dtype(x.dtype)

    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_X, x),
        ttnn.cb_descriptor_from_sharded_tensor(CB_RMS, rms),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, out),
    ]
    if has_gamma:
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_GAMMA, gamma))
    if has_gamma and "normed" in scratch:
        cbs.append(_scratch_cb(CB_NORMED, block_ht * wt, t_interm))
    if has_gamma and "normed_sub" in scratch:
        cbs.append(_scratch_cb(CB_NORMED, block_ht * min(sub_chunk or wt, wt), t_interm))
    if has_gamma and "rms_full" in scratch:
        cbs.append(_scratch_cb(CB_RMS_FULL, block_ht, t_interm))
    if has_gamma and "gamma_full" in scratch:
        cbs.append(_scratch_cb(CB_GAMMA_FULL, wt, t_interm))

    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scale_compute.cpp"),
        core_ranges=single_core(),
        compile_time_args=[wt, block_ht, dest_block, kernel_iters, variant, 1 if has_gamma else 0, sub_chunk],
        config=compute_kernel_config,
    )
    return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)


def scratch_bytes(arm, *, wt, block_ht, dtype, has_gamma=True, sub_chunk=0):
    """L1 the arm's INTERMEDIATE CBs cost (the currency the fusion frees)."""
    if not has_gamma:
        return 0
    _, scratch = VARIANTS[arm]
    page = ttnn.tile_size(interm_dtype(dtype))
    pages = 0
    if "normed" in scratch:
        pages += block_ht * wt
    if "normed_sub" in scratch:
        pages += block_ht * min(sub_chunk or wt, wt)
    if "rms_full" in scratch:
        pages += block_ht
    if "gamma_full" in scratch:
        pages += wt
    return pages * page


def run_arm(x, gamma, rms, *, arm, wt, block_ht, dest_block, kernel_iters, compute_kernel_config, sub_chunk=0):
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([block_ht * TILE, wt * TILE]),
        x.dtype,
        ttnn.TILE_LAYOUT,
        x.device(),
        sharded_config(block_ht * TILE, wt * TILE),
    )
    desc = create_program_descriptor(
        x,
        gamma,
        rms,
        out,
        arm=arm,
        wt=wt,
        block_ht=block_ht,
        dest_block=dest_block,
        kernel_iters=kernel_iters,
        compute_kernel_config=compute_kernel_config,
        sub_chunk=sub_chunk,
    )
    tensors = [x, rms, out] if gamma is None else [x, gamma, rms, out]
    return ttnn.generic_op(tensors, desc)
