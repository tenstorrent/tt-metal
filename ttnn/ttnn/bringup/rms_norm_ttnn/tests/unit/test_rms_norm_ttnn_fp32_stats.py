# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""CHANGELOG 3 / 4 -- the sum of squares stays fp32 at fp32_dest_acc_en.

DO NOT DELETE.  What this pins:

  1. NO SCALE BIAS ACROSS WIDTH CHUNKS.  A row wider than one chunk is reduced chunk by chunk and the running
     sum is carried in the fp32 accumulator CB.  reduce()'s default reload (CopySeedPairs) added an ODD chunk's
     leftover tile with a DEST_TO_SRCB reuse add, which moved the fp32 carry into a bf16 srcB: every carry was
     truncated to bf16, the mean square came out low and every row's scale high.  Measured at MiMo's
     (2048, 4096) weight shape (then 3 chunks of 43): row-norm ratio vs float64 1.00090, rel L2 0.00248; after
     the fix 0.99968 / 0.00197 (native ttnn.rms_norm: 0.99945 / 0.00205).  The kernel now uses CopySeedSfpuAdd
     at fp32_dest_acc_en.  The device cases below each run several chunks, most of them ODD, at fp32 DEST.
     What is left is the carry's reload: copy_tile unpacks the fp32 accumulator through srcA, i.e. at tf32
     (10 mantissa bits, truncated), in every reload mode the helper has.  Lossless would need the accumulator
     CB tagged UnpackToDestFp32, which cb_row_stat cannot be (pass B also reads it as an FPU operand).  It is 8x
     smaller than the bf16 truncation and grows with the chunk count: measured +0.4e-4 at 2 chunks, +1.7e-4 at 4,
     +4.5e-4 at 7 (unit weight).  The limit is therefore per chunk, 1e-4 x chunks; before the fix a 3-chunk row
     was at +9.3e-4.
  2. cb_x_squared IS fp32 EXACTLY WHEN IT IS AN ACCUMULATOR (CHANGELOG 4, which narrowed CHANGELOG 3's "both
     statistics CBs fp32 at fp32 DEST"), on both builders.  With the DEST fold on (x_squared_wt < WT_CHUNK) each
     of its tiles is a sum of several squares and it is fp32; without the fold (a partial last tile, a chunk with
     no divisor 2..SQ_FOLD_GROUP) it holds single x^2 tiles and keeps the intermediate format.  cb_normalized is
     read once and is always the intermediate format.  The partial-width 0/1 mask in cb_scaler follows
     cb_x_squared (it is unpacked in that format; a bf16 mask read as fp32 scales the output, which is what the
     source tests' w=4022 / w=200 cases caught).  At fp32_dest_acc_en=False nothing is fp32, so that program is
     the one it was.
"""

from __future__ import annotations

import pytest
import torch

import ttnn

import ttnn.bringup.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as PD
from ttnn.bringup.rms_norm_ttnn.tests.unit.builders import BUILDER_IDS, BUILDERS

_READER_CT_WT_CHUNK = 2
_READER_CT_NUM_W_CHUNKS = 3


def _config(fp32):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi4
    c.fp32_dest_acc_en = fp32
    c.math_approx_mode = False
    return c


def _tensors(device, H, W, mode, x=None, r=None, w=None):
    kw = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = torch.zeros(1, 1, H, W) if x is None else x
    tx = ttnn.from_torch(x.bfloat16(), **kw)
    tw = None
    if "gamma" in mode:
        tw = ttnn.from_torch((torch.zeros(1, 1, 1, W) if w is None else w).bfloat16().reshape(1, 1, 1, W), **kw)
    tr = None
    if "residual" in mode:
        tr = ttnn.from_torch((torch.zeros(1, 1, H, W) if r is None else r).bfloat16(), **kw)
    return tx, tw, tr


def _descriptor(device, H, W, mode, fp32, builder):
    tx, tw, tr = _tensors(device, H, W, mode)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, H, W]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    return BUILDERS[builder](tx, out, weight=tw, residual=tr, epsilon=1e-6, compute_kernel_config=_config(fp32))


def _cb_page_sizes(desc):
    # CBFormatDescriptor.data_format does not convert to Python; the tile page size tells fp32 from bf16
    return {fd.buffer_index: fd.page_size for cb in desc.cbs for fd in cb.format_descriptors}


# (W, mode, folded at fp32 DEST): the chunk the fp32-DEST solve picks (at H=2048) decides the fold.
FORMAT_CASES = [
    (4096, "gamma", True),  # 64 x 2, folds in groups of 16
    (4096, "residual_gamma", False),  # 43 x 3, prime chunk: no fold
    (4022, "gamma", False),  # partial last tile: never folds
    (256, "gamma", True),  # chunk 8 <= DEST_ACC_SQUARE_MAX_WT: the flat fold
]


@pytest.mark.parametrize("builder", BUILDER_IDS)
@pytest.mark.parametrize("fp32", [True, False], ids=["fp32dest", "bf16dest"])
@pytest.mark.parametrize("W, mode, folded", FORMAT_CASES, ids=[f"{w}-{m}" for w, m, _ in FORMAT_CASES])
def test_cb_x_squared_is_fp32_only_as_an_accumulator(device, builder, fp32, W, mode, folded):
    desc = _descriptor(device, 2048, W, mode, fp32, builder)
    page = _cb_page_sizes(desc)
    wt_chunk = list(desc.kernels[0].compile_time_args)[_READER_CT_WT_CHUNK]
    if fp32:
        assert (PD._x_squared_wt(wt_chunk, W % 32) < wt_chunk) == folded, f"the case no longer hits fold={folded}"
    bf16, f32 = ttnn.tile_size(ttnn.bfloat16), ttnn.tile_size(ttnn.float32)
    want_sq = f32 if (fp32 and folded) else bf16
    assert page[PD.CB_X_SQUARED] == want_sq
    assert page[PD.CB_NORMALIZED] == bf16
    # the partial-width mask is read in cb_x_squared's format; a full-width build keeps the bf16 1.0 scaler
    assert page[PD.CB_SCALER] == (want_sq if W % 32 else bf16)


# (H, W, mode): each is several width chunks at fp32 DEST (blocking at CHANGELOG 4 in the comment; at CHANGELOG 3
# they were 32 x 4, 45 x 2 for 2848 gamma, 23 x 7, 45 x 5 and 21 x 6).
DEVICE_CASES = [
    (2048, 4096, "gamma"),  # MiMo's shape, 64 x 2, folded
    (2048, 4096, "residual_gamma"),  # 43 x 3, odd, prime chunk: no fold (bf16 cb_x_squared)
    (2048, 2848, "residual_gamma"),  # 45 x 2, odd, folded
    (2048, 5120, "gamma"),  # 40 x 4, folded
    (1024, 7168, "none"),  # 75 x 3, odd, folded
    (2048, 4022, "residual_gamma"),  # 42 x 3, partial last tile: no fold (the source test's shape)
]


@pytest.mark.parametrize("H, W, mode", DEVICE_CASES, ids=[f"{h}x{w}-{m}" for h, w, m in DEVICE_CASES])
def test_no_scale_bias_across_width_chunks(device, H, W, mode):
    desc = _descriptor(device, H, W, mode, True, "cpp")
    reader = list(desc.kernels[0].compile_time_args)
    assert reader[_READER_CT_NUM_W_CHUNKS] > 1, "the case must reduce its rows in more than one chunk"

    torch.manual_seed(0)
    x = torch.randn(1, 1, H, W) * (0.5 + torch.rand(1, 1, H, 1) * 4)
    r = torch.randn(1, 1, H, W) if "residual" in mode else None
    ones = torch.ones(W)  # a unit weight isolates the statistics from the weight multiply
    tx, tw, tr = _tensors(device, H, W, mode, x=x, r=r, w=ones)
    cfg = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    y = ttnn.to_torch(
        ttnn.bringup.rms_norm(
            tx,
            weight=tw,
            residual_input_tensor=tr,
            epsilon=1e-6,
            compute_kernel_config=cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    ).double()
    t = x.bfloat16().double() + (r.bfloat16().double() if r is not None else 0)
    ref = t * torch.rsqrt((t * t).mean(-1, keepdim=True) + 1e-6)
    ratio = (y.norm(dim=-1) / ref.norm(dim=-1)).flatten()
    bias = ratio.mean().item() - 1
    rel = ((y - ref).norm() / ref.norm()).item()
    chunks = f"{reader[_READER_CT_WT_CHUNK]} x {reader[_READER_CT_NUM_W_CHUNKS]}"
    print(f"{H}x{W} {mode} chunks {chunks}: row-norm ratio bias {bias:+.2e}, rel L2 {rel:.5f}")
    # Measured after the fix (see the module docstring): bias +0.4e-4 (2 chunks) .. +4.5e-4 (7 chunks), rel L2
    # 0.0017-0.0024 (bf16 output floor ~0.0017; the residual case adds t's rounding).
    lim = 1e-4 * reader[_READER_CT_NUM_W_CHUNKS]
    assert abs(bias) < lim, f"per-row scale bias {bias:+.2e} > {lim:.1e} (chunks {chunks})"
    assert rel < 0.003, f"rel L2 {rel:.5f} (chunks {chunks})"


# (H, W, mode): fold on / off, residual on / off, partial last tile, at both DEST widths (blocking at CHANGELOG 4).
PRECISION_CASES = [
    (2048, 256, "gamma"),  # chunk 8: the flat fold
    (2048, 256, "residual_gamma"),
    (2048, 4096, "gamma"),  # fp32 DEST 64 x 2: grouped fold (16-bit DEST 43 x 3)
    (2048, 4096, "residual_gamma"),  # 43 x 3: prime chunk, no fold
    (2048, 4022, "gamma"),  # 42 x 3: partial last tile, no fold
    (2048, 4022, "residual_gamma"),
    (2048, 5120, "gamma"),  # fp32 DEST 40 x 4: grouped fold
    (1024, 7168, "none"),  # fp32 DEST 75 x 3: grouped fold, no weight
]


@pytest.mark.parametrize("fp32", [True, False], ids=["fp32dest", "bf16dest"])
@pytest.mark.parametrize("H, W, mode", PRECISION_CASES, ids=[f"{h}x{w}-{m}" for h, w, m in PRECISION_CASES])
def test_precision_against_float64(device, H, W, mode, fp32):
    """rel L2 vs a float64 reference with a random weight.  Measured at CHANGELOG 4 (HiFi4): fp32 DEST 0.00168
    (no weight) / 0.00237 (weight: y is rounded to bf16 in cb_normalized and again on the output) / 0.0029-0.0030
    (residual: t is rounded too); 16-bit DEST 0.0036-0.0056.  Native ttnn.rms_norm on the same inputs: fp32 DEST
    0.0017-0.0020, 16-bit DEST 0.016-0.044."""
    torch.manual_seed(0)
    x = torch.randn(1, 1, H, W) * (0.5 + torch.rand(1, 1, H, 1) * 4)
    r = torch.randn(1, 1, H, W) if "residual" in mode else None
    w = (1 + 0.5 * torch.randn(W)) if "gamma" in mode else None
    tx, tw, tr = _tensors(device, H, W, mode, x=x, r=r, w=w)
    cfg = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32,
        packer_l1_acc=False,
    )
    y = ttnn.to_torch(
        ttnn.bringup.rms_norm(
            tx,
            weight=tw,
            residual_input_tensor=tr,
            epsilon=1e-6,
            compute_kernel_config=cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    ).double()
    t = x.bfloat16().double() + (r.bfloat16().double() if r is not None else 0)
    ref = t * torch.rsqrt((t * t).mean(-1, keepdim=True) + 1e-6)
    if w is not None:
        ref = ref * w.bfloat16().double()
    rel = ((y - ref).norm() / ref.norm()).item()
    print(f"{H}x{W} {mode} fp32_dest={fp32}: rel L2 {rel:.5f}")
    lim = (0.0035 if "residual" in mode else 0.0028) if fp32 else 0.0065
    assert rel < lim, f"rel L2 {rel:.5f} >= {lim}"
