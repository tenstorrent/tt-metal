# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""CHANGELOG 3 -- the sum of squares and the normalized block stay fp32 at fp32_dest_acc_en.

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
  2. THE STATISTICS CBs ARE fp32 AT fp32 DEST, on both builders: cb_x_squared, cb_normalized, and the
     partial-width 0/1 mask in cb_scaler (it is unpacked in cb_x_squared's format; a bf16 mask read as fp32
     scales the output, which is what the source tests' w=4022 / w=200 cases caught).  At fp32_dest_acc_en=False
     they keep the input's intermediate format, so that program is the one it was.
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


@pytest.mark.parametrize("builder", BUILDER_IDS)
@pytest.mark.parametrize("fp32", [True, False], ids=["fp32dest", "bf16dest"])
@pytest.mark.parametrize("W", [4096, 4022], ids=["aligned", "partial"])
def test_statistics_cbs_are_fp32_at_fp32_dest(device, builder, fp32, W):
    page = _cb_page_sizes(_descriptor(device, 2048, W, "gamma", fp32, builder))
    want = ttnn.tile_size(ttnn.float32 if fp32 else ttnn.bfloat16)
    assert page[PD.CB_X_SQUARED] == want
    assert page[PD.CB_NORMALIZED] == want
    # the partial-width mask is read in cb_x_squared's format; a full-width build keeps the bf16 1.0 scaler
    assert page[PD.CB_SCALER] == (want if W % 32 else ttnn.tile_size(ttnn.bfloat16))


# (H, W, mode): each is several width chunks at fp32 DEST (blocking at the time of the fix in the comment).
DEVICE_CASES = [
    (2048, 4096, "gamma"),  # MiMo's shape, 32 x 4
    (2048, 2848, "gamma"),  # 45 x 2, odd
    (2048, 5120, "gamma"),  # 23 x 7, odd
    (1024, 7168, "none"),  # 45 x 5, odd
    (2048, 4022, "residual_gamma"),  # 21 x 6, odd, partial last tile (the source test's shape)
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
