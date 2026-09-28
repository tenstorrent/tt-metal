# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""CHANGELOG 5 -- at fp32_dest_acc_en the weight and bias are applied in DEST and y is rounded once.

DO NOT DELETE.  What this pins:

  1. NO cb_normalized AT fp32 DEST WITH A WEIGHT OR BIAS, on both builders.  Pass B computes x * (1/rms) on the
     FPU into fp32 DEST, unpacks gamma / bias with their Row broadcast into a second DEST slot (UnaryBcast) and
     applies them on the SFPU (PcRowBcast in the compute kernel), then packs y once.  Until CHANGELOG 5 x * (1/rms)
     was packed to cb_normalized in bf16 and read back for the weight multiply, so y was rounded twice.  At 16-bit
     DEST the program is the one it was (cb_normalized kept): DEST holds x * (1/rms) at bf16 precision there anyway.
  2. y AGAINST float64 IN EVERY MODE THE PATH TAKES: gamma, bias, gamma + bias, none, a residual with and without
     return_residual_sum, TILE and ROW_MAJOR, interleaved / HEIGHT / WIDTH (the cross-core combine, whose pass B
     used to run gamma first) / BLOCK placements, a partial last tile, multi-row blocks, several width chunks,
     bf16 / fp32 / bfp8 inputs, both DEST widths.  The reference is float64 from the op's own rounded inputs (the
     residual sum t is the bf16 tensor the op materializes and returns), so at fp32 DEST with bf16 in / out the
     only error left is the output rounding: rel L2 ~0.0017 (the bf16 floor for these inputs).  The limit 0.0021
     fails the double-rounded CHANGELOG 4 program (0.0023-0.0024 on these inputs, measured) and anything worse.
"""

from __future__ import annotations

import pytest
import torch

import ttnn

import ttnn.bringup.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as PD
from eval.sharding import auto_shard_config, shard_config
from ttnn.bringup.rms_norm_ttnn.tests.unit.builders import BUILDER_IDS, BUILDERS

_ML = ttnn.TensorMemoryLayout
_READER_CT_BLOCK_ROWS = 4  # packed: block_rows | (txn_rows - 1) << 16

MODES = {  # (weight, bias, residual)
    "none": (False, False, False),
    "gamma": (True, False, False),
    "bias": (False, True, False),
    "gamma_bias": (True, True, False),
    "residual_gamma": (True, False, True),
    "residual_gamma_bias": (True, True, True),
}


def _config(fp32):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi4
    c.fp32_dest_acc_en = fp32
    c.math_approx_mode = False
    return c


def _torch_dtype(dtype):
    return torch.float32 if dtype == ttnn.float32 else torch.bfloat16


@pytest.mark.parametrize("builder", BUILDER_IDS)
@pytest.mark.parametrize("fp32", [True, False], ids=["fp32dest", "bf16dest"])
@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("H, W", [(2048, 4096), (256, 1024), (256, 1000)], ids=["2048x4096", "256x1024", "256x1000"])
def test_cb_normalized_only_off_the_dest_path(device, builder, fp32, mode, H, W):
    has_w, has_b, has_r = MODES[mode]
    kw = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tx = ttnn.from_torch(torch.zeros(1, 1, H, W).bfloat16(), **kw)
    pc = lambda: ttnn.from_torch(torch.zeros(1, 1, 1, W).bfloat16(), **kw)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, H, W]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    desc = BUILDERS[builder](
        tx,
        out,
        weight=pc() if has_w else None,
        bias=pc() if has_b else None,
        residual=ttnn.from_torch(torch.zeros(1, 1, H, W).bfloat16(), **kw) if has_r else None,
        epsilon=1e-6,
        compute_kernel_config=_config(fp32),
    )
    cbs = {fd.buffer_index for cb in desc.cbs for fd in cb.format_descriptors}
    want = (has_w or has_b) and not fp32
    assert (PD.CB_NORMALIZED in cbs) == want, f"cb_normalized allocated={PD.CB_NORMALIZED in cbs}, want {want}"


# (shape, layout, placement, mode, dtype[, explicit shard]): each row is one regime / operand combination of
# the DEST path.  A residual returns its sum (return_residual_sum) on TILE with "residual_gamma"; "residual_gamma_bias"
# and ROW_MAJOR (which has no residual-sum output) return y only.
_BLOCK_28 = ([896, 128], (8, 8))  # 28 tile-rows per core: multi-row blocks
_HEIGHT_16 = ([512, 256], (4, 1))  # 16 tile-rows per core
CASES = [
    ((1, 1, 256, 1024), ttnn.TILE_LAYOUT, _ML.INTERLEAVED, "gamma", ttnn.bfloat16),  # RESIDENT, block_rows > 1
    ((1, 1, 256, 1024), ttnn.TILE_LAYOUT, _ML.INTERLEAVED, "bias", ttnn.bfloat16),
    ((1, 1, 256, 1024), ttnn.TILE_LAYOUT, _ML.INTERLEAVED, "gamma_bias", ttnn.bfloat16),
    ((1, 1, 256, 1024), ttnn.TILE_LAYOUT, _ML.INTERLEAVED, "none", ttnn.bfloat16),
    ((1, 1, 256, 1024), ttnn.TILE_LAYOUT, _ML.INTERLEAVED, "residual_gamma", ttnn.bfloat16),
    ((1, 1, 256, 1024), ttnn.TILE_LAYOUT, _ML.INTERLEAVED, "residual_gamma_bias", ttnn.bfloat16),
    ((1, 1, 256, 1000), ttnn.TILE_LAYOUT, _ML.INTERLEAVED, "gamma_bias", ttnn.bfloat16),  # partial last tile
    ((1, 1, 256, 1000), ttnn.ROW_MAJOR_LAYOUT, _ML.INTERLEAVED, "gamma", ttnn.bfloat16),
    ((1, 1, 256, 1024), ttnn.ROW_MAJOR_LAYOUT, _ML.INTERLEAVED, "residual_gamma_bias", ttnn.bfloat16),
    ((1, 1, 2048, 4096), ttnn.TILE_LAYOUT, _ML.INTERLEAVED, "gamma", ttnn.bfloat16),  # MiMo: 2 chunks, 1-row blocks
    ((1, 1, 2048, 4096), ttnn.TILE_LAYOUT, _ML.INTERLEAVED, "residual_gamma", ttnn.bfloat16),  # 3 chunks of 43
    ((1, 1, 1024, 1024), ttnn.TILE_LAYOUT, _ML.HEIGHT_SHARDED, "gamma_bias", ttnn.bfloat16),
    ((1, 1, 256, 2048), ttnn.TILE_LAYOUT, _ML.WIDTH_SHARDED, "gamma", ttnn.bfloat16),  # cross-core combine
    ((1, 1, 256, 2048), ttnn.TILE_LAYOUT, _ML.WIDTH_SHARDED, "residual_gamma_bias", ttnn.bfloat16),
    ((1, 1, 1024, 1024), ttnn.TILE_LAYOUT, _ML.BLOCK_SHARDED, "gamma", ttnn.bfloat16),
    ((1, 1, 256, 1024), ttnn.ROW_MAJOR_LAYOUT, _ML.WIDTH_SHARDED, "gamma_bias", ttnn.bfloat16),  # the BAND scheme
    ((1, 1, 256, 1024), ttnn.TILE_LAYOUT, _ML.INTERLEAVED, "gamma_bias", ttnn.float32),
    ((1, 1, 256, 1024), ttnn.TILE_LAYOUT, _ML.INTERLEAVED, "residual_gamma", ttnn.float32),
    ((1, 1, 256, 1024), ttnn.TILE_LAYOUT, _ML.INTERLEAVED, "gamma_bias", ttnn.bfloat8_b),
    ((1, 1, 1024, 1024), ttnn.TILE_LAYOUT, _ML.BLOCK_SHARDED, "gamma", ttnn.bfloat8_b),
    ((1, 1, 7168, 1024), ttnn.TILE_LAYOUT, _ML.BLOCK_SHARDED, "gamma", ttnn.bfloat16, _BLOCK_28),
    ((1, 1, 7168, 1024), ttnn.TILE_LAYOUT, _ML.BLOCK_SHARDED, "residual_gamma_bias", ttnn.bfloat16, _BLOCK_28),
    ((1, 1, 2048, 256), ttnn.TILE_LAYOUT, _ML.HEIGHT_SHARDED, "gamma_bias", ttnn.bfloat16, _HEIGHT_16),
    ((1, 1, 2048, 256), ttnn.TILE_LAYOUT, _ML.HEIGHT_SHARDED, "residual_gamma", ttnn.bfloat16, _HEIGHT_16),
]
_MULTI_ROW = [c for c in CASES if len(c) > 5]
_ML_ID = {_ML.INTERLEAVED: "int", _ML.HEIGHT_SHARDED: "height", _ML.WIDTH_SHARDED: "width", _ML.BLOCK_SHARDED: "block"}
_DT_ID = {ttnn.bfloat16: "bf16", ttnn.float32: "fp32", ttnn.bfloat8_b: "bfp8"}


def _case_id(c):
    shape, layout, ml, mode, dt = c[:5]
    shard = "-shard" if len(c) > 5 else ""
    return f"{shape[-2]}x{shape[-1]}-{'tile' if layout == ttnn.TILE_LAYOUT else 'rm'}-{_ML_ID[ml]}{shard}-{mode}-{_DT_ID[dt]}"


def _memory_config(case, device):
    shape, layout, ml, _, dtype = case[:5]
    if ml == _ML.INTERLEAVED:
        return ttnn.DRAM_MEMORY_CONFIG
    if len(case) > 5:
        return shard_config(case[5][0], case[5][1], ml, layout=layout, dtype=dtype, device=device)
    return auto_shard_config(list(shape), ml, layout=layout, dtype=dtype, device=device)


def _limit(dtype, fp32):
    # fp32 DEST, one output rounding: the output format's floor (bf16 ~0.0017, measured on these inputs).  bfp8
    # (shared exponent per 16) and fp32 (x and the stat enter the FPU as tf32) have their own floors.
    # 16-bit DEST keeps its old program; its limit is test_rms_norm_ttnn_fp32_stats.py's.
    if dtype == ttnn.bfloat8_b:
        return 0.012
    if dtype == ttnn.float32:
        return 0.001 if fp32 else 0.006
    return 0.0021 if fp32 else 0.0065


@pytest.mark.parametrize("fp32", [True, False], ids=["fp32dest", "bf16dest"])
@pytest.mark.parametrize("case", CASES, ids=[_case_id(c) for c in CASES])
def test_y_against_float64(device, case, fp32):
    shape, layout, memory_layout, mode, dtype = case[:5]
    has_w, has_b, has_r = MODES[mode]
    ret_t = has_r and layout == ttnn.TILE_LAYOUT and mode == "residual_gamma"
    H, W = shape[-2], shape[-1]
    td = _torch_dtype(dtype)
    torch.manual_seed(0)
    x = (torch.randn(shape) * (0.5 + torch.rand(*shape[:-1], 1) * 4)).to(td)
    r = torch.randn(shape).to(td) if has_r else None
    # per-channel operands in the input dtype (bfp8 needs TILE; a bf16 weight on a bfp8 input is the model form)
    pdt = ttnn.bfloat16 if dtype == ttnn.bfloat8_b else dtype
    ptd = _torch_dtype(pdt)
    w = (1 + 0.5 * torch.randn(W)).to(ptd) if has_w else None
    b = (0.5 * torch.randn(W)).to(ptd) if has_b else None

    mc = _memory_config(case, device)
    tx = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=mc)
    tr = ttnn.from_torch(r, dtype=dtype, layout=layout, device=device, memory_config=mc) if has_r else None
    tw = ttnn.from_torch(w.reshape(1, 1, 1, W), dtype=pdt, layout=layout, device=device) if has_w else None
    tb = ttnn.from_torch(b.reshape(1, 1, 1, W), dtype=pdt, layout=layout, device=device) if has_b else None
    cfg = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32,
        packer_l1_acc=False,
    )
    kwargs = dict(weight=tw, bias=tb, epsilon=1e-6, compute_kernel_config=cfg, memory_config=mc)
    if has_r:
        kwargs.update(residual_input_tensor=tr, return_residual_sum=ret_t)
    res = ttnn.bringup.rms_norm(tx, **kwargs)
    y_t, t_t = res if ret_t else (res, None)
    assert y_t.layout == layout and y_t.dtype == dtype
    y = ttnn.to_torch(y_t).double()

    # float64 reference from what the op actually consumes: x (and t) as stored on device
    t = ttnn.to_torch(tx).double()
    if has_r:
        t_want = t + ttnn.to_torch(tr).double()
        if ret_t:
            t_dev = ttnn.to_torch(t_t).double()
            t_rel = ((t_dev - t_want).norm() / t_want.norm()).item()
            # t = x + r on the FPU (tf32 operands), packed in the input format
            assert t_rel < (0.001 if dtype == ttnn.float32 and fp32 else 0.006), f"residual sum rel L2 {t_rel:.5f}"
            t = t_dev  # y is normalized from the materialized t
        else:
            # y is normalized from the t the op materializes in cb_x_sum, which is not bf16_rne(x + r) (the FPU
            # add's own rounding; ~11% of elements differ by one ulp).  Read that t back from a TILE interleaved
            # call that returns it.
            kw_t = dict(dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            _, t_ret = ttnn.bringup.rms_norm(
                ttnn.from_torch(ttnn.to_torch(tx), **kw_t),
                residual_input_tensor=ttnn.from_torch(ttnn.to_torch(tr), **kw_t),
                epsilon=1e-6,
                compute_kernel_config=cfg,
                return_residual_sum=True,
            )
            t = ttnn.to_torch(t_ret).double()
    ref = t * torch.rsqrt((t * t).mean(-1, keepdim=True) + 1e-6)
    if has_w:
        ref = ref * ttnn.to_torch(tw).double().reshape(W)
    if has_b:
        ref = ref + ttnn.to_torch(tb).double().reshape(W)
    assert torch.isfinite(y).all(), "output carries Inf/NaN"
    rel = ((y - ref).norm() / ref.norm()).item()
    lim = _limit(dtype, fp32)
    print(f"{_case_id(case)} fp32_dest={fp32}: rel L2 {rel:.5f} (limit {lim})")
    assert rel < lim, f"rel L2 {rel:.5f} >= {lim}"


def test_cases_reach_multi_row_blocks_and_several_chunks(device):
    """At fp32 DEST the explicit-shard cases run multi-row blocks and the 2048x4096 ones several width chunks, so
    the DEST path's block walk and its per-chunk gamma offsets are both exercised."""
    kw = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def reader_ct(case):
        shape, _, _, mode, _ = case[:5]
        has_w, has_b, has_r = MODES[mode]
        W = shape[-1]
        mc = _memory_config(case, device)
        tx = ttnn.from_torch(torch.zeros(shape).bfloat16(), memory_config=mc, **kw)
        pc = lambda: ttnn.from_torch(torch.zeros(1, 1, 1, W).bfloat16(), **kw)
        out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
        d = BUILDERS["cpp"](
            tx,
            out,
            weight=pc() if has_w else None,
            bias=pc() if has_b else None,
            residual=ttnn.from_torch(torch.zeros(shape).bfloat16(), memory_config=mc, **kw) if has_r else None,
            epsilon=1e-6,
            compute_kernel_config=_config(True),
        )
        return list(d.kernels[0].compile_time_args)

    for case in _MULTI_ROW:
        assert reader_ct(case)[_READER_CT_BLOCK_ROWS] & 0xFFFF > 1, f"{_case_id(case)} is not multi-row"
    many = ((1, 1, 2048, 4096), ttnn.TILE_LAYOUT, _ML.INTERLEAVED, "gamma", ttnn.bfloat16)
    assert reader_ct(many)[3] > 1  # NUM_W_CHUNKS


@pytest.mark.parametrize(
    "w_dtype, b_dtype",
    [(ttnn.float32, ttnn.bfloat16), (ttnn.bfloat8_b, ttnn.bfloat8_b), (ttnn.bfloat16, ttnn.float32)],
    ids=["w_fp32-b_bf16", "w_bfp8-b_bfp8", "w_bf16-b_fp32"],
)
def test_per_channel_formats_in_dest(device, w_dtype, b_dtype):
    """bf16 activations with weight / bias in other formats: UnaryBcast unpacks each in its own format."""
    H, W = 256, 1024
    torch.manual_seed(0)
    x = (torch.randn(1, 1, H, W) * (0.5 + torch.rand(1, 1, H, 1) * 4)).bfloat16()
    kw = dict(layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, **kw)
    tw = ttnn.from_torch((1 + 0.5 * torch.randn(1, 1, 1, W)), dtype=w_dtype, **kw)
    tb = ttnn.from_torch(0.5 * torch.randn(1, 1, 1, W), dtype=b_dtype, **kw)
    cfg = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    y = ttnn.to_torch(ttnn.bringup.rms_norm(tx, weight=tw, bias=tb, epsilon=1e-6, compute_kernel_config=cfg)).double()
    t = x.double()
    ref = t * torch.rsqrt((t * t).mean(-1, keepdim=True) + 1e-6)
    ref = ref * ttnn.to_torch(tw).double().reshape(W) + ttnn.to_torch(tb).double().reshape(W)
    rel = ((y - ref).norm() / ref.norm()).item()
    print(f"w {w_dtype} b {b_dtype}: rel L2 {rel:.5f}")
    assert rel < 0.0021, f"rel L2 {rel:.5f}"
