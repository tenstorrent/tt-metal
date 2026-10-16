# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.sparse_sdpa(high_precision=True): Float32 flash running state and exact softmax exp.

The source op keeps the running output and row-sum (L1-accumulated across k_chunks, rescaled by the flash correction
once per chunk) in bf16 and uses the fast approximate exp for the scores. On GLM-5.3's DSA attention (64 heads, latent
512, 2176 indices, k_chunk 128) that gives rel L2 0.0066 against float32 on the same bf16 inputs, with single
head-rows off by 3%. high_precision keeps both in Float32, sums the packed bf16 probabilities (the ones PV multiplies),
keeps 1/sum in Float32 and takes the exact exp. Covered here:
- accuracy vs a float32 torch reference on the same bf16 inputs, GLM geometry (H 64, K_DIM = v_dim 512, scale 1/16,
  17 chunks) and a DeepSeek-like one (H 32, K_DIM 576, v_dim 512), dense rows and rows with a sentinel tail (partial
  and whole masked chunks): rel L2, per-row norm ratio and the output coefficient, each tighter than the source op;
- option off: bit-identical to ttnn.transformer.sparse_sdpa (fp32 dest on and off, bf16 and fp8 kv);
- the program cache: the option is in the key (two programs for the same inputs);
- the refusals: fp32_dest_acc_en off, fp8 kv;
- with an attention sink (one logit per head, added to the softmax denominator only; deepseek_v41_flash_d_p window
  attention): accuracy vs float32 at DeepSeek-V4.1's geometry (H 64, K_DIM = v_dim 512, 128 window ids, scale 1 with
  the softmax scale folded into q, and scale 512^-0.5) and over 4 chunks (the sink rides the SALAD-combined row-sum),
  sinks small and dominant; the sink moves the output (it is not ignored);
- fp32_output=True (with high_precision; deepseek_v41_flash_d_p window attention): a Float32 output, the fp32 DEST
  value untilized exactly (not on the bf16 or TF32 grid), within one bf16 step of the bf16 output and closer to the
  float32 golden; its own program; refused without high_precision.
- fp32_scores=True (with high_precision and fp32_output; deepseek_v41_flash_d_p S.swa.11): Float32 scores and
  probabilities, every Float32 value the FPU reads rounded to TF32 (nearest) in DEST, a float32 q accepted: vs float32
  on the float32 q, closer than the bf16 scores with a bf16 q and much closer with the float32 q (rel 0.0026-0.0068 ->
  0.0009-0.0022), one chunk with a sink, 4 chunks (eltwise max, SALAD), 6 query tile-rows in 2 DST groups; its own
  program; refused without high_precision + fp32_output, and a float32 q without it. (A math-thread SFPU add in the
  row-sum, racing the pack thread's SFPU rounding, failed the 2-group cases with rel 2-400 during development.)
- bf16_probs=True (with fp32_scores and a bf16 q; deepseek_v41_flash_d_p FIX.2): DeepSeek's sparse_attn kernel
  stage by stage (Float32 scores and running max, the row-sum from the TF32 probabilities, then bf16 probabilities
  before P@V, a bf16 output allowed): the probabilities read back exactly through one-hot keys equal
  rne_bf16(rne_tf32(exp(score))) everywhere (with the rounding disabled: 0.26); closer to the kernel than the bf16
  scores, no bias against the TF32 probabilities; the bf16 output is the fp32 one rounded to nearest; its own program;
  refused without fp32_scores or with a float32 q.
"""

import pytest
import torch

import ttnn

MASKED = 0xFFFFFFFF


def _ckc(fp32=True):
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=False
    )


def _inputs(H, S, T, topk, k_dim, n_valid, q_scale, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(1, H, S, k_dim, generator=g) * q_scale).to(torch.bfloat16)
    kv = torch.randn(1, 1, T, k_dim, generator=g).to(torch.bfloat16)
    idx = torch.full((1, 1, S, topk), -1, dtype=torch.int64)
    for s in range(S):
        nv = max(1, min(topk, n_valid(s)))
        idx[0, 0, s, :nv] = torch.randperm(T, generator=g)[:nv]
    return q, kv, idx


def _golden(q, kv, idx, scale, v_dim, sink=None):
    """float32 on the bf16 inputs: [1, H, S, v_dim]. sink [H] (the op's sink tensor values): exp(sink * scale) joins
    each head's softmax denominator only."""
    qf, kvf = q.float()[0], kv.float()[0, 0]
    i = idx[0, 0]
    valid = i >= 0
    sel = kvf[i.clamp(min=0)]  # [S, W, K]
    sc = torch.einsum("hsk,swk->hsw", qf, sel) * scale
    sc = sc.masked_fill(~valid[None], float("-inf"))
    if sink is None:
        return torch.einsum("hsw,swv->hsv", sc.softmax(-1), sel[..., :v_dim])[None]
    m = sc.amax(-1, keepdim=True)
    p = torch.exp(sc - m)
    den = p.sum(-1, keepdim=True) + torch.exp(sink.float().view(-1, 1, 1) * scale - m)
    return (torch.einsum("hsw,swv->hsv", p, sel[..., :v_dim]) / den)[None]


def _dev(t, device, dtype):
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _run(op, fmt, device, q, kv, idx, v_dim, scale, kc, ckc, kv_dtype=ttnn.bfloat16, **kw):
    tq = _dev(q, device, ttnn.bfloat16)
    tkv = _dev(kv.float() if kv_dtype == ttnn.fp8_e4m3 else kv, device, kv_dtype)
    ti = _dev(idx.to(torch.int32), device, ttnn.uint32)
    out = op(tq, tkv, ti, v_dim, kv_format=fmt, scale=scale, k_chunk_size=kc, compute_kernel_config=ckc, **kw)
    return ttnn.to_torch(out).float()


def _fork(device, q, kv, idx, v_dim, scale, kc, ckc, **kw):
    return _run(
        ttnn.bringup.sparse_sdpa, ttnn.bringup.SparseKVFormat.BF16, device, q, kv, idx, v_dim, scale, kc, ckc, **kw
    )


def _source(device, q, kv, idx, v_dim, scale, kc, ckc):
    return _run(
        ttnn.transformer.sparse_sdpa, ttnn.transformer.SparseKVFormat.BF16, device, q, kv, idx, v_dim, scale, kc, ckc
    )


def _stats(got, want):
    rel = ((got - want).norm() / want.norm()).item()
    ratio = got.norm(dim=-1) / want.norm(dim=-1)  # per (head, query row)
    coef = ((got * want).sum() / (want * want).sum()).item()
    return rel, ratio.min().item(), ratio.max().item(), coef


# name -> (H, S, T, topk, k_dim, v_dim, scale, k_chunk, n_valid(s), q scale), (max rel L2, per-row ratio, max |coef - 1|)
# Limits from the measurement (source -> high_precision): glm rel 0.0080 -> 0.0017, ratio [0.973, 1.004] ->
# [0.990, 1.009], coef 1.0014 -> 0.9998; sharp 0.0157 -> 0.0107, [0.972, 1.013] -> [0.974, 1.014], 0.9949 -> 0.9976;
# ds 0.0201 -> 0.0044, [0.972, 1.005] -> [0.994, 1.008], 0.9944 -> 1.0007. In the sharp case both are bound by the
# bf16 scores buffer (large logits), which this option does not change.
CASES = {
    # GLM-5.3 DSA: 2176 ids (2051 + sentinels), k_chunk 128 -> 17 chunks; rows with 1..2176 valid ids
    "glm_h64_k512": (
        (64, 64, 4096, 2176, 512, 512, 1 / 16, 128, lambda s: [2051, 2048, 1, 33, 700, 2176][s % 6], 1.0),
        (0.0025, (0.985, 1.015), 0.001),
    ),
    # sharper softmax (a few keys dominate a row), as a trained model's attention
    "glm_h64_k512_sharp": (
        (64, 32, 4096, 2176, 512, 512, 1 / 16, 128, lambda s: 2051 - 64 * s, 3.0),
        (0.013, (0.965, 1.02), 0.004),
    ),
    # DeepSeek-like: 32 heads, K_DIM 576 (RoPE columns) with V the first 512, default-like scale, 4 chunks
    "ds_h32_k576": (
        (32, 64, 2048, 512, 576, 512, 576**-0.5, 128, lambda s: [512, 300, 129, 128][s % 4], 1.0),
        (0.006, (0.99, 1.012), 0.0015),
    ),
}


@pytest.mark.parametrize("case", list(CASES))
def test_high_precision_accuracy(device, case):
    (H, S, T, topk, k_dim, v_dim, scale, kc, nv, qs), (max_rel, (lo, hi), max_coef) = CASES[case]
    q, kv, idx = _inputs(H, S, T, topk, k_dim, nv, qs)
    want = _golden(q, kv, idx, scale, v_dim)
    src = _stats(_source(device, q, kv, idx, v_dim, scale, kc, _ckc()), want)
    hp = _stats(_fork(device, q, kv, idx, v_dim, scale, kc, _ckc(), high_precision=True), want)
    fmt = "rel {:.5f} ratio [{:.4f}, {:.4f}] coef {:.5f}"
    print(f"{case}: source " + fmt.format(*src) + " | high_precision " + fmt.format(*hp))
    rel, rmin, rmax, coef = hp
    assert torch.isfinite(torch.tensor(hp)).all()
    assert rel <= max_rel, f"rel L2 {rel:.5f} > {max_rel}"
    assert lo <= rmin and rmax <= hi, f"per-row norm ratio [{rmin:.4f}, {rmax:.4f}] outside [{lo}, {hi}]"
    assert abs(coef - 1) <= max_coef, f"coefficient {coef:.5f}"
    assert rel < src[0], f"high_precision rel {rel:.5f} not below the source op's {src[0]:.5f}"
    assert abs(coef - 1) < abs(src[3] - 1), f"coefficient {coef:.5f} not closer to 1 than the source op's {src[3]:.5f}"


@pytest.mark.parametrize("fp32", [True, False], ids=["fp32_dest", "bf16_dest"])
def test_default_is_source_bitwise(device, fp32):
    (H, S, T, topk, k_dim, v_dim, scale, kc, nv, qs), _ = CASES["glm_h64_k512"]
    q, kv, idx = _inputs(H, S, T, topk, k_dim, nv, qs, seed=1)
    a = _source(device, q, kv, idx, v_dim, scale, kc, _ckc(fp32))
    b = _fork(device, q, kv, idx, v_dim, scale, kc, _ckc(fp32))
    assert torch.equal(a, b)


def test_default_is_source_bitwise_fp8_kv(device):
    q, kv, idx = _inputs(32, 32, 1024, 256, 576, lambda s: 256 - 7 * s, 1.0, seed=2)
    kw = dict(scale=576**-0.5, kc=128, ckc=_ckc(True), kv_dtype=ttnn.fp8_e4m3)
    a = _run(ttnn.transformer.sparse_sdpa, ttnn.transformer.SparseKVFormat.FP8_E4M3, device, q, kv, idx, 512, **kw)
    b = _run(ttnn.bringup.sparse_sdpa, ttnn.bringup.SparseKVFormat.FP8_E4M3, device, q, kv, idx, 512, **kw)
    assert torch.equal(a, b)


def test_program_cache_key(device):
    (H, S, T, topk, k_dim, v_dim, scale, kc, nv, qs), _ = CASES["ds_h32_k576"]
    q, kv, idx = _inputs(H, S, T, topk, k_dim, nv, qs, seed=3)
    device.clear_program_cache()
    n0 = device.num_program_cache_entries()
    a = _fork(device, q, kv, idx, v_dim, scale, kc, _ckc(), high_precision=False)
    n1 = device.num_program_cache_entries()
    b = _fork(device, q, kv, idx, v_dim, scale, kc, _ckc(), high_precision=True)
    n2 = device.num_program_cache_entries()
    _fork(device, q, kv, idx, v_dim, scale, kc, _ckc(), high_precision=True)
    assert n1 - n0 >= 1 and n2 - n1 == n1 - n0, "high_precision must be a separate program"
    assert device.num_program_cache_entries() == n2, "a repeated call must hit the program cache"
    assert not torch.equal(a, b)


def test_refusals(device, expect_error):
    q, kv, idx = _inputs(32, 32, 1024, 256, 576, lambda s: 256, 1.0, seed=4)
    kw = dict(scale=576**-0.5, kc=128)
    with expect_error(RuntimeError, "fp32_dest_acc_en"):
        _fork(device, q, kv, idx, 512, ckc=_ckc(False), high_precision=True, **kw)
    with expect_error(RuntimeError, "bf16 q and a BF16 kv"):
        _run(
            ttnn.bringup.sparse_sdpa,
            ttnn.bringup.SparseKVFormat.FP8_E4M3,
            device,
            q,
            kv,
            idx,
            512,
            ckc=_ckc(True),
            kv_dtype=ttnn.fp8_e4m3,
            high_precision=True,
            **kw,
        )


def _sink(device, sink):
    return ttnn.from_torch(
        sink.reshape(1, 1, 1, -1).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# name -> (H, S, T, topk, k_dim, scale, k_chunk, n_valid(s), q scale, (sink spread, sink offset) in logit units),
#         (max rel L2, per-row ratio); the op's sink values are logits / scale (the op multiplies them by scale).
# deepseek_v41_flash_d_p window attention: 64 heads, K = V = 512, 128 window ids (a short row at the sequence start),
# k_chunk 128; the model folds 512^-0.5 into q and passes scale 1 (exact) with the sink as is. Limits from the
# measurement: rel 0.0028 / 0.0023 / 0.0042 / 0.0068, rows within [0.985, 1.012] (dominant sink: small outputs;
# 4 chunks: sharper scores, std 1.5).
SINK_CASES = {
    "ds41_window_scale1": (
        (64, 64, 2176, 128, 512, 1.0, 128, lambda s: [128, 128, 1, 37, 96][s % 5], 0.05, (0.2, 0.0)),
        (0.004, (0.99, 1.01)),
    ),
    "ds41_window_scale_rsqrt512": (
        (64, 64, 2176, 128, 512, 512**-0.5, 128, lambda s: [128, 64, 1][s % 3], 1.0, (0.2, 0.0)),
        (0.004, (0.99, 1.01)),
    ),
    # the sink dominates the denominator of the short rows (9 keys)
    "ds41_window_dominant_sink": (
        (64, 64, 2176, 128, 512, 1.0, 128, lambda s: [128, 9][s % 2], 0.05, (4.0, 3.0)),
        (0.006, (0.98, 1.015)),
    ),
    # 4 chunks: the sink joins the SALAD-combined Float32 row-sum. A power-of-two scale: with 576^-0.5 the sink's
    # exp (like the scores') takes the scale as a bf16 immediate, +0.19 % here, which moves the output coefficient
    # to 0.994 for sink logits of 6-10 (K-27; the model passes scale 1)
    "ds_h32_k576_chunks": (
        (32, 64, 2048, 512, 576, 1 / 16, 128, lambda s: [512, 300, 129, 128][s % 4], 1.0, (1.0, 6.0)),
        (0.009, (0.98, 1.016)),
    ),
}


@pytest.mark.parametrize("case", list(SINK_CASES))
def test_high_precision_sink(device, case):
    (H, S, T, topk, k_dim, scale, kc, nv, qs, (spread, offset)), (max_rel, (lo, hi)) = SINK_CASES[case]
    q, kv, idx = _inputs(H, S, T, topk, k_dim, nv, qs, seed=5)
    g = torch.Generator().manual_seed(6)
    logit = torch.randn(H, generator=g) * spread
    logit = logit.abs() + offset if offset else logit
    sink = (logit / scale).to(torch.bfloat16)
    v_dim = 512
    want = _golden(q, kv, idx, scale, v_dim, sink)
    got = _fork(device, q, kv, idx, v_dim, scale, kc, _ckc(), high_precision=True, attention_sink=_sink(device, sink))
    rel, rmin, rmax, coef = _stats(got, want)
    no_sink = _stats(_golden(q, kv, idx, scale, v_dim), want)[0]
    print(f"{case}: rel {rel:.5f} ratio [{rmin:.4f}, {rmax:.4f}] coef {coef:.5f} (no-sink golden off by {no_sink:.4f})")
    assert torch.isfinite(got).all()
    assert rel <= max_rel, f"rel L2 {rel:.5f} > {max_rel}"
    assert lo <= rmin and rmax <= hi, f"per-row norm ratio [{rmin:.4f}, {rmax:.4f}] outside [{lo}, {hi}]"
    assert no_sink > 3 * max_rel, f"the sink barely moves this case ({no_sink:.5f}): it would not catch a dropped sink"


# fp32_output cases: (CASES or SINK_CASES name, how much closer to the golden than the bf16 output, at least)
FP32_OUT_CASES = ["ds41_window_scale1", "ds41_window_dominant_sink", "ds_h32_k576_chunks", "ds_h32_k576"]


def _case_inputs(case):
    """(q, kv, idx, v_dim, scale, kc, sink tensor values or None) of a CASES / SINK_CASES entry."""
    if case in SINK_CASES:
        (H, S, T, topk, k_dim, scale, kc, nv, qs, (spread, offset)), _ = SINK_CASES[case]
        q, kv, idx = _inputs(H, S, T, topk, k_dim, nv, qs, seed=5)
        logit = torch.randn(H, generator=torch.Generator().manual_seed(6)) * spread
        logit = logit.abs() + offset if offset else logit
        return q, kv, idx, 512, scale, kc, (logit / scale).to(torch.bfloat16)
    (H, S, T, topk, k_dim, v_dim, scale, kc, nv, qs), _ = CASES[case]
    q, kv, idx = _inputs(H, S, T, topk, k_dim, nv, qs, seed=7)
    return q, kv, idx, v_dim, scale, kc, None


@pytest.mark.parametrize("case", FP32_OUT_CASES)
def test_fp32_output(device, case):
    q, kv, idx, v_dim, scale, kc, sink = _case_inputs(case)
    kw = dict(high_precision=True)
    if sink is not None:
        kw["attention_sink"] = _sink(device, sink)
    want = _golden(q, kv, idx, scale, v_dim, sink)
    tq, ti = _dev(q, device, ttnn.bfloat16), _dev(idx.to(torch.int32), device, ttnn.uint32)
    tkv = _dev(kv, device, ttnn.bfloat16)
    call = dict(kv_format=ttnn.bringup.SparseKVFormat.BF16, scale=scale, k_chunk_size=kc, compute_kernel_config=_ckc())
    o32 = ttnn.bringup.sparse_sdpa(tq, tkv, ti, v_dim, fp32_output=True, **call, **kw)
    o16 = ttnn.bringup.sparse_sdpa(tq, tkv, ti, v_dim, **call, **kw)
    assert o32.dtype == ttnn.float32 and o16.dtype == ttnn.bfloat16
    assert list(o32.shape) == list(o16.shape) and o32.layout == ttnn.ROW_MAJOR_LAYOUT
    a, b = ttnn.to_torch(o32), ttnn.to_torch(o16).float()
    assert a.dtype == torch.float32 and torch.isfinite(a).all()
    rel32, rel16 = _stats(a, want)[0], _stats(b, want)[0]
    # half a bf16 step of the bf16 output (the bf16 output is the same value rounded to nearest by the packer)
    ulp = torch.where(b == 0, torch.full_like(b, 2.0**-133), b.abs() * 2.0**-7)
    worst = ((a - b).abs() / ulp).max().item()
    bits = a.view(torch.int32)
    off_bf16 = ((bits & 0xFFFF) != 0).float().mean().item()
    off_tf32 = ((bits & 0x1FFF) != 0).float().mean().item()  # TF32 keeps 10 mantissa bits: the low 13 are zero
    print(
        f"{case}: rel fp32 out {rel32:.5f} | bf16 out {rel16:.5f}; |fp32 - bf16| <= {worst:.3f} bf16 steps; "
        f"off the bf16 grid {off_bf16:.4f}, off the TF32 grid {off_tf32:.4f}"
    )
    assert worst <= 0.5001, f"the fp32 output is {worst:.3f} bf16 steps from the bf16 output (its rounding: <= 0.5)"
    assert off_bf16 > 0.95 and off_tf32 > 0.9, "the fp32 output was rounded (bf16 or TF32 grid)"
    assert rel32 < rel16, f"fp32 output rel {rel32:.5f} not below the bf16 output's {rel16:.5f}"


def test_fp32_output_program_and_refusal(device, expect_error):
    q, kv, idx, v_dim, scale, kc, _ = _case_inputs("ds_h32_k576")
    device.clear_program_cache()
    _fork(device, q, kv, idx, v_dim, scale, kc, _ckc(), high_precision=True)
    n1 = device.num_program_cache_entries()
    _fork(device, q, kv, idx, v_dim, scale, kc, _ckc(), high_precision=True, fp32_output=True)
    n2 = device.num_program_cache_entries()
    assert n2 > n1, "fp32_output must be a separate program"
    with expect_error(RuntimeError, "fp32_output requires high_precision"):
        _fork(device, q, kv, idx, v_dim, scale, kc, _ckc(), fp32_output=True)


# ---------------------------------------------------------------- fp32_scores (deepseek_v41_flash_d_p S.swa.11)
# fp32_scores=True (with high_precision and fp32_output): the scores and the softmax probabilities stay in a Float32 CB
# (default: rounded to bf16) and q may be float32. Golden: float32 on the float32 q. Cases: the DS41 window (sink,
# short rows), 4 chunks (the running max's eltwise max and the SALAD combine), and 6 query tile-rows in 2 DST groups
# (the per-group format restore).
FP32_SCORES_CASES = {
    "ds41_window_scale1": SINK_CASES["ds41_window_scale1"][0],
    "ds_h32_k576_chunks": SINK_CASES["ds_h32_k576_chunks"][0],
    "h192_k128_groups": (192, 32, 1024, 256, 128, 1 / 8, 128, lambda s: [256, 200, 129, 7][s % 4], 2.0, (1.0, 0.0)),
}


def _fp32_scores_inputs(case, seed=11):
    H, S, T, topk, k_dim, scale, kc, nv, qs, (spread, offset) = FP32_SCORES_CASES[case]
    g = torch.Generator().manual_seed(seed)
    _, kv, idx = _inputs(H, S, T, topk, k_dim, nv, qs, seed=seed)
    q = torch.randn(1, H, S, k_dim, generator=g) * qs  # float32, off the bf16 grid
    logit = torch.randn(H, generator=g) * spread
    logit = logit.abs() + offset if offset else logit
    v_dim = min(512, k_dim)
    return q, kv, idx, v_dim, scale, kc, (logit / scale).to(torch.bfloat16)


@pytest.mark.parametrize("case", list(FP32_SCORES_CASES))
def test_fp32_scores(device, case):
    q, kv, idx, v_dim, scale, kc, sink = _fp32_scores_inputs(case)
    want = _golden(q, kv, idx, scale, v_dim, sink)
    tkv, ti = _dev(kv, device, ttnn.bfloat16), _dev(idx.to(torch.int32), device, ttnn.uint32)
    call = dict(
        kv_format=ttnn.bringup.SparseKVFormat.BF16,
        scale=scale,
        k_chunk_size=kc,
        compute_kernel_config=_ckc(),
        attention_sink=_sink(device, sink),
        high_precision=True,
        fp32_output=True,
    )
    q16, q32 = _dev(q.to(torch.bfloat16), device, ttnn.bfloat16), _dev(q, device, ttnn.float32)
    base = ttnn.to_torch(ttnn.bringup.sparse_sdpa(q16, tkv, ti, v_dim, **call)).float()
    s16 = ttnn.to_torch(ttnn.bringup.sparse_sdpa(q16, tkv, ti, v_dim, fp32_scores=True, **call)).float()
    o32 = ttnn.bringup.sparse_sdpa(q32, tkv, ti, v_dim, fp32_scores=True, **call)
    assert o32.dtype == ttnn.float32
    s32 = ttnn.to_torch(o32).float()
    r_base, r_s16, r_s32 = (_stats(x, want) for x in (base, s16, s32))
    fmt = "rel {:.5f} ratio [{:.4f}, {:.4f}] coef {:.5f}"
    print(
        f"{case}: bf16 scores, bf16 q "
        + fmt.format(*r_base)
        + " | fp32 scores, bf16 q "
        + fmt.format(*r_s16)
        + " | fp32 scores, fp32 q "
        + fmt.format(*r_s32)
    )
    for r in (r_s16, r_s32):
        assert torch.isfinite(torch.tensor(r)).all()
        assert 0.98 <= r[1] and r[2] <= 1.02, f"per-row norm ratio [{r[1]:.4f}, {r[2]:.4f}]"
        assert abs(r[3] - 1) <= 0.003, f"coefficient {r[3]:.5f}"
    # bf16 q: the only change is the scores / probabilities buffer; fp32 q adds q's own rounding
    assert r_s16[0] < r_base[0], f"fp32 scores rel {r_s16[0]:.5f} not below the bf16 scores' {r_base[0]:.5f}"
    assert r_s32[0] < 0.6 * r_base[0], f"fp32 scores + fp32 q rel {r_s32[0]:.5f} vs bf16 {r_base[0]:.5f}"


def test_fp32_scores_program_and_refusal(device, expect_error):
    q, kv, idx, v_dim, scale, kc, _ = _case_inputs("ds_h32_k576")
    device.clear_program_cache()
    _fork(device, q, kv, idx, v_dim, scale, kc, _ckc(), high_precision=True, fp32_output=True)
    n1 = device.num_program_cache_entries()
    _fork(device, q, kv, idx, v_dim, scale, kc, _ckc(), high_precision=True, fp32_output=True, fp32_scores=True)
    n2 = device.num_program_cache_entries()
    assert n2 > n1, "fp32_scores must be a separate program"
    with expect_error(RuntimeError, "fp32_scores requires high_precision"):
        _fork(device, q, kv, idx, v_dim, scale, kc, _ckc(), high_precision=True, fp32_scores=True)
    with expect_error(RuntimeError, "float32 only with fp32_scores"):
        tq = _dev(q.float(), device, ttnn.float32)
        ttnn.bringup.sparse_sdpa(
            tq,
            _dev(kv, device, ttnn.bfloat16),
            _dev(idx.to(torch.int32), device, ttnn.uint32),
            v_dim,
            kv_format=ttnn.bringup.SparseKVFormat.BF16,
            scale=scale,
            k_chunk_size=kc,
            compute_kernel_config=_ckc(),
            high_precision=True,
            fp32_output=True,
        )


# ---------------------------------------------------------------- bf16_probs (deepseek_v41_flash_d_p FIX.2)
# bf16_probs=True (with fp32_scores and a bf16 q): DeepSeek's sparse_attn kernel (inference/kernel.py) stage by stage:
# bf16 q / kv, Float32 scores, running max and row-sum, the probabilities rounded to bf16 before P@V, fp32
# accumulation, a bf16 output (fp32_output may be off). Golden: that kernel in float32 torch on the same bf16 inputs
# (P rounded to bf16 after the row-sum, the output rounded to bf16 last). Cases: the fp32_scores ones (one chunk with a
# sink, 4 chunks, 2 DST groups).
def _kernel_golden(q, kv, idx, scale, v_dim, sink, bf16_p=True):
    """DeepSeek's sparse_attn on bf16 q / kv in float32: s = (q . k) scale, m = max, p = exp(s - m), den = sum p +
    exp(sink scale - m), out = (bf16(p) @ v) / den (fp32 p with bf16_p False)."""
    qf, kvf = q.float()[0], kv.float()[0, 0]
    i = idx[0, 0]
    sel = kvf[i.clamp(min=0)]
    sc = torch.einsum("hsk,swk->hsw", qf, sel) * scale
    sc = sc.masked_fill(~(i >= 0)[None], float("-inf"))
    m = sc.amax(-1, keepdim=True)
    p = torch.exp(sc - m)
    den = p.sum(-1, keepdim=True) + torch.exp(sink.float().view(-1, 1, 1) * scale - m)
    if bf16_p:
        p = p.to(torch.bfloat16).float()
    return (torch.einsum("hsw,swv->hsv", p, sel[..., :v_dim]) / den)[None]


@pytest.mark.parametrize("case", list(FP32_SCORES_CASES))
def test_bf16_probs(device, case):
    q, kv, idx, v_dim, scale, kc, sink = _fp32_scores_inputs(case)
    q = q.to(torch.bfloat16)
    want = _kernel_golden(q, kv, idx, scale, v_dim, sink)
    want_fp32p = _kernel_golden(q, kv, idx, scale, v_dim, sink, bf16_p=False)
    tq, tkv = _dev(q, device, ttnn.bfloat16), _dev(kv, device, ttnn.bfloat16)
    ti = _dev(idx.to(torch.int32), device, ttnn.uint32)
    call = dict(
        kv_format=ttnn.bringup.SparseKVFormat.BF16,
        scale=scale,
        k_chunk_size=kc,
        compute_kernel_config=_ckc(),
        attention_sink=_sink(device, sink),
        high_precision=True,
    )
    o16 = ttnn.bringup.sparse_sdpa(tq, tkv, ti, v_dim, fp32_scores=True, bf16_probs=True, **call)
    assert o16.dtype == ttnn.bfloat16 and o16.layout == ttnn.ROW_MAJOR_LAYOUT
    b16 = ttnn.to_torch(o16).float()
    b32 = ttnn.to_torch(
        ttnn.bringup.sparse_sdpa(tq, tkv, ti, v_dim, fp32_scores=True, bf16_probs=True, fp32_output=True, **call)
    ).float()
    t32 = ttnn.to_torch(
        ttnn.bringup.sparse_sdpa(tq, tkv, ti, v_dim, fp32_scores=True, fp32_output=True, **call)
    ).float()
    base = ttnn.to_torch(ttnn.bringup.sparse_sdpa(tq, tkv, ti, v_dim, **call)).float()  # bf16 scores, bf16 out
    r16, r32, rt, rb = (_stats(x, want) for x in (b16, b32, t32, base))
    # the bf16 rounding of P is there: the option moves the output from the TF32 probabilities' by as much as the
    # kernel's bf16 P moves it from fp32 P (an option that did nothing would move it by 0)
    d_dev = ((b32 - t32).norm() / t32.norm()).item()
    d_ker = ((want - want_fp32p).norm() / want_fp32p.norm()).item()
    # the bf16 output is the bf16_probs fp32 output rounded to nearest by the packer (<= half a bf16 step)
    ulp = torch.where(b16 == 0, torch.full_like(b16, 2.0**-133), b16.abs() * 2.0**-7)
    worst = ((b32 - b16).abs() / ulp).max().item()
    bits = b32.view(torch.int32)
    off_bf16 = ((bits & 0xFFFF) != 0).float().mean().item()
    fmt = "rel {:.5f} ratio [{:.4f}, {:.4f}] coef {:.5f}"
    print(
        f"{case} vs the kernel: bf16_probs bf16 out "
        + fmt.format(*r16)
        + " | bf16_probs fp32 out "
        + fmt.format(*r32)
        + " | TF32 probs fp32 out "
        + fmt.format(*rt)
        + " | bf16 scores bf16 out "
        + fmt.format(*rb)
        + f" || P rounding: device {d_dev:.5f}, kernel {d_ker:.5f}; |fp32 - bf16 out| <= {worst:.3f} steps"
    )
    for r in (r16, r32):
        assert torch.isfinite(torch.tensor(r)).all()
        assert 0.98 <= r[1] and r[2] <= 1.02, f"per-row norm ratio [{r[1]:.4f}, {r[2]:.4f}]"
        assert abs(r[3] - 1) <= 0.003, f"coefficient {r[3]:.5f}"
    assert worst <= 0.5001, f"the bf16 output is {worst:.3f} bf16 steps from the fp32 one (its rounding: <= 0.5)"
    assert off_bf16 > 0.9, "the fp32 output of bf16_probs was rounded"
    # (d_dev is printed only: the two programs' own in-tile rounding noise (DS41-16) differs by ~6e-4 even without the
    # rounding of P, measured; test_bf16_probs_values reads the probabilities back exactly)
    # rounded to nearest: no bias (truncating P would shrink the output by ~1e-3 relative)
    assert abs(r32[3] - rt[3]) <= 5e-4, f"bf16_probs coefficient {r32[3]:.5f} vs TF32 probs' {rt[3]:.5f}"
    # the Float32 scores: closer to the kernel than the bf16 scores, both with a bf16 output
    assert r16[0] < rb[0], f"bf16_probs rel {r16[0]:.5f} not below the bf16 scores' {rb[0]:.5f}"


def _rne_bits(x: torch.Tensor, drop: int) -> torch.Tensor:
    """float32 x rounded to nearest-even with the low `drop` mantissa bits cleared (13: TF32, 16: bf16)."""
    i = x.float().contiguous().view(torch.int32)
    i = (i + ((1 << (drop - 1)) - 1) + ((i >> drop) & 1)) & ~((1 << drop) - 1)
    return i.view(torch.float32)


@pytest.mark.parametrize("bf16_probs", [True, False], ids=["bf16_probs", "tf32_probs"])
def test_bf16_probs_values(device, bf16_probs):
    """The probabilities themselves, read back exactly: keys = values = one-hot rows e_j (j < 128), so every score is
    q[j] (one product, exact) and every P@V output column one product p_j x (1/sum) (bf16 x TF32: exact in fp32);
    the fp32 output then gives p_j = out[j] / out[argmax] exactly. Key 0's score is 0, the row max, so the FPU's
    score - max (TF32-precise: it truncates a difference wider than 11 bits) is exact and p = exp(score), the
    accurate fp32 exp, rounded to TF32 (nearest) in DEST. bf16_probs: then to bf16 (nearest): every p_j on the bf16
    grid and equal to rne_bf16(rne_tf32(exp(q_j))) (truncation would match about half); without it the TF32 value."""
    H, S, W, k_dim = 32, 32, 128, 512
    g = torch.Generator().manual_seed(21)
    kv = torch.zeros(1, 1, W, k_dim)
    kv[0, 0, torch.arange(W), torch.arange(W)] = 1.0
    q = torch.zeros(1, H, S, k_dim)
    q[..., 1:W] = -6.0 * torch.rand(1, H, S, W - 1, generator=g) - 2.0**-7  # < 0: key 0 (score 0) is the max
    q = q.to(torch.bfloat16)
    idx = torch.stack([torch.randperm(W, generator=g) for _ in range(S)]).view(1, 1, S, W)
    out = ttnn.to_torch(
        ttnn.bringup.sparse_sdpa(
            _dev(q, device, ttnn.bfloat16),
            _dev(kv.to(torch.bfloat16), device, ttnn.bfloat16),
            _dev(idx.to(torch.int32), device, ttnn.uint32),
            k_dim,
            kv_format=ttnn.bringup.SparseKVFormat.BF16,
            scale=1.0,
            k_chunk_size=128,
            compute_kernel_config=_ckc(),
            high_precision=True,
            fp32_output=True,
            fp32_scores=True,
            bf16_probs=bf16_probs,
        )
    ).float()[..., :W]
    p = (out.double() / out[..., :1].double()).float()  # exact: out[j] = p_j x r with one r per row, p_0 = 1
    tf32 = _rne_bits(torch.exp(q[..., :W].double()).float(), 13)
    want = _rne_bits(tf32, 16) if bf16_probs else tf32
    bits = p.view(torch.int32)
    on_grid = ((bits & 0xFFFF) == 0).float().mean().item()
    match = (p == want).float().mean().item()
    trunc = (p == (tf32.view(torch.int32) & ~0xFFFF).view(torch.float32)).float().mean().item()
    print(
        f"bf16_probs={bf16_probs}: on the bf16 grid {on_grid:.4f}, == the golden {match:.5f}, == truncated {trunc:.4f}"
    )
    if bf16_probs:
        assert match >= 0.999, f"only {match:.5f} of the probabilities equal the golden"
        assert on_grid == 1.0, f"{1 - on_grid:.4f} of the probabilities are off the bf16 grid"
    else:
        # control: the TF32 probabilities are not rounded to bf16. (They reach P@V with ~9 bits, not 11: 0.495 equal
        # rne_tf32(exp), the values on the bf16 grid are the truncated ones; measured, deepseek_v41_flash_d_p FIX.2.)
        assert on_grid < 0.5, f"{on_grid:.4f} of the TF32 probabilities are on the bf16 grid"


def test_bf16_probs_program_and_refusal(device, expect_error):
    q, kv, idx, v_dim, scale, kc, _ = _case_inputs("ds_h32_k576")
    device.clear_program_cache()
    _fork(device, q, kv, idx, v_dim, scale, kc, _ckc(), high_precision=True, fp32_output=True, fp32_scores=True)
    n1 = device.num_program_cache_entries()
    kw = dict(high_precision=True, fp32_output=True, fp32_scores=True, bf16_probs=True)
    _fork(device, q, kv, idx, v_dim, scale, kc, _ckc(), **kw)
    n2 = device.num_program_cache_entries()
    assert n2 > n1, "bf16_probs must be a separate program"
    with expect_error(RuntimeError, "bf16_probs requires fp32_scores"):
        _fork(device, q, kv, idx, v_dim, scale, kc, _ckc(), high_precision=True, fp32_output=True, bf16_probs=True)
    with expect_error(RuntimeError, "bf16_probs requires fp32_scores=true and a bf16 q"):
        ttnn.bringup.sparse_sdpa(
            _dev(q.float(), device, ttnn.float32),
            _dev(kv, device, ttnn.bfloat16),
            _dev(idx.to(torch.int32), device, ttnn.uint32),
            v_dim,
            kv_format=ttnn.bringup.SparseKVFormat.BF16,
            scale=scale,
            k_chunk_size=kc,
            compute_kernel_config=_ckc(),
            **kw,
        )
