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
- the refusals: fp32_dest_acc_en off, fp8 kv, an attention sink.
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


def _golden(q, kv, idx, scale, v_dim):
    """float32 on the bf16 inputs: [1, H, S, v_dim]."""
    qf, kvf = q.float()[0], kv.float()[0, 0]
    i = idx[0, 0]
    valid = i >= 0
    sel = kvf[i.clamp(min=0)]  # [S, W, K]
    sc = torch.einsum("hsk,swk->hsw", qf, sel) * scale
    sc = sc.masked_fill(~valid[None], float("-inf"))
    return torch.einsum("hsw,swv->hsv", sc.softmax(-1), sel[..., :v_dim])[None]


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
    sink = ttnn.from_torch(
        torch.zeros(1, 1, 1, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with expect_error(RuntimeError, "attention_sink"):
        _fork(device, q, kv, idx, 512, ckc=_ckc(True), high_precision=True, attention_sink=sink, **kw)
