# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.moe_grouped_topk's fp32_activation option (deepseek_v41_flash_d_p P.22): the score activation compiled
with the fp32 SFPU polynomials (INP_FLOAT32, as the stock unary op does for an fp32 input).

Without it the kernel gets the bf16-grade softplus (degree 6 on [0, 5], 0 below -5): sqrt(softplus(x)) ~8e-4
relative off on [-3, 9] and up to 0.08 absolute below, which swaps near-tied experts even with fp32_selection.
    test_activation_matches_stock   sqrtsoftplus keys (biased_scores, zero bias) == stock ttnn.sqrt(ttnn.softplus(x))
                                    on fp32 bit for bit, over a logit sweep [-30, 30]; and within 5e-5 of float64.
                                    Off: the old error (> 1e-2 absolute somewhere in the sweep).
    test_selection_and_weights      DeepSeek-V4.1 routing (1 group, top 6 of 96 / 384, route_scale 1.5, a bias for
                                    the selection only), fp32_selection on: the selected sets equal float64's on every
                                    row whose 6th / 7th key gap is > 1e-5; the bf16 weights within one bf16 rounding
                                    of float64's (rel <= 2.5e-3) on those rows. Off: rows select other experts.
    test_sigmoid_unchanged          sigmoid (DeepSeek-V3 / Kimi) outputs bit-identical with the option on and off.
    test_off_is_source              option off == ttnn.experimental.deepseek_prefill.moe_grouped_topk, bit for bit.
"""

import pytest
import torch

import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG


def _dev(t, device, dtype=ttnn.float32):
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)


def _gate(device, logits, bias, k, scale, *, act, score_func="sqrtsoftplus", dump=False, op=None):
    rows, e = logits.shape[-2:]
    ld, bd = _dev(logits.reshape(1, 1, rows, e), device), _dev(bias.expand(rows, e).reshape(1, 1, rows, e), device)
    kw = dict(
        n_groups=1,
        summed_experts_per_group=1,
        topk_groups=1,
        n_activated_experts=k,
        route_scale=scale,
        epsilon=1e-20,
        stable_sort=True,
        score_func=score_func,
        fp32_selection=True,
        memory_config=DRAM,
    )
    dd = _dev(torch.zeros(1, 1, rows, e), device) if dump else None
    if dump:
        kw["biased_scores"] = dd
    if op is None:
        w, i = ttnn.bringup.moe_grouped_topk(ld, bd, fp32_activation=act, **kw)
    else:
        w, i = op(ld, bd, **kw)
    out = (
        ttnn.to_torch(w).float().reshape(rows, -1)[:, :k],
        ttnn.to_torch(i).to(torch.int64).reshape(rows, -1)[:, :k],
        ttnn.to_torch(dd).double().reshape(rows, e) if dump else None,
    )
    for t in (ld, bd, w, i) + ((dd,) if dump else ()):
        ttnn.deallocate(t)
    return out


def _f64(logits, bias, k, scale):
    sc = torch.nn.functional.softplus(logits.double()).sqrt()
    key = sc + bias.double()
    idx = key.topk(k, dim=-1)[1]
    w = sc.gather(-1, idx)
    w = w / (w.sum(-1, keepdim=True) + 1e-20) * scale
    srt = key.sort(-1, descending=True)[0]
    return idx, w, srt[:, k - 1] - srt[:, k]


def test_activation_matches_stock(device):
    rows, e = 256, 96
    x = torch.linspace(-30, 30, rows * e).reshape(rows, e).float()
    zero = torch.zeros(e)
    _, _, keys = _gate(device, x, zero, 6, 1.5, act=True, dump=True)
    xd = _dev(x.reshape(1, 1, rows, e), device)
    sp = ttnn.softplus(xd, beta=1.0, threshold=20.0, memory_config=DRAM)
    sq = ttnn.sqrt(sp, memory_config=DRAM)
    stock = ttnn.to_torch(sq).double().reshape(rows, e)
    for t in (xd, sp, sq):
        ttnn.deallocate(t)
    assert torch.equal(keys, stock), f"max |fork - stock| {(keys - stock).abs().max().item():.3e}"
    want = torch.nn.functional.softplus(x.double()).sqrt()
    err = (keys - want).abs().max().item()
    assert err <= 5e-5, f"fp32_activation: max |keys - float64| {err:.3e} > 5e-5"
    _, _, keys_off = _gate(device, x, zero, 6, 1.5, act=False, dump=True)
    err_off = (keys_off - want).abs().max().item()
    assert err_off > 1e-2, f"option off should keep the source's bf16-grade softplus (max err {err_off:.3e})"


@pytest.mark.parametrize("experts", [96, 384], ids=["ds41-scaled-96e", "ds41-384e"])
@pytest.mark.parametrize("rows", [1024, 64], ids=["rows1024", "rows64"])
def test_selection_and_weights(device, experts, rows):
    torch.manual_seed(experts + rows)
    k, scale = 6, 1.5
    logits = torch.randn(rows, experts) * 2.0 - 2.0  # DS41-like: most logits negative, a top tail near 0..4
    bias = torch.randn(experts) * 0.05
    idx64, w64, gap = _f64(logits, bias, k, scale)
    clear = gap > 1e-5
    assert clear.sum() > 0.9 * rows
    w, i, _ = _gate(device, logits, bias, k, scale, act=True)
    got, want = [frozenset(r.tolist()) for r in i], [frozenset(r.tolist()) for r in idx64]
    bad = [r for r in range(rows) if clear[r] and got[r] != want[r]]
    assert not bad, f"{len(bad)} clear rows select other experts than float64 (first {bad[:5]})"
    assert all(len(s) == k and max(s) < experts for s in got)
    dense_got = torch.zeros(rows, experts, dtype=torch.float64).scatter_(1, i, w.double())
    dense_want = torch.zeros(rows, experts, dtype=torch.float64).scatter_(1, idx64, w64)
    m = clear.nonzero().flatten()
    rel = ((dense_got[m] - dense_want[m]).norm() / dense_want[m].norm()).item()
    assert rel <= 2.5e-3, f"weights rel {rel:.3e} vs float64 (one bf16 rounding ~1.6e-3)"
    # the option matters: off, the bf16-grade softplus moves keys by up to 8e-2
    _, i_off, _ = _gate(device, logits, bias, k, scale, act=False)
    off = sum(frozenset(i_off[r].tolist()) != want[r] for r in range(rows))
    print(f"experts {experts} rows {rows}: fp32_activation rows != float64 0 of {int(clear.sum())} clear; off {off}")


def test_sigmoid_unchanged(device):
    torch.manual_seed(1)
    rows, e = 128, 384
    logits, bias = torch.randn(rows, e) * 3, torch.randn(e) * 0.1
    on = _gate(device, logits, bias, 8, 2.5, act=True, score_func="sigmoid", dump=True)
    off = _gate(device, logits, bias, 8, 2.5, act=False, score_func="sigmoid", dump=True)
    for a, b in zip(on, off):
        assert torch.equal(a, b)


def test_off_is_source(device):
    torch.manual_seed(2)
    rows, e = 128, 96
    logits, bias = torch.randn(rows, e) * 2 - 2, torch.randn(e) * 0.05
    fork = _gate(device, logits, bias, 6, 1.5, act=False, dump=True)
    src = _gate(
        device, logits, bias, 6, 1.5, act=None, dump=True, op=ttnn.experimental.deepseek_prefill.moe_grouped_topk
    )
    for a, b in zip(fork, src):
        assert torch.equal(a, b)
