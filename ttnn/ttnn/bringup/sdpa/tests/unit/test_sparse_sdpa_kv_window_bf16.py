# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.sparse_sdpa(kv_format=BF16, kv_window=...) (fork option KV_WINDOW, deepseek_v41_flash_d_p P.38): a bf16
history table + a bf16 window tensor (the per-chip window region). The index row is [window k_chunk | top-k]; each
part has its own sentinel tail (as ROW_BFP4), chunk 0 is read from kv_window, the top-k chunks from kv. Covered here,
one device, DeepSeek-V4.1's geometry (K_DIM = v_dim 512, top-k 512 after a 128 window, scale 1 with the softmax scale
folded into q, an attention sink), random inputs:
- a full window: the output equals the plain BF16 path's, bit for bit, on one table [window rows; history rows] with
  the same list layout (the chunks line up), for top-k parts of 0 / 1 / 31 / 32 / 200 / 512 keys, with and without
  fp32 DEST, H 32 / 64, and under max_grid (the model's 110-core cap, agent rule 18);
- a partial window (1 / 17 / 32 / 100 / 127 keys, masked within chunk 0 while top-k chunks follow): vs float32 on the
  same values, as close as the BF16 path with the compacted list;
- deterministic (A, B, A x5); a uint8 kv_window and TOPK == k_chunk refused with a BF16 kv."""

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing import determinism

MASKED = -1
K_DIM, WIN, TOPK, KC = 512, 128, 640, 128
TOP_COUNTS = [0, 1, 31, 32, 200, 512]
WIN_COUNTS = [1, 17, 32, 100, 127]


def _ckc(fp32=False):
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True, fp32_dest_acc_en=fp32, packer_l1_acc=False
    )


def _inputs(H, S, Tw, T, n_win, n_top, seed=0):
    """q, window rows, history rows (bf16), the [window | top-k] index row, sink."""
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(1, H, S, K_DIM, generator=g) * K_DIM**-0.5).to(torch.bfloat16)
    win = torch.randn(Tw, K_DIM, generator=g).to(torch.bfloat16)
    hist = (torch.randn(T, K_DIM, generator=g) * torch.pow(2.0, torch.randint(-3, 3, (T, 1), generator=g).float())).to(
        torch.bfloat16
    )
    idx = torch.full((1, 1, S, TOPK), MASKED, dtype=torch.int64)
    for s in range(S):
        w, k = n_win(s), n_top(s)
        idx[0, 0, s, :w] = torch.randperm(Tw, generator=g)[:w]
        idx[0, 0, s, WIN : WIN + k] = torch.randperm(T, generator=g)[:k]
    sink = (torch.randn(H, generator=g) * 0.5).to(torch.bfloat16)
    return q, win, hist, idx, sink


def _dev(t, device, dtype):
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _split(device, q, win, hist, idx, sink, ckc, max_grid=None):
    tq, ti = _dev(q, device, ttnn.bfloat16), _dev(idx.to(torch.int32), device, ttnn.uint32)
    ts = _dev(sink.view(1, 1, 1, -1), device, ttnn.bfloat16)
    tw = _dev(win.view(1, 1, -1, K_DIM), device, ttnn.bfloat16)
    th = _dev(hist.view(1, 1, -1, K_DIM), device, ttnn.bfloat16)
    kw = {} if max_grid is None else {"max_grid": ttnn.CoreCoord(*max_grid)}

    def run():
        return ttnn.bringup.sparse_sdpa(
            tq,
            th,
            ti,
            K_DIM,
            kv_format=ttnn.bringup.SparseKVFormat.BF16,
            scale=1.0,
            k_chunk_size=KC,
            compute_kernel_config=ckc,
            attention_sink=ts,
            kv_window=tw,
            **kw,
        )

    return run


def _combined(device, q, win, hist, idx, sink, ckc, compact):
    """The plain BF16 path on one table [window rows; history rows]: top-k ids shifted by Tw; ``compact`` moves the
    top-k ids up to the window's last valid id (one sentinel tail), else the layout is kept."""
    Tw = win.shape[0]
    table = torch.cat([win, hist]).view(1, 1, -1, K_DIM)
    i = idx.clone()
    top = i[..., WIN:]
    i[..., WIN:] = torch.where(top >= 0, top + Tw, top)
    if compact:
        for s in range(i.shape[2]):
            row = i[0, 0, s]
            keep = row[row >= 0]
            i[0, 0, s] = MASKED
            i[0, 0, s, : len(keep)] = keep
    out = ttnn.bringup.sparse_sdpa(
        _dev(q, device, ttnn.bfloat16),
        _dev(table, device, ttnn.bfloat16),
        _dev(i.to(torch.int32), device, ttnn.uint32),
        K_DIM,
        kv_format=ttnn.bringup.SparseKVFormat.BF16,
        scale=1.0,
        k_chunk_size=KC,
        compute_kernel_config=ckc,
        attention_sink=_dev(sink.view(1, 1, 1, -1), device, ttnn.bfloat16),
    )
    return ttnn.to_torch(out)


def _golden(q, win, hist, idx, sink):
    keys = torch.cat([win, hist]).float()
    Tw = win.shape[0]
    i = idx[0, 0].clone()
    top = i[:, WIN:]
    i[:, WIN:] = torch.where(top >= 0, top + Tw, top)
    valid = i >= 0
    sel = keys[i.clamp(min=0)]
    sc = torch.einsum("hsk,swk->hsw", q.float()[0], sel).masked_fill(~valid[None], float("-inf"))
    m = torch.maximum(sc.amax(-1, keepdim=True), sink.float().view(-1, 1, 1))
    p = torch.exp(sc - m)
    den = p.sum(-1, keepdim=True) + torch.exp(sink.float().view(-1, 1, 1) - m)
    return (torch.einsum("hsw,swv->hsv", p, sel) / den)[None]


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("grid", [None, (11, 10)], ids=["full_grid", "cap110"])
@pytest.mark.parametrize("fp32", [False, True], ids=["dest16", "dest32"])
@pytest.mark.parametrize("H", [32, 64])
def test_full_window_equals_one_table(device, H, fp32, grid):
    S = 2 * len(TOP_COUNTS)
    q, win, hist, idx, sink = _inputs(H, S, 288, 4096, lambda s: WIN, lambda s: TOP_COUNTS[s % len(TOP_COUNTS)])
    got = ttnn.to_torch(_split(device, q, win, hist, idx, sink, _ckc(fp32), grid)())
    exp = _combined(device, q, win, hist, idx, sink, _ckc(fp32), compact=False)
    diff = (got.float() != exp.float()).any(-1)[0]  # [H, S]
    toks = diff.any(0).nonzero().flatten().tolist()
    assert not diff.any(), (
        f"{int(diff.sum())} head-rows differ; tokens {toks} "
        f"(top-k counts {[TOP_COUNTS[t % len(TOP_COUNTS)] for t in toks]}); rel {_rel(got, exp):.4f}"
    )
    print(
        f"H {H} {'dest32' if fp32 else 'dest16'} {grid}: rel vs float32 {_rel(got, _golden(q, win, hist, idx, sink)):.4f}"
    )


def test_partial_window_vs_float32(device):
    H, S = 64, 2 * len(WIN_COUNTS) * 3
    nw = lambda s: WIN_COUNTS[s % len(WIN_COUNTS)]
    nt = lambda s: [0, 37, 512][(s // len(WIN_COUNTS)) % 3]
    q, win, hist, idx, sink = _inputs(H, S, 288, 4096, nw, nt, seed=1)
    run = _split(device, q, win, hist, idx, sink, _ckc())
    got = ttnn.to_torch(run())
    gold = _golden(q, win, hist, idx, sink)
    base = _combined(device, q, win, hist, idx, sink, _ckc(), compact=True)
    r_got, r_base = _rel(got, gold), _rel(base, gold)
    per_tok = ((got.float() - gold).flatten(2).norm(dim=-1) / gold.flatten(2).norm(dim=-1))[0].amax(0)
    print(
        f"partial window: rel vs float32 {r_got:.4f} (one table, compacted list: {r_base:.4f}); worst {per_tok.max():.4f}"
    )
    # the same keys in another chunk split: as close to float32 as the compacted one-table path (one bf16 rounding of
    # the flash state per chunk either way)
    assert r_got <= 1.1 * r_base + 1e-4, (r_got, r_base)
    assert per_tok.max() <= 1.1 * (
        ((base.float() - gold).flatten(2).norm(dim=-1) / gold.flatten(2).norm(dim=-1))[0].amax(0).max()
    ), per_tok.max()

    q2, win2, hist2, idx2, sink2 = _inputs(H, S, 288, 4096, nw, nt, seed=2)
    run_b = _split(device, q2, win2, hist2, idx2, sink2, _ckc())
    determinism.assert_deterministic(run, run_b, label="kv_window_bf16")


def test_refusals(device, expect_error):
    q, win, hist, idx, sink = _inputs(32, 2, 256, 1024, lambda s: WIN, lambda s: 8)
    tq, ti = _dev(q, device, ttnn.bfloat16), _dev(idx.to(torch.int32), device, ttnn.uint32)
    th = _dev(hist.view(1, 1, -1, K_DIM), device, ttnn.bfloat16)
    w8 = _dev(torch.zeros(1, 1, 256, 576, dtype=torch.uint8), device, ttnn.uint8)
    with expect_error(RuntimeError, "a BF16 kv takes a bf16 kv_window"):
        ttnn.bringup.sparse_sdpa(
            tq, th, ti, K_DIM, kv_format=ttnn.bringup.SparseKVFormat.BF16, scale=1.0, k_chunk_size=KC, kv_window=w8
        )
    tw = _dev(win.view(1, 1, -1, K_DIM), device, ttnn.bfloat16)
    ti_win = _dev(idx[..., :KC].contiguous().to(torch.int32), device, ttnn.uint32)
    with expect_error(RuntimeError, "needs TOPK > k_chunk_size"):
        ttnn.bringup.sparse_sdpa(
            tq, th, ti_win, K_DIM, kv_format=ttnn.bringup.SparseKVFormat.BF16, scale=1.0, k_chunk_size=KC, kv_window=tw
        )
