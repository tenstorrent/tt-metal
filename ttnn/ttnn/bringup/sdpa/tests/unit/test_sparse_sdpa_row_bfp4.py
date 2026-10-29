# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.sparse_sdpa(kv_format=ROW_BFP4, kv_window=...): bfloat4_b row records + a bf16 window tensor
(deepseek_v41_flash_d_p 4-bit compressed KV). The index row is [window k_chunk | top-k]; each part has its own
sentinel tail. Covered here, one device, DeepSeek-V4.1's geometry (K_DIM = v_dim 512, top-k 512 after a 128 window,
scale 1 with the softmax scale folded into q, an attention sink):
- a full window: the output equals the BF16 path's, bit for bit, on one bf16 table holding the same values (the
  bfloat4_b values are exact in bf16, and the chunks line up), for top-k parts of 0 / 1 / 31 / 32 / 200 / 512 keys;
- a partial window (1 / 17 / 32 / 100 keys, masked within chunk 0 while top-k chunks follow): vs float32 on the same
  values, as close as the BF16 path with the compacted list the model uses today;
- deterministic (A, B, A x5); refused without kv_window, and kv_window refused with another format;
- a bfloat8_b window (kv_window as 576 B records) equals the bf16 window holding the same values, bit for bit, full and
  partial windows (lowbit, DS41-150); kv_format ROW_BFP8 (a window-only layer: kv = the records, TOPK == k_chunk)
  equals the BF16 path on the same values; both deterministic; their refusals."""

import importlib.util
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing import determinism

_spec = importlib.util.spec_from_file_location(
    "ckv_bfp4_reference", Path(__file__).parents[3] / "ckv_bfp4_ttnn" / "tests" / "reference.py"
)
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)

MASKED = -1
K_DIM, WIN, TOPK, KC = 512, 128, 640, 128


def _ckc(fp32=False):
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True, fp32_dest_acc_en=fp32, packer_l1_acc=False
    )


def _inputs(H, S, Tw, T, n_win, n_top, seed=0):
    """q, window rows (bf16), row records (uint8) and their bf16 values, the [window | top-k] index row, sink."""
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(1, H, S, K_DIM, generator=g) * K_DIM**-0.5).to(torch.bfloat16)
    win = torch.randn(Tw, K_DIM, generator=g).to(torch.bfloat16)
    x = torch.randn(T, K_DIM, generator=g) * torch.pow(2.0, torch.randint(-3, 3, (T, 1), generator=g).float())
    rows = ref.encode_rows(x.bfloat16())
    vals = ref.decode_rows(rows).to(torch.bfloat16)  # exact in bf16
    idx = torch.full((1, 1, S, TOPK), MASKED, dtype=torch.int64)
    for s in range(S):
        w, k = n_win(s), n_top(s)
        idx[0, 0, s, :w] = torch.randperm(Tw, generator=g)[:w]
        idx[0, 0, s, WIN : WIN + k] = torch.randperm(T, generator=g)[:k]
    sink = (torch.randn(H, generator=g) * 0.5).to(torch.bfloat16)
    return q, win, rows, vals, idx, sink


def _dev(t, device, dtype):
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _bfp4(device, q, win, rows, idx, sink, ckc):
    tq, ti, ts = _dev(q, device, ttnn.bfloat16), _dev(idx.to(torch.int32), device, ttnn.uint32), None
    ts = _dev(sink.view(1, 1, 1, -1), device, ttnn.bfloat16)
    tw = _dev(win.view(1, 1, -1, K_DIM), device, ttnn.bfloat16)
    tr = _dev(rows.view(1, 1, -1, rows.shape[-1]), device, ttnn.uint8)

    def run():
        return ttnn.bringup.sparse_sdpa(
            tq,
            tr,
            ti,
            K_DIM,
            kv_format=ttnn.bringup.SparseKVFormat.ROW_BFP4,
            scale=1.0,
            k_chunk_size=KC,
            compute_kernel_config=ckc,
            attention_sink=ts,
            kv_window=tw,
        )

    return run


def _bf16_combined(device, q, win, vals, idx, sink, ckc, compact):
    """The BF16 path on one table [window rows; record values]: top-k ids shifted by Tw; ``compact`` moves the top-k
    ids up to the window's last valid id (the list the model builds today), else the layout is kept."""
    Tw = win.shape[0]
    table = torch.cat([win, vals]).view(1, 1, -1, K_DIM)
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


def _golden(q, win, vals, idx, sink):
    keys = torch.cat([win, vals]).float()
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


TOP_COUNTS = [0, 1, 31, 32, 200, 512]


@pytest.mark.parametrize("fp32", [False, True], ids=["dest16", "dest32"])
@pytest.mark.parametrize("H", [32, 64])
def test_full_window_equals_bf16_on_the_same_values(device, H, fp32):
    S = 2 * len(TOP_COUNTS)
    q, win, rows, vals, idx, sink = _inputs(H, S, 256, 4096, lambda s: WIN, lambda s: TOP_COUNTS[s % len(TOP_COUNTS)])
    run = _bfp4(device, q, win, rows, idx, sink, _ckc(fp32))
    got = ttnn.to_torch(run())
    exp = _bf16_combined(device, q, win, vals, idx, sink, _ckc(fp32), compact=False)
    diff = (got.float() != exp.float()).any(-1)[0]  # [H, S]
    toks = diff.any(0).nonzero().flatten().tolist()
    assert not diff.any(), (
        f"{int(diff.sum())} head-rows differ; tokens {toks} (top-k counts {[TOP_COUNTS[t % len(TOP_COUNTS)] for t in toks]}); "
        f"rel {_rel(got, exp):.4f}"
    )
    gold = _golden(q, win, vals, idx, sink)
    print(f"H {H} {'dest32' if fp32 else 'dest16'}: rel vs float32 {_rel(got, gold):.4f}")


WIN_COUNTS = [1, 17, 32, 100, 127]


def test_partial_window_vs_float32(device):
    H, S = 64, 2 * len(WIN_COUNTS) * 3
    nw = lambda s: WIN_COUNTS[s % len(WIN_COUNTS)]
    nt = lambda s: [0, 37, 512][(s // len(WIN_COUNTS)) % 3]
    q, win, rows, vals, idx, sink = _inputs(H, S, 256, 4096, nw, nt, seed=1)
    run = _bfp4(device, q, win, rows, idx, sink, _ckc())
    got = ttnn.to_torch(run())
    gold = _golden(q, win, vals, idx, sink)
    base = _bf16_combined(device, q, win, vals, idx, sink, _ckc(), compact=True)
    r_got, r_base = _rel(got, gold), _rel(base, gold)
    per_tok = ((got.float() - gold).flatten(2).norm(dim=-1) / gold.flatten(2).norm(dim=-1))[0].amax(0)
    print(
        f"partial window: rel vs float32 {r_got:.4f} (BF16 path, compacted list: {r_base:.4f}); worst token {per_tok.max():.4f}"
    )
    assert r_got <= 1.1 * r_base + 1e-4, (r_got, r_base)
    assert per_tok.max() < 0.05, per_tok.max()

    q2, win2, rows2, _, idx2, sink2 = _inputs(H, S, 256, 4096, nw, nt, seed=2)
    run_b = _bfp4(device, q2, win2, rows2, idx2, sink2, _ckc())
    determinism.assert_deterministic(run, run_b, label="row_bfp4")


def test_refusals(device, expect_error):
    q, win, rows, vals, idx, sink = _inputs(32, 2, 256, 1024, lambda s: WIN, lambda s: 8)
    tq, ti = _dev(q, device, ttnn.bfloat16), _dev(idx.to(torch.int32), device, ttnn.uint32)
    tr = _dev(rows.view(1, 1, -1, rows.shape[-1]), device, ttnn.uint8)
    tw = _dev(win.view(1, 1, -1, K_DIM), device, ttnn.bfloat16)
    with expect_error(RuntimeError, "kv_window"):
        ttnn.bringup.sparse_sdpa(tq, tr, ti, K_DIM, kv_format=ttnn.bringup.SparseKVFormat.ROW_BFP4, scale=1.0)
    with expect_error(RuntimeError, "kv_window"):
        ttnn.bringup.sparse_sdpa(tq, tw, ti, K_DIM, kv_format=ttnn.bringup.SparseKVFormat.BF16, scale=1.0, kv_window=tw)


# ---------------------------------------------------------------------------------------- the bfloat8_b window
def _rec8(win):
    """bf16 window rows -> (their values on the bfloat8_b grid as bf16 (exact), the 576 B records)."""
    rows = ref.encode_rows(win, mbits=7)
    return ref.decode_rows(rows).to(torch.bfloat16), rows


@pytest.mark.parametrize("fp32", [False, True], ids=["dest16", "dest32"])
def test_bfp8_window_equals_bf16_window(device, fp32):
    H, S = 64, 2 * len(WIN_COUNTS) * 3
    nw = lambda s: [WIN, *WIN_COUNTS][s % (len(WIN_COUNTS) + 1)]
    nt = lambda s: [0, 37, 512][(s // (len(WIN_COUNTS) + 1)) % 3]
    q, win, rows, vals, idx, sink = _inputs(H, S, 256, 4096, nw, nt, seed=3)
    win8, rec8 = _rec8(win)
    ckc = _ckc(fp32)
    exp = ttnn.to_torch(_bfp4(device, q, win8, rows, idx, sink, ckc)())
    tq, ti = _dev(q, device, ttnn.bfloat16), _dev(idx.to(torch.int32), device, ttnn.uint32)
    ts = _dev(sink.view(1, 1, 1, -1), device, ttnn.bfloat16)
    tr = _dev(rows.view(1, 1, -1, rows.shape[-1]), device, ttnn.uint8)
    tw8 = _dev(rec8.view(1, 1, -1, 576), device, ttnn.uint8)

    def run(tq=tq, tr=tr, ti=ti, ts=ts, tw=tw8):
        return ttnn.bringup.sparse_sdpa(
            tq,
            tr,
            ti,
            K_DIM,
            kv_format=ttnn.bringup.SparseKVFormat.ROW_BFP4,
            scale=1.0,
            k_chunk_size=KC,
            compute_kernel_config=ckc,
            attention_sink=ts,
            kv_window=tw,
        )

    out = run()
    got = ttnn.to_torch(out)
    diff = (got.float() != exp.float()).any(-1)[0]
    assert not diff.any(), f"{int(diff.sum())} head-rows differ; rel {_rel(got, exp):.4f}"
    print(
        f"bfp8 window {'dest32' if fp32 else 'dest16'}: rel vs float32 {_rel(got, _golden(q, win8, vals, idx, sink)):.4f}"
    )
    q2, win2, rows2, _, idx2, sink2 = _inputs(H, S, 256, 4096, nw, nt, seed=4)
    b = dict(
        tq=_dev(q2, device, ttnn.bfloat16),
        tr=_dev(rows2.view(1, 1, -1, rows2.shape[-1]), device, ttnn.uint8),
        ti=_dev(idx2.to(torch.int32), device, ttnn.uint32),
        ts=_dev(sink2.view(1, 1, 1, -1), device, ttnn.bfloat16),
        tw=_dev(_rec8(win2)[1].view(1, 1, -1, 576), device, ttnn.uint8),
    )
    determinism.assert_deterministic(run, lambda: run(**b), first=out, label="bfp8 window")


@pytest.mark.parametrize("H", [32, 64])
def test_row_bfp8_window_only_equals_bf16(device, H):
    S, Tw = 3 * len(WIN_COUNTS), 2176
    g = torch.Generator().manual_seed(5)
    q = (torch.randn(1, H, S, K_DIM, generator=g) * K_DIM**-0.5).to(torch.bfloat16)
    win8, rec8 = _rec8(torch.randn(Tw, K_DIM, generator=g).to(torch.bfloat16))
    idx = torch.full((1, 1, S, WIN), MASKED, dtype=torch.int64)
    for s in range(S):
        n = [WIN, *WIN_COUNTS][s % (len(WIN_COUNTS) + 1)]
        idx[0, 0, s, :n] = torch.randperm(Tw, generator=g)[:n]
    sink = (torch.randn(H, generator=g) * 0.5).to(torch.bfloat16)
    ti, ts = _dev(idx.to(torch.int32), device, ttnn.uint32), _dev(sink.view(1, 1, 1, -1), device, ttnn.bfloat16)
    tq = _dev(q, device, ttnn.bfloat16)
    common = dict(scale=1.0, k_chunk_size=KC, compute_kernel_config=_ckc(), attention_sink=ts)
    exp = ttnn.bringup.sparse_sdpa(
        tq,
        _dev(win8.view(1, 1, Tw, K_DIM), device, ttnn.bfloat16),
        ti,
        K_DIM,
        kv_format=ttnn.bringup.SparseKVFormat.BF16,
        **common,
    )
    tk = _dev(rec8.view(1, 1, Tw, 576), device, ttnn.uint8)

    def run(kv=tk):
        return ttnn.bringup.sparse_sdpa(tq, kv, ti, K_DIM, kv_format=ttnn.bringup.SparseKVFormat.ROW_BFP8, **common)

    out = run()
    got = ttnn.to_torch(out)
    exp = ttnn.to_torch(exp)
    diff = (got.float() != exp.float()).any(-1)[0]
    assert not diff.any(), f"{int(diff.sum())} head-rows differ; rel {_rel(got, exp):.4f}"
    other = _dev(
        _rec8(torch.randn(Tw, K_DIM, generator=g).to(torch.bfloat16))[1].view(1, 1, Tw, 576), device, ttnn.uint8
    )
    determinism.assert_deterministic(run, lambda: run(other), first=out, label="row_bfp8")


def test_bfp8_refusals(device, expect_error):
    q, win, rows, vals, idx, sink = _inputs(32, 2, 256, 1024, lambda s: WIN, lambda s: 8)
    tq, ti = _dev(q, device, ttnn.bfloat16), _dev(idx.to(torch.int32), device, ttnn.uint32)
    tr = _dev(rows.view(1, 1, -1, rows.shape[-1]), device, ttnn.uint8)
    rec8 = _rec8(win)[1]
    bad = _dev(rec8[:, :320].contiguous().view(1, 1, -1, 320), device, ttnn.uint8)
    with expect_error(RuntimeError, "kv_window"):
        ttnn.bringup.sparse_sdpa(
            tq, tr, ti, K_DIM, kv_format=ttnn.bringup.SparseKVFormat.ROW_BFP4, scale=1.0, kv_window=bad
        )
    with expect_error(RuntimeError, "TOPK == k_chunk_size"):  # a top-k part with ROW_BFP8
        ttnn.bringup.sparse_sdpa(
            tq,
            _dev(rec8.view(1, 1, -1, 576), device, ttnn.uint8),
            ti,
            K_DIM,
            kv_format=ttnn.bringup.SparseKVFormat.ROW_BFP8,
            scale=1.0,
        )


# ------------------------------------------------------------------------------- the bfloat8_b window as blocks
def _blocks(rec8):
    """576 B records [Tw, 576] (Tw % 32 == 0) -> window blocks [Tw / 32, BLOCK_BYTES] (gaps zero)."""
    e, d = rec8[:, :32].reshape(-1, 32 * 32), rec8[:, 64:].reshape(-1, 32 * 512)
    out = torch.zeros(e.shape[0], ref.BLOCK_BYTES, dtype=torch.uint8)
    out[:, : e.shape[1]] = e
    out[:, ref.BLOCK_EXP2 : ref.BLOCK_EXP2 + e.shape[1]] = e
    out[:, ref.BLOCK_DATA :] = d
    return out


def _consecutive(idx, S, Tw, n_win, g):
    """Window ids of each query: n_win(s) consecutive rows from a random start (odd and even, across blocks)."""
    for s in range(S):
        w = n_win(s)
        idx[0, 0, s, :WIN] = MASKED
        idx[0, 0, s, :w] = int(torch.randint(0, Tw - w + 1, (1,), generator=g)) + torch.arange(w)
    return idx


@pytest.mark.parametrize("fp32", [False, True], ids=["dest16", "dest32"])
def test_bfp8_window_blocks_equal_records(device, fp32):
    """kv_window as blocks (32 rows a page) equals the 576 B records holding the same values, bit for bit, with full
    and partial windows of consecutive rows and top-k parts of 0 / 37 / 512."""
    H, S, Tw = 64, 2 * len(WIN_COUNTS) * 3, 256
    nw = lambda s: [WIN, *WIN_COUNTS][s % (len(WIN_COUNTS) + 1)]
    nt = lambda s: [0, 37, 512][(s // (len(WIN_COUNTS) + 1)) % 3]
    q, win, rows, vals, idx, sink = _inputs(H, S, Tw, 4096, nw, nt, seed=6)
    idx = _consecutive(idx, S, Tw, nw, torch.Generator().manual_seed(7))
    rec8 = _rec8(win)[1]
    tq, ti = _dev(q, device, ttnn.bfloat16), _dev(idx.to(torch.int32), device, ttnn.uint32)
    ts = _dev(sink.view(1, 1, 1, -1), device, ttnn.bfloat16)
    tr = _dev(rows.view(1, 1, -1, rows.shape[-1]), device, ttnn.uint8)
    common = dict(scale=1.0, k_chunk_size=KC, compute_kernel_config=_ckc(fp32), attention_sink=ts)
    F = ttnn.bringup.SparseKVFormat.ROW_BFP4
    exp = ttnn.to_torch(
        ttnn.bringup.sparse_sdpa(
            tq, tr, ti, K_DIM, kv_format=F, kv_window=_dev(rec8.view(1, 1, Tw, 576), device, ttnn.uint8), **common
        )
    )
    tb = _dev(_blocks(rec8).view(1, 1, Tw // 32, -1), device, ttnn.uint8)

    def run(tw=tb):
        return ttnn.bringup.sparse_sdpa(tq, tr, ti, K_DIM, kv_format=F, kv_window=tw, **common)

    out = run()
    got = ttnn.to_torch(out)
    diff = (got.float() != exp.float()).any(-1)[0]
    assert not diff.any(), f"{int(diff.sum())} head-rows differ; rel {_rel(got, exp):.4f}"
    other = _blocks(_rec8(torch.randn(Tw, K_DIM, generator=torch.Generator().manual_seed(8)).bfloat16())[1])
    determinism.assert_deterministic(
        run, lambda: run(_dev(other.view(1, 1, Tw // 32, -1), device, ttnn.uint8)), first=out, label="blocks"
    )


@pytest.mark.parametrize("H", [32, 64])
def test_row_bfp8_window_only_blocks_equal_records(device, H):
    """ROW_BFP8 with kv as blocks equals kv as records, consecutive windows (full and partial)."""
    S, Tw = 3 * len(WIN_COUNTS), 2176
    g = torch.Generator().manual_seed(9)
    q = (torch.randn(1, H, S, K_DIM, generator=g) * K_DIM**-0.5).to(torch.bfloat16)
    rec8 = _rec8(torch.randn(Tw, K_DIM, generator=g).to(torch.bfloat16))[1]
    nw = lambda s: [WIN, *WIN_COUNTS][s % (len(WIN_COUNTS) + 1)]
    idx = _consecutive(torch.full((1, 1, S, WIN), MASKED, dtype=torch.int64), S, Tw, nw, g)
    sink = (torch.randn(H, generator=g) * 0.5).to(torch.bfloat16)
    ti, ts = _dev(idx.to(torch.int32), device, ttnn.uint32), _dev(sink.view(1, 1, 1, -1), device, ttnn.bfloat16)
    tq = _dev(q, device, ttnn.bfloat16)
    common = dict(scale=1.0, k_chunk_size=KC, compute_kernel_config=_ckc(), attention_sink=ts)
    F = ttnn.bringup.SparseKVFormat.ROW_BFP8
    exp = ttnn.to_torch(
        ttnn.bringup.sparse_sdpa(
            tq, _dev(rec8.view(1, 1, Tw, 576), device, ttnn.uint8), ti, K_DIM, kv_format=F, **common
        )
    )
    tb = _dev(_blocks(rec8).view(1, 1, Tw // 32, -1), device, ttnn.uint8)

    def run(kv=tb):
        return ttnn.bringup.sparse_sdpa(tq, kv, ti, K_DIM, kv_format=F, **common)

    out = run()
    got = ttnn.to_torch(out)
    diff = (got.float() != exp.float()).any(-1)[0]
    assert not diff.any(), f"{int(diff.sum())} head-rows differ; rel {_rel(got, exp):.4f}"
    other = _blocks(_rec8(torch.randn(Tw, K_DIM, generator=g).bfloat16())[1])
    determinism.assert_deterministic(
        run, lambda: run(_dev(other.view(1, 1, Tw // 32, -1), device, ttnn.uint8)), first=out, label="row_bfp8 blocks"
    )
