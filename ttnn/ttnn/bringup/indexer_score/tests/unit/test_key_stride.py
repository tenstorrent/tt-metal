# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.indexer_score_dsa key_stride (pooled keys), single device.

key_stride R: the K sequence is 1/R of the query token sequence; key j pools tokens [R*j, R*j + R) and is visible to
query token p iff R*j + R - 1 <= p. T and kv_len are in keys, chunk_start_idx and q rows in tokens. Checked against a
float32 torch reference (tests/reference.py dsa_score) on the same bf16 inputs, contiguous K:
- every pool-future column is -inf and every visible column finite, exactly (rows with no visible key are all -inf);
- visible columns: relative L2 error (fp32 DEST, HiFi4, k_chunk 32 = the per-column gate path, measured ~0.002);
- several chunk starts: key-tile-aligned diagonals, every staircase pattern (start/32 % R), rows that see nothing
  (start 0), and a runtime kv_len (columns >= kv_len are don't-care);
- refusals: R not in {1, 2, 4, 8}, and a chunk start past T in keys.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

import ttnn

pytestmark = pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="indexer_score is Blackhole-only")

_spec = importlib.util.spec_from_file_location(
    "bringup_indexer_score_tests_reference_ks", Path(__file__).resolve().parents[1] / "reference.py"
)
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)

HEADS, DIM = 32, 128


def _inputs(sq, t, seed):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, HEADS, sq, DIM, generator=g).to(torch.bfloat16)
    k = torch.randn(1, 1, t, DIM, generator=g).to(torch.bfloat16)
    w = (torch.randn(1, 1, sq, HEADS, generator=g) / 64).to(torch.bfloat16)  # some gates negative
    return q, k, w


def _run(device, q, k, w, chunk_start, key_stride, k_chunk=32, kv_len=None, fidelity=ttnn.MathFidelity.HiFi4):
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    dev = lambda x: ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)  # noqa: E731
    out = ttnn.bringup.indexer_score_dsa(
        dev(q),
        dev(k),
        dev(w),
        chunk_start_idx=chunk_start,
        program_config=ttnn.bringup.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=k_chunk, head_group_size=0),
        compute_kernel_config=ckc,
        kv_len=kv_len,
        key_stride=key_stride,
    )
    return ttnn.to_torch(out).float().reshape(q.shape[2], k.shape[2])


def _check(out, want, cols, label):
    out, want = out[:, :cols], want[:, :cols]
    future = torch.isneginf(want)
    n_mask = int((~torch.isneginf(out[future])).sum())
    n_nonfinite = int((~torch.isfinite(out[~future])).sum())
    assert n_mask == 0, f"{label}: {n_mask} pool-future columns not -inf"
    assert n_nonfinite == 0, f"{label}: {n_nonfinite} visible columns not finite"
    a, b = out[~future], want[~future]
    rel = float((a - b).norm() / b.norm()) if b.numel() else 0.0
    print(f"{label}: {int((~future).sum())} visible scores, {int(future.sum())} masked, rel L2 {rel:.5f}")
    assert rel < 0.004, f"{label}: rel {rel}"


# Sq 256 tokens (8 q tiles), T = 2048 tokens of keys (512 keys at R=4, 1024 at R=2). Starts in tokens: 0 (rows 0..R-2 see no key), 32 / 96 (every
# staircase pattern at R=4 appears down the 8 q tiles of each), 640, and 1792 (the last q row sees the last key).
@pytest.mark.parametrize("key_stride", [4, 2], ids=["r4", "r2"])
@pytest.mark.parametrize("chunk_start", [0, 32, 96, 640, 1792])
def test_key_stride_contiguous(device, key_stride, chunk_start):
    sq, t = 256, 2048 // key_stride
    q, k, w = _inputs(sq, t, seed=11 + chunk_start)
    want = ref.dsa_score(q[0], k[0, 0], w[0, 0], chunk_start, key_stride=key_stride)
    out = _run(device, q, k, w, chunk_start, key_stride)
    if chunk_start == 0:
        assert torch.isneginf(out[: key_stride - 1]).all(), "rows that see no key must be all -inf"
    _check(out, want, t, f"R{key_stride} start {chunk_start}")


@pytest.mark.parametrize("k_chunk", [32, 64], ids=["k32", "k64"])
def test_key_stride_kv_len(device, k_chunk):
    # Growing-cache prefill: T allocated at 1024 keys, kv_len 384 valid; the chunk's 256 tokens start at token 1280
    # (key 320), so its last rows' windows end at key 383. Columns >= kv_len are don't-care.
    sq, t, r, start, kv_len = 256, 1024, 4, 1280, 384
    q, k, w = _inputs(sq, t, seed=5)
    want = ref.dsa_score(q[0], k[0, 0], w[0, 0], start, key_stride=r)
    out = _run(device, q, k, w, start, r, k_chunk=k_chunk, kv_len=kv_len)
    if k_chunk == 32:
        _check(out, want, kv_len, f"kv_len {kv_len} k_chunk {k_chunk}")
    else:
        # Blocked bcast-col multiply (k_chunk > 32) is LoFi-like whatever the fidelity (CHANGELOG): mask exact,
        # accuracy loose.
        future = torch.isneginf(want[:, :kv_len])
        assert torch.isneginf(out[:, :kv_len][future]).all()
        assert torch.isfinite(out[:, :kv_len][~future]).all()
        a, b = out[:, :kv_len][~future], want[:, :kv_len][~future]
        assert float((a - b).norm() / b.norm()) < 0.04


def test_key_stride_one_matches_default(device):
    # key_stride=1 is the source op: same program and the plain causal mask.
    sq, t, start = 128, 512, 256
    q, k, w = _inputs(sq, t, seed=3)
    a = _run(device, q, k, w, start, 1)
    want = ref.dsa_score(q[0], k[0, 0], w[0, 0], start)
    _check(a, want, t, "R1")


def test_key_stride_refusals(device, expect_error):
    sq, t = 64, 256
    q, k, w = _inputs(sq, t, seed=1)
    with expect_error(RuntimeError, "key_stride must be 1, 2, 4 or 8"):
        _run(device, q, k, w, 0, 3)
    with expect_error(RuntimeError, "starts at or past T"):
        _run(device, q, k, w, 4 * t, 4)  # token 1024 = key 256 = T
