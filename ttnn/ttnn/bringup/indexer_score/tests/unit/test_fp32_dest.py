# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup indexer_score with fp32 DEST (opt-in through compute_kernel_config.fp32_dest_acc_en).

The source op rejects fp32 DEST. The fork accepts it for DSA scoring (one head-summed plane, learned gates): q.k,
the gate MAC over heads and the accumulator run in fp32, and the output stays bf16. It also reconfigures srcA to the
mask CB's format before the causal mask (under fp32 DEST the mul phase leaves srcA in Float32, and the bf16 -inf mask
tiles were unpacked as fp32: rows 16..31 of full tiles unmasked, diagonal tiles wrong).

Checked against a float32 torch reference, at the Hy4 indexer head geometry (32 heads x 128):
- every future column is -inf and every causal column finite (the mask), on the per-column path (k_chunk 32, where
  the gate multiply honours the fidelity) and on the blocked-mul path (k_chunk 64 / 128);
- per-column path accuracy: rel L2 error 0.0019 vs the bf16-DEST op's 0.0118 on the same inputs (the blocked path's
  multiply is LoFi-like whatever the fidelity, so fp32 DEST does not improve it: 0.029 vs 0.023);
- the refusals: fp32 DEST with MSA (constant gate, or several groups) is rejected.
"""

import pytest
import torch

import ttnn

pytestmark = pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="indexer_score is Blackhole-only")

HEADS, DIM = 32, 128


def _inputs(sq, t, seed=7):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, HEADS, sq, DIM, generator=g).to(torch.bfloat16)
    k = torch.randn(1, 1, t, DIM, generator=g).to(torch.bfloat16)
    w = (torch.randn(1, 1, sq, HEADS, generator=g) / 64).to(torch.bfloat16)  # some gates negative
    return q, k, w


def _ref(q, k, w, chunk_start):
    sq, t = q.shape[2], k.shape[2]
    score = torch.zeros(sq, t)
    for h in range(HEADS):
        score += torch.relu(q[0, h].float() @ k[0, 0].float().T) * w[0, 0, :, h : h + 1].float()
    future = torch.arange(t)[None, :] > chunk_start + torch.arange(sq)[:, None]
    return score.masked_fill(future, float("-inf")), future


def _run(device, q, k, w, chunk_start, k_chunk, fp32):
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32,
        packer_l1_acc=False,
    )
    dev = lambda x: ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)  # noqa: E731
    out = ttnn.bringup.indexer_score_dsa(
        dev(q),
        dev(k),
        dev(w),
        chunk_start_idx=chunk_start,
        program_config=ttnn.bringup.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=k_chunk, head_group_size=0),
        compute_kernel_config=ckc,
    )
    return ttnn.to_torch(out).float().reshape(q.shape[2], k.shape[2])


def _rel(out, ref, future):
    c = ~future
    return ((out[c] - ref[c]).norm() / ref[c].norm()).item()


@pytest.mark.parametrize("k_chunk", [32, 64, 128], ids=["percol_k32", "blocked_k64", "blocked_k128"])
@pytest.mark.parametrize("sq, t", [(64, 512), (128, 1024)], ids=["sq64_t512", "sq128_t1024"])
def test_fp32_dest_mask_and_accuracy(device, k_chunk, sq, t):
    chunk_start = t - sq  # the chunk's queries see the whole prefix; the last rows sit on the diagonal
    q, k, w = _inputs(sq, t)
    ref, future = _ref(q, k, w, chunk_start)
    out = _run(device, q, k, w, chunk_start, k_chunk, fp32=True)
    assert torch.isneginf(out[future]).all(), f"{(~torch.isneginf(out[future])).sum().item()} future columns not -inf"
    assert torch.isfinite(
        out[~future]
    ).all(), f"{(~torch.isfinite(out[~future])).sum().item()} causal columns not finite"
    rel = _rel(out, ref, future)
    base = _rel(_run(device, q, k, w, chunk_start, k_chunk, fp32=False), ref, future)
    print(f"k_chunk {k_chunk} sq {sq} t {t}: rel L2 fp32 DEST {rel:.5f}, bf16 DEST {base:.5f}")
    if k_chunk == 32:
        # Per-column path: fp32 MAC, HiFi4 gate multiply. Measured 0.0019 (about the bf16 output rounding); the
        # bf16-DEST op scores 0.0118 on the same inputs.
        assert rel < 0.004, rel
        assert rel < 0.5 * base, (rel, base)
    else:
        # Blocked path (k_chunk > 32): the custom bcast-col multiply has no fidelity phases (LoFi-like), so fp32 DEST
        # does not make it more accurate: measured 0.0287 (bf16 DEST 0.0229). Only the mask is checked strictly.
        assert rel < 0.04, rel


def test_fp32_dest_rejects_msa(device, expect_error):
    sq, t = 64, 256
    g = torch.Generator().manual_seed(3)
    q = torch.randn(1, 8, sq, DIM, generator=g).to(torch.bfloat16)
    k = torch.randn(1, 1, t, DIM, generator=g).to(torch.bfloat16)
    ckc = ttnn.init_device_compute_kernel_config(device.arch(), fp32_dest_acc_en=True)
    dev = lambda x: ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)  # noqa: E731
    cfg = ttnn.bringup.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=32, head_group_size=0)
    for groups in (1, 2):
        with expect_error(RuntimeError, "supported only for DSA scoring"):
            ttnn.bringup.indexer_score_msa(
                dev(q),
                dev(k),
                num_groups=groups,
                chunk_start_idx=t - sq,
                program_config=cfg,
                compute_kernel_config=ckc,
            )
