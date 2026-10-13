# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Torch semantics of ttnn.bringup.prepare_chunk_recurrence (the KDA chunk-preparation op), for the fork's model-case
tests. Taken from the source op's test oracle (tests/ttnn/nightly/unit_tests/operations/experimental/kda/
test_prepare_chunk_recurrence.py:_oracle) and the op's docstring, plus the arguments the oracle leaves out: gate_scale
(g is multiplied by it before the cumulative sum) and beta_logits_column_offset (beta = sigmoid of the logits at
columns offset + h of a token-major tensor). The fork's precise_gate_factors only changes how exactly the anchored gate
exponents are formed, not what they are, so the reference is the same either way.

Per head h and 32-token chunk n (rows t of the chunk), with q, k L2-normalized over K (eps 1e-6) and q scaled by K^-1/2:
    G = cumsum_t(gate_scale * g),  beta = sigmoid(logit)
    v_beta = beta v, kd = beta k exp(G), q_decay = q exp(G), k_dec_t = (k exp(G_last - G))^T, final_decay = exp(G_last)
    intra = tril(q exp(G) (k exp(-G))^T)                  (diagonal included)
    t_inv = (I + strict_tril(beta k exp(G) (k exp(-G))^T))^-1
The matmuls run in float64 with the exponents anchored at G_last / 2 (exp(G - G_last/2) exp(G_last/2 - G') =
exp(G - G')), as the oracle does: the masked-out triangle can overflow otherwise."""

from __future__ import annotations

import torch

CHUNK = 32
OUTPUT_NAMES = ("v_beta", "kd", "q_decay", "intra", "k_dec_t", "final_decay", "t_inv")


def _chunks(t: torch.Tensor, heads: int, dim: int) -> torch.Tensor:
    """[1, T, H * dim] -> [H, N, 32, dim] float64."""
    n = t.shape[1] // CHUNK
    return t.double().reshape(n * CHUNK, heads, dim).permute(1, 0, 2).reshape(heads, n, CHUNK, dim)


def prepare_chunk_recurrence(q, k, v, g, beta_src, *, num_heads, gate_scale=1.0, beta_logits_column_offset=None):
    """q, k, g [1, T, H * K], v [1, T, H * V]; beta_src: token-major logits [1, T, columns] (with
    beta_logits_column_offset) -> the seven outputs, float64, in OUTPUT_NAMES order."""
    H = num_heads
    K, V = q.shape[-1] // H, v.shape[-1] // H
    assert beta_logits_column_offset is not None, "only the beta-logits form is modelled"
    logits = beta_src[..., beta_logits_column_offset : beta_logits_column_offset + H]
    beta = _chunks(torch.sigmoid(logits.double()), H, 1)
    q, k, v = _chunks(q, H, K), _chunks(k, H, K), _chunks(v, H, V)
    g = _chunks(g, H, K) * gate_scale
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * K**-0.5
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    G = torch.cumsum(g, dim=2)
    G_last = G[:, :, -1:]
    anchor = G_last * 0.5
    up, down = torch.exp(G - anchor), torch.exp(anchor - G)

    v_beta = beta * v
    kd = beta * k * torch.exp(G)
    q_decay = q * torch.exp(G)
    k_dec_t = (k * torch.exp(G_last - G)).transpose(-1, -2)
    final_decay = torch.exp(G_last[:, :, 0]).unsqueeze(-1)  # [H, N, K, 1]
    akk = (beta * k * up) @ (k * down).transpose(-1, -2)
    intra = ((q * up) @ (k * down).transpose(-1, -2)).tril()
    eye = torch.eye(CHUNK, dtype=torch.float64)
    t_inv = torch.linalg.inv(eye + torch.tril(akk, diagonal=-1))
    return v_beta, kd, q_decay, intra, k_dec_t, final_decay, t_inv
