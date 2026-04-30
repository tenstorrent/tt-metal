#!/usr/bin/env python3
"""Reference implementation of lookup-table TQ attention.

Codebook is per-element scalar (8 centroids, each a single float). Each
element of K is independently quantized:
  K[h, p, d] = centroids[K_idx[h, p, d]] * K_norm[h, p]   (norm is per-(h, p))

Standard path computes Q·K via:
  K_full = dequantize(K_idx, K_norm)        # [NQH, P, D] BF16
  scores = Q @ K_full.T                     # [NQH, P]

Lookup path:
  T[h, d, c] = Q[h, d] * centroids[c]       # precompute once: [NQH, D, 8]
  scores[h, p] = K_norm[h, p] * sum_d T[h, d, K_idx[h, p, d]]

The lookup path replaces dequant (cascade per element) + matmul (FPU) with
a per-element gather (one of 8 precomputed scalars) + reduction. Should be
mathematically equivalent up to floating-point precision.
"""
import sys

import torch

sys.path.insert(0, "/localdev/mtairum/tt-metal")
from turbo_quant.codebook import get_codebook


def standard_path(Q, K_idx, K_norm, centroids, scale):
    """Dequant K, then Q·K matmul. Returns scores [NQH, P]."""
    # Dequant: K[h, p, d] = centroids[K_idx[h, p, d]] * K_norm[h, p]
    K_centroid = centroids[K_idx]  # [NKH, P, D]
    K_full = K_centroid * K_norm.unsqueeze(-1)  # [NKH, P, D]
    # GQA: each Q head reads from one KV head. With NKH=1 and NQH=4, all Q heads share K.
    nqh = Q.shape[0]
    nkh = K_idx.shape[0]
    q_per_kv = nqh // nkh
    K_per_q = K_full.repeat_interleave(q_per_kv, dim=0)  # [NQH, P, D]
    # Q @ K.T: scores[h, p] = sum_d Q[h, d] * K_per_q[h, p, d]
    scores = torch.einsum("hd,hpd->hp", Q, K_per_q) * scale
    return scores


def lookup_path(Q, K_idx, K_norm, centroids, scale):
    """Precompute Q·centroid lookup table; per-position gather + multiply."""
    # T[h, d, c] = Q[h, d] * centroids[c]
    # Q: [NQH, D], centroids: [num_centroids]
    T = Q.unsqueeze(-1) * centroids.unsqueeze(0).unsqueeze(0)  # [NQH, D, num_centroids]

    nqh = Q.shape[0]
    nkh = K_idx.shape[0]
    q_per_kv = nqh // nkh
    K_idx_per_q = K_idx.repeat_interleave(q_per_kv, dim=0)  # [NQH, P, D]
    K_norm_per_q = K_norm.repeat_interleave(q_per_kv, dim=0)  # [NQH, P]

    # For each (h, p, d): pick T[h, d, K_idx_per_q[h, p, d]]
    # Then sum over d, multiply by K_norm.
    # Use gather:
    nqh, P, D = K_idx_per_q.shape
    T_flat = T  # [NQH, D, num_centroids]
    # Gather along last dim: indices shape [NQH, D, P], values shape [NQH, D, num_centroids]
    indices = K_idx_per_q.permute(0, 2, 1)  # [NQH, D, P]
    gathered = torch.gather(T_flat, 2, indices)  # [NQH, D, P]
    summed = gathered.sum(dim=1)  # [NQH, P]
    scores = summed * K_norm_per_q * scale
    return scores


def main():
    head_dim = 128
    num_pos = 32
    nkh = 1
    nqh = 4
    bits = 3
    num_centroids = 2**bits
    scale = head_dim**-0.5

    torch.manual_seed(0)
    codebook = get_codebook(head_dim, bits, device="cpu", dtype=torch.float32)
    centroids = codebook.centroids  # [8] float

    Q = torch.randn(nqh, head_dim)
    K_idx = torch.randint(0, num_centroids, (nkh, num_pos, head_dim))
    K_norm = torch.rand(nkh, num_pos) * 0.5 + 0.5  # nonzero positive

    print(f"Codebook (8 centroids): {centroids.tolist()}")
    print(f"Q shape: {Q.shape}")
    print(f"K_idx shape: {K_idx.shape}")
    print(f"K_norm shape: {K_norm.shape}")

    s_standard = standard_path(Q, K_idx, K_norm, centroids, scale)
    s_lookup = lookup_path(Q, K_idx, K_norm, centroids, scale)

    diff = (s_standard - s_lookup).abs()
    cos = torch.nn.functional.cosine_similarity(s_standard.flatten().unsqueeze(0), s_lookup.flatten().unsqueeze(0))

    print(f"\nStandard path scores (head 0, first 5 positions): {s_standard[0, :5].tolist()}")
    print(f"Lookup path scores   (head 0, first 5 positions): {s_lookup[0, :5].tolist()}")
    print(f"\nMax abs diff: {diff.max().item():.2e}")
    print(f"Mean abs diff: {diff.mean().item():.2e}")
    print(f"Cosine similarity: {cos.item():.6f}")

    if diff.max() < 1e-4:
        print("\n✓ MATH VALIDATED — lookup-table path matches standard within float precision")
        return 0
    else:
        print(f"\n✗ DIVERGENCE — lookup path doesn't match standard")
        return 1


if __name__ == "__main__":
    sys.exit(main())
