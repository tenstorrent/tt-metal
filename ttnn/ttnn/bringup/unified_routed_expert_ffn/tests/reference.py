# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""What ttnn.bringup.unified_routed_expert_moe computes on one device, in torch float32.

For each local slot e: g = global_expert_idx_table[e], n = expert_token_counts[g] (0 when outside
[min_active_tokens, max_active_tokens], capped at max_dispatched_tokens_per_expert), o = expert_region_offsets[g];
rows x[o : o + n] go through the SwiGLU FFN of expert e and land at out[o : o + n]:
    out = act(x @ gate_e, x @ up_e) @ down_e,   Silu: act(g, u) = silu(g) * u
Every other row of the output is not written (the op allocates it uninitialized): don't-care. high_precision only
changes the device precision (fidelity, bf16 intermediates, Float32 partials), not the semantics.
"""

from __future__ import annotations

# torch is imported inside each function: ttnn/ allows no module-level torch import (check-torch-imports-in-ttnn)


def activation(name: str, g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    import torch

    if name == "Silu":
        return torch.nn.functional.silu(g) * u
    raise NotImplementedError(f"no reference for activation {name}")


def routed_expert_moe(
    x: torch.Tensor,
    offsets: torch.Tensor,
    counts: torch.Tensor,
    idx_table: torch.Tensor,
    gates,
    ups,
    downs,
    max_tokens: int,
    act: str = "Silu",
    min_active_tokens: int = 0,
    max_active_tokens: int = 2**32 - 1,
):
    """x [tokens, emb]; offsets, counts [n_experts]; idx_table [experts_per_chip]; gates/ups [e] -> [emb, hidden],
    downs [e] -> [hidden, emb] (callables or tensors). -> (out [tokens, emb] float32, valid [tokens] bool)."""
    import torch

    out = torch.zeros(x.shape, dtype=torch.float32)
    valid = torch.zeros(x.shape[0], dtype=torch.bool)
    for e in range(len(idx_table)):
        g = int(idx_table[e])
        n = int(counts[g])
        n = 0 if n < min_active_tokens or n > max_active_tokens else min(n, max_tokens)
        if n == 0:
            continue
        o = int(offsets[g])
        w = [m(e) if callable(m) else m[e] for m in (gates, ups, downs)]
        xe = x[o : o + n].float()
        out[o : o + n] = activation(act, xe @ w[0].float(), xe @ w[1].float()) @ w[2].float()
        valid[o : o + n] = True
    return out, valid
