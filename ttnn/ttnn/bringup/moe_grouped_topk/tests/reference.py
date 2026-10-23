# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.moe_grouped_topk in torch, single group (n_groups 1, topk_groups 1), from the op's docstring
(moe_grouped_topk_nanobind.cpp) and the source op's golden (tests/ttnn/nightly/.../deepseek_prefill/
test_moe_grouped_topk.py): scores = act(logits) (sigmoid, or sqrtsoftplus = sqrt(softplus(x))); keys = scores + bias
(the selection only); the top-k experts by key, descending (stable: the lower expert id first on a tie); weights = the
UNBIASED scores of the selected experts / (their sum + epsilon) x route_scale."""

from __future__ import annotations

import torch


def scores(logits, score_func):
    if score_func == "sqrtsoftplus":
        return torch.nn.functional.softplus(logits).sqrt()
    assert score_func == "sigmoid", score_func
    return torch.sigmoid(logits)


def keys(logits, bias, score_func):
    """The selection keys in the inputs' dtype."""
    return scores(logits, score_func) + bias


def select(k, top_k):
    """[rows, top_k] expert ids by key, descending, ties to the lower id (a stable sort of -key)."""
    return torch.sort(-k, dim=-1, stable=True).indices[..., :top_k]


def weights(logits, idx, score_func, route_scale, epsilon):
    """The normalized weights of the experts ``idx`` (a fixed selection), in the logits' dtype."""
    w = scores(logits, score_func).gather(-1, idx)
    return w / (w.sum(-1, keepdim=True) + epsilon) * route_scale


def gap(k, top_k):
    """Per row: the key of the k-th selected expert minus the best unselected one (a row whose gap is within the
    keys' error can legitimately select another expert)."""
    s = torch.sort(k, dim=-1, descending=True).values
    return s[..., top_k - 1] - s[..., top_k]
