# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""One case per distinct ttnn.bringup.moe_grouped_topk call a model makes (bringup-fork-tests skill). Append only;
never edit or loosen another model's case. Shapes are per device, exactly as captured (fork_calls.json)."""

CASES = [
    {
        # deepseek_v41_flash_d_p O.1 (scaled 2x2 run, ladder rung last; tt/router.py:TtRouter._fused, ROUTER_IMPL=fused):
        # the router after the gate matmul on the chip's 1024 rows: sqrtsoftplus scores of the 96 (scaled) experts, a
        # per-expert fp32 bias for the selection only (passed full-shape, [rows, E]), top 6, the unbiased scores
        # normalized x 1.5 / (sum + 1e-20); fp32 keys (fp32_selection), the fp32 softplus (fp32_activation, DS41-105).
        # Logits ~ 2 N(0, 1) - 2 (most negative, a top tail), bias 0.05 N(0, 1) per expert. Checks (test docstring):
        # the selected sets on every clear row (match), the weights within noise_limit (fp32 logits / bias, bf16 out).
        "id": "deepseek_v41_flash_d_p-2x2-s1024-e96-top6-sqrtsoftplus-fp32",
        "model": "deepseek_v41_flash_d_p",
        "task": "O.1",
        "sig": "0fad887387",
        "mesh": [2, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        "scores": {"shape": [1, 1, 1024, 96], "dtype": "FLOAT32", "layout": "TILE"},
        "bias": {"shape": [1, 1, 1024, 96], "dtype": "FLOAT32", "layout": "TILE"},
        "n_groups": 1,
        "summed_experts_per_group": 1,
        "topk_groups": 1,
        "n_activated_experts": 6,
        "route_scale": 1.5,
        "epsilon": 1e-20,
        "stable_sort": True,
        "score_func": "sqrtsoftplus",
        "fp32_selection": True,
        "fp32_activation": True,
        "logit_scale": 2.0,
        "logit_shift": -2.0,
        "bias_scale": 0.05,
        "seed": 0,
        "noise": {"keys": {"inputs": ["fp32", "fp32"], "out": "fp32"}, "weights": {"inputs": ["fp32"], "out": "bf16"}},
    },
]
