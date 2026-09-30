# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Cases of ttnn.bringup.mhc_pre_xing / mhc_pre_xing_pack (the Xing4.0 entry of the fork), one per distinct call
xing40_a4b_d_p makes (tt/mhc.py:TtHcWeights._fused, tt/collapse.py:TtHcCollapse._fused), at the per-chip shapes of
its 4x2 mesh (SP 4 x TP 2: T = S/4 rows, the chip's 1792 columns of each of the 4 streams). Append only.

mode: "pack" (mix row + streams -> row with sum x^2 in column 24), "coef" (reduced row -> hc), "collapse" (hc +
streams -> y), "both" (reduced row + streams -> hc and y). The coefficient inputs are built from a random full-width
projection (x @ fn^T, sum x^2 over n*H = 4 x 3584 columns), so the rsqrt scale and the logits are realistic.
Limits: measured rel L2 <= 3e-7 (hc parts, y, sum x^2) on seeds 0-3; a 1e-4 scale of any output fails them."""

_MESH = {"mesh": [4, 2], "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576}}
_XING = {"n": 4, "hidden": 3584, "norm_eps": 1e-6, "hc_eps": 1e-6, "sinkhorn_iters": 20, "clamp": [-30.0, 30.0]}
_LIMITS = {"pcc": 0.99999, "max_rel": {"pre": 1e-5, "post": 1e-5, "comb": 1e-5, "y": 1e-5, "ss": 1e-5}}

CASES = [
    # attn_hc / ffn_hc partial row on the ladder's last chunk (5120 tokens -> 1280 rows per chip)
    {"id": "xing40_a4b_d_p-4x2-pack-t1280-c1792", "mode": "pack", "T": 1280, "C": 1792, "seed": 0},
    # the coefficients after the axis-1 all_reduce: moderate logits, and one layer-like case whose comb logits
    # exceed the [-30, 30] clamp (a_res = 12) so the clamp and the row-max shift matter
    {"id": "xing40_a4b_d_p-4x2-coef-t1280", "mode": "coef", "T": 1280, "C": 1792, "seed": 1, "scale": [0.8, 0.5, 3.0]},
    {
        "id": "xing40_a4b_d_p-4x2-coef-t1280-clamp",
        "mode": "coef",
        "T": 1280,
        "C": 1792,
        "seed": 2,
        "scale": [0.8, 0.5, 12.0],
    },
    # component-test chunk (s4096 chunk 1: 2048 tokens -> 512 rows per chip)
    {"id": "xing40_a4b_d_p-4x2-coef-t512", "mode": "coef", "T": 512, "C": 1792, "seed": 3, "scale": [0.8, 0.5, 3.0]},
    # attn_collapse / ffn_collapse: y = sum_i pre_i x_i from a finished hc
    {
        "id": "xing40_a4b_d_p-4x2-collapse-t1280-c1792",
        "mode": "collapse",
        "T": 1280,
        "C": 1792,
        "seed": 4,
        "scale": [0.8, 0.5, 3.0],
    },
    # the one-program coefficients + collapse (not used by the model's default split steps; kept covered)
    {
        "id": "xing40_a4b_d_p-4x2-both-t512-c1792",
        "mode": "both",
        "T": 512,
        "C": 1792,
        "seed": 5,
        "scale": [0.8, 0.5, 3.0],
    },
]
for _c in CASES:
    _c.update({"model": "xing40_a4b_d_p", "task": "P.2b", **_MESH, **_XING, **_LIMITS})
