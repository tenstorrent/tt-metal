# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf 1 guard set (single chip, run under scripts/run_safe_pytest.sh --profile): one representative per distinct
kernel path x placement, REPS calls each; read with guard_report.py <ops_perf_results.csv>."""
import pytest
import ttnn

from ttnn.operations.mhc_post import mhc_post

N = 4
REPS = 4


@pytest.mark.parametrize("T,C", [(160, 5120), (1024, 1792)])
def test_plain_fp32(device, T, C):
    import torch

    torch.manual_seed(0)
    dev = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    f, x = dev(torch.randn(1, 1, T, C)), dev(torch.randn(1, 1, T, N * C))
    post, comb = dev(torch.rand(1, 1, T, N)), dev(torch.rand(1, 1, T, N * N))
    for _ in range(REPS):
        mhc_post(f, x, post, comb)
    ttnn.synchronize_device(device)


@pytest.mark.parametrize(
    "T,mode,norm,coef_source",
    [
        (160, "plain", "off", "packed"),
        (160, "stats", "off", "packed"),
        (160, "collapse", "off", "packed"),
        (160, "collapse", "on", "packed"),
        (1280, "plain", "off", "packed"),
        (1280, "stats", "off", "packed"),
        (1280, "collapse", "on", "packed"),
        (1280, "collapse", "on", "separate"),
    ],
)
def test_ds41(device, T, mode, norm, coef_source):
    from eval.golden_tests.mhc_post.feature_spec import _case
    from eval.golden_tests.mhc_post.helpers import call_mhc_post, make_case

    case = make_case(
        _case((1, 1, T, N * 5120)),
        mode=mode,
        coef_source=coef_source,
        dtype=ttnn.bfloat16,
        sublayer_dtype=ttnn.bfloat16,
        norm=norm,
    )
    for _ in range(REPS):
        call_mhc_post(
            case,
            mode=mode,
            coef_source=coef_source,
            dtype=ttnn.bfloat16,
            sublayer_dtype=ttnn.bfloat16,
            device=device,
            math_fidelity=ttnn.MathFidelity.HiFi2,
            op=mhc_post,
        )
    ttnn.synchronize_device(device)
