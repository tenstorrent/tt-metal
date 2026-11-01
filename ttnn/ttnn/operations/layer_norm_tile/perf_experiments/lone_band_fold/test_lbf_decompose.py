# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Error decomposition on large-mean rows (gamma = 1, beta = 0): per-row shift (mean error) vs the rest."""

import os

import pytest
import importlib

torch = importlib.import_module("torch")  # no global torch import under ttnn (pre-commit)
import ttnn

from ttnn.operations.layer_norm_tile import layer_norm
from tests.ttnn.unit_tests.operations.layer_norm_tile.test_layer_norm_tile import (
    DRAM,
    L1,
    compute_config,
    make_affine,
    reference_layer_norm,
    to_device,
)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("fp32_dest", [False, True], ids=["bf16dest", "fp32dest"])
def test_decompose(device, fp32_dest, seed):
    shape = (32, 1, 32, 768)
    torch.manual_seed(seed)
    x = (100.0 + torch.randn(shape)).to(torch.bfloat16)
    res = torch.randn(shape).to(torch.bfloat16)
    gamma, beta = torch.ones(768).to(torch.bfloat16), torch.zeros(768).to(torch.bfloat16)
    out = layer_norm(
        to_device(x, device, DRAM),
        make_affine(gamma, device, ttnn.ROW_MAJOR_LAYOUT, L1),
        make_affine(beta, device, ttnn.ROW_MAJOR_LAYOUT, L1),
        to_device(res, device, DRAM),
        compute_kernel_config=compute_config(fp32_dest),
    )
    exp = reference_layer_norm(x, gamma, beta, res).float().reshape(-1, 768)
    act = ttnn.to_torch(out).float().reshape(-1, 768)
    shift = (act - exp).mean(-1)
    scale = act.std(-1) / exp.std(-1)
    rest = (act - exp - shift[:, None]).pow(2).mean().sqrt().item()
    print(
        f"\nLBF_DECOMP fold={os.environ.get('LBF_FOLD')} dest={'fp32' if fp32_dest else 'bf16'} seed={seed} "
        f"total={(act - exp).pow(2).mean().sqrt().item():.5f} shift_rms={shift.pow(2).mean().sqrt().item():.5f} "
        f"scale_err_rms={(scale - 1).pow(2).mean().sqrt().item():.5f} rest_rms={rest:.5f}"
    )
