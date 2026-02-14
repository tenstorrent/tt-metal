# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test for e02_ttnn_basics - Learn ttnn ops and pytest workflow.

Run: ./run.sh "e02 and solution"
"""

from pathlib import Path

import pytest
import torch
import ttnn

from ttnn.tutorials.onboarding.common.utils import load_module, pcc

LESSON_DIR = Path(__file__).parent


@pytest.mark.parametrize("module_name", ["exercise", "solution"])
def test_e02_ttnn_basics(module_name):
    """Test matmul_add implementation against PyTorch reference."""
    module = load_module(module_name, LESSON_DIR)
    reference = load_module("reference", LESSON_DIR)

    device = ttnn.open_device(device_id=0)

    a = torch.rand(32, 32, dtype=torch.float32)
    b = torch.rand(32, 32, dtype=torch.float32)
    c = torch.rand(32, 32, dtype=torch.float32)

    result = module.matmul_add(device, a, b, c)
    expected = reference.matmul_add(a, b, c)

    ttnn.close_device(device)

    result_pcc = pcc(result, expected)
    assert result_pcc >= 0.99, f"PCC {result_pcc:.4f} < 0.99"
