# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch
import ttnn

from ttnn.tutorials.onboarding.common.utils import load_module, pcc

LESSON_DIR = Path(__file__).parent


@pytest.mark.parametrize("module_name", ["exercise", "solution"])
def test_e05(module_name):
    """Test sign operation."""
    module = load_module(module_name, LESSON_DIR)
    reference = load_module("reference", LESSON_DIR)

    device = ttnn.open_device(device_id=0)

    input_tensor = torch.randn(64, 64, dtype=torch.float32)
    result = module.sign(device, input_tensor)
    expected = reference.sign(input_tensor)

    ttnn.close_device(device)

    assert pcc(result, expected) >= 0.99
