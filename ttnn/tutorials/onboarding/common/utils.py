# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for onboarding tests."""

from pathlib import Path
import importlib.util

import torch


def load_module(name: str, lesson_dir: Path):
    """Load a module from a lesson directory."""
    spec = importlib.util.spec_from_file_location(name, lesson_dir / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pcc(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Compute Pearson Correlation Coefficient between two tensors."""
    actual_flat = actual.flatten().float()
    expected_flat = expected.flatten().float()
    return torch.corrcoef(torch.stack([actual_flat, expected_flat]))[0, 1].item()
