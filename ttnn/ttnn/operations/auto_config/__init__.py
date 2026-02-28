# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Auto-optimal configuration selection infrastructure for TTNN operations.

This module provides automatic, hardware-aware configuration selection for
TTNN operations, starting with matmul. It generates, validates, scores, and
caches the most performant configurations for any given input signature.

Usage:
    import ttnn
    from ttnn.operations.auto_config import matmul_auto

    output = matmul_auto(input_a, input_b)
"""

from ttnn.operations.auto_config.matmul_auto import matmul_auto
from ttnn.operations.auto_config.base import AutoConfigSelector
from ttnn.operations.auto_config.config_cache import ConfigCache

__all__ = [
    "matmul_auto",
    "AutoConfigSelector",
    "ConfigCache",
]
