# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Softmax operation for TTNN.

Numerically-stable softmax along the last (W) or second-to-last (H) dimension.
"""

from .softmax import softmax, default_compute_kernel_config, validate, SUPPORTED, EXCLUSIONS, INPUT_TAGGERS

__all__ = ["softmax", "default_compute_kernel_config", "validate", "SUPPORTED", "EXCLUSIONS", "INPUT_TAGGERS"]
