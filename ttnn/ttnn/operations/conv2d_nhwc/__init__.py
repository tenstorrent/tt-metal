# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""conv2d_nhwc — channels-last 2D convolution."""

from .conv2d_nhwc import (
    EXCLUSIONS,
    INPUT_TAGGERS,
    PROPERTIES,
    SUPPORTED,
    conv2d_nhwc,
    prepare_conv2d_bias,
    prepare_conv2d_weights,
    validate,
)

__all__ = [
    "EXCLUSIONS",
    "INPUT_TAGGERS",
    "PROPERTIES",
    "SUPPORTED",
    "conv2d_nhwc",
    "prepare_conv2d_bias",
    "prepare_conv2d_weights",
    "validate",
]
