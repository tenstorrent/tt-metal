# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""gated_delta_net_backward — backward pass of the chunked gated delta rule."""

from .gated_delta_net_backward import (
    EXCLUSIONS,
    INPUT_TAGGERS,
    PROPERTIES,
    SUPPORTED,
    gated_delta_net_backward,
    validate,
)

__all__ = [
    "gated_delta_net_backward",
    "validate",
    "INPUT_TAGGERS",
    "SUPPORTED",
    "EXCLUSIONS",
    "PROPERTIES",
]
