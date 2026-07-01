# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Measured auto-configuration helpers exposed as ``ttnn.experimental.auto_config``."""

from .matmul import explain_matmul

__all__ = [
    "explain_matmul",
]
