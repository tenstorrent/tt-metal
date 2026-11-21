# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from .slice import slice, SUPPORTED, EXCLUSIONS, INPUT_TAGGERS, validate  # noqa: F401  (shadows builtin by design)

__all__ = ["slice", "SUPPORTED", "EXCLUSIONS", "INPUT_TAGGERS", "validate"]
