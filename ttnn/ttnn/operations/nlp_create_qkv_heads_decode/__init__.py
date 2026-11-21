# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from .nlp_create_qkv_heads_decode import (
    EXCLUSIONS,
    INPUT_TAGGERS,
    SUPPORTED,
    nlp_create_qkv_heads_decode,
    validate,
)

__all__ = ["nlp_create_qkv_heads_decode", "SUPPORTED", "EXCLUSIONS", "INPUT_TAGGERS", "validate"]
