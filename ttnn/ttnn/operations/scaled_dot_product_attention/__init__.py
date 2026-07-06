# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Scaled Dot Product Attention (Flash Attention) for TTNN."""

from .scaled_dot_product_attention import scaled_dot_product_attention

__all__ = ["scaled_dot_product_attention"]
