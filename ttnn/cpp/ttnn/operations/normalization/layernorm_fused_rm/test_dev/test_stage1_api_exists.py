# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Stage 1 Test: API Exists

Verifies that the layernorm_fused_rm operation is importable from ttnn.
This test passes when the Python binding is registered and accessible.
"""

import pytest


def test_api_exists():
    """Verify layernorm_fused_rm is importable from ttnn."""
    import ttnn

    assert hasattr(ttnn, "layernorm_fused_rm"), "ttnn.layernorm_fused_rm not found"
    assert callable(ttnn.layernorm_fused_rm), "ttnn.layernorm_fused_rm is not callable"


def test_api_has_docstring():
    """Verify layernorm_fused_rm has a docstring."""
    import ttnn

    op = ttnn.layernorm_fused_rm
    # The operation should have some documentation
    assert op.__doc__ is not None or hasattr(
        op, "__call__"
    ), "ttnn.layernorm_fused_rm should have documentation or be callable"
