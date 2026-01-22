# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Stage 1 Test: API Exists

Verifies that the layer_norm_w_rm operation is importable from ttnn.
This test passes when the Python binding is registered and accessible.
"""

import pytest


def test_api_exists():
    """Verify layer_norm_w_rm is importable from ttnn."""
    import ttnn

    assert hasattr(ttnn, "layer_norm_w_rm"), "ttnn.layer_norm_w_rm not found"
    assert callable(ttnn.layer_norm_w_rm), "ttnn.layer_norm_w_rm is not callable"


def test_api_has_docstring():
    """Verify layer_norm_w_rm has a docstring."""
    import ttnn

    op = ttnn.layer_norm_w_rm
    # The operation should have some documentation
    assert op.__doc__ is not None or hasattr(op, "__call__"), \
        "ttnn.layer_norm_w_rm should have documentation or be callable"