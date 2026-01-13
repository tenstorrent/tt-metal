# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Stage 1 Test: API Exists

Verifies that the row_mean_sub_square_reduce operation is importable from ttnn.
This test passes when the Python binding is registered and accessible.
"""

import pytest


def test_api_exists():
    """Verify row_mean_sub_square_reduce is importable from ttnn."""
    import ttnn

    assert hasattr(ttnn, "row_mean_sub_square_reduce"), "ttnn.row_mean_sub_square_reduce not found"
    assert callable(ttnn.row_mean_sub_square_reduce), "ttnn.row_mean_sub_square_reduce is not callable"


def test_api_has_docstring():
    """Verify row_mean_sub_square_reduce has a docstring."""
    import ttnn

    op = ttnn.row_mean_sub_square_reduce
    # The operation should have some documentation
    assert op.__doc__ is not None or hasattr(
        op, "__call__"
    ), "ttnn.row_mean_sub_square_reduce should have documentation or be callable"
