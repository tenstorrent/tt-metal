# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""fabric_reduce_scatter_ttnn: ``ttnn.bringup.fabric_reduce_scatter``, the native C++ ttnn.fabric_reduce_scatter:
a line / ring reduce-scatter over the fabric along one mesh axis. fabric_all_gather_py.py in this folder is the Python
all-gather prototype, kept for its own tests. From mstaletovic/mimo-v2-dp
(ttnn/ttnn/operations/examples/fabric_reduce_scatter)."""

from .fabric_reduce_scatter import fabric_reduce_scatter, prepare

__all__ = ["fabric_reduce_scatter", "prepare"]
