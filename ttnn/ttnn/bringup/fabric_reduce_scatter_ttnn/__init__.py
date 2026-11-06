# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""fabric_reduce_scatter_ttnn: ``ttnn.bringup.fabric_reduce_scatter`` (a Python ProgramDescriptor op, PYTHON_OPS):
a line / ring reduce-scatter over the fabric along one mesh axis, with the planning code of the Python
fabric_all_gather (fabric_all_gather_py.py in this folder: the Python all-gather prototype, kept here when the C++
fork ttnn.bringup.fabric_all_gather was retired). From mstaletovic/mimo-v2-dp
(ttnn/ttnn/operations/examples/fabric_reduce_scatter)."""

from .fabric_reduce_scatter import fabric_reduce_scatter, prepare

__all__ = ["fabric_reduce_scatter", "prepare"]
