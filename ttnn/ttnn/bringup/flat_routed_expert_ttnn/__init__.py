# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""flat_routed_expert_ttnn: the C++ op ``ttnn.bringup.flat_routed_expert`` (+ ``ttnn.bringup.flat_routed_expert_plan``)
and its Python side, ``flat_expert.py``: ``FlatRoutedExpert`` (weights in the op's bank layout, per-device expert
tables; the model-facing wrapper) and ``FlatExpert`` (the Python ProgramDescriptor builder, the parity reference).
The folder is not named flat_routed_expert: a package of the op's own name would shadow ttnn.bringup.flat_routed_expert."""
