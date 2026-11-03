# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""balanced_tail_split bench: the golden suite's cached fabric mesh (production router payload)."""
from eval.golden_tests.ring_joint_scaled_dot_product_attention.conftest import (  # noqa: F401
    _fabric_mesh_cache,
    mesh_device,
)
