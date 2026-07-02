# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Self-contained Python all_gather CCL op (generic_op + MeshProgramDescriptor)."""

from .all_gather import EXCLUSIONS, INPUT_TAGGERS, SUPPORTED, all_gather

__all__ = ["all_gather", "EXCLUSIONS", "INPUT_TAGGERS", "SUPPORTED"]
