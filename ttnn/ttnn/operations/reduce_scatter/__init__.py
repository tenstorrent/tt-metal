# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""reduce_scatter — self-contained Python CCL op (generic_op + MeshProgramDescriptor).

Registry contract: the golden/eval harness reads SUPPORTED / EXCLUSIONS /
INPUT_TAGGERS at the PACKAGE level, so they are re-exported here alongside the op.
"""

from .reduce_scatter import EXCLUSIONS, INPUT_TAGGERS, SUPPORTED, reduce_scatter

__all__ = ["reduce_scatter", "EXCLUSIONS", "INPUT_TAGGERS", "SUPPORTED"]
