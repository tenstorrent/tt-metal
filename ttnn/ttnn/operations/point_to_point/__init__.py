# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Self-contained Python point_to_point CCL op (generic_op + MeshProgramDescriptor).

Re-exports the registry contract (SUPPORTED / EXCLUSIONS / INPUT_TAGGERS) at the
package level so the golden/eval harness can import them from the package.
"""

from .point_to_point import EXCLUSIONS, INPUT_TAGGERS, SUPPORTED, point_to_point

__all__ = ["point_to_point", "EXCLUSIONS", "INPUT_TAGGERS", "SUPPORTED"]
