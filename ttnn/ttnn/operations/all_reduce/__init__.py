# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""all_reduce — element-wise SUM across a MeshDevice line, identical on every device.

The registry contract (SUPPORTED / EXCLUSIONS / INPUT_TAGGERS) is re-exported at
the package level alongside the op — the golden/eval harness reads it from here.
"""

from .all_reduce import EXCLUSIONS, INPUT_TAGGERS, SUPPORTED, all_reduce

__all__ = ["all_reduce", "EXCLUSIONS", "INPUT_TAGGERS", "SUPPORTED"]
