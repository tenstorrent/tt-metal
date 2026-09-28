# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""reduce_scatter — element-wise sum across a MeshDevice line, scattered.

The registry contract (SUPPORTED / EXCLUSIONS / INPUT_TAGGERS) is re-exported at
the package level alongside the op — the golden/eval harness reads it from here.
"""

from .reduce_scatter import EXCLUSIONS, INPUT_TAGGERS, SUPPORTED, reduce_scatter

__all__ = ["reduce_scatter", "EXCLUSIONS", "INPUT_TAGGERS", "SUPPORTED"]
