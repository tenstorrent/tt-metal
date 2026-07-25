# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""all_reduce — self-contained Python CCL + compute op.

Re-exports the public entry point AND the registry-model contract
(``SUPPORTED`` / ``EXCLUSIONS`` / ``INPUT_TAGGERS``) at PACKAGE level, which is
where the golden/eval harness reads them from.
"""

from .all_reduce import EXCLUSIONS, INPUT_TAGGERS, SUPPORTED, all_reduce

__all__ = ["all_reduce", "EXCLUSIONS", "INPUT_TAGGERS", "SUPPORTED"]
