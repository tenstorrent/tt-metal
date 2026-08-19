# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from .reduce_scatter import EXCLUSIONS, INPUT_TAGGERS, SUPPORTED, reduce_scatter

# EXCLUSIONS / INPUT_TAGGERS / SUPPORTED are the op's registry contract (the
# golden/eval harness reads them at the package level — see
# eval/golden_tests/reduce_scatter/test_golden.py and the runtime xfail-gate in
# eval/golden_tests/conftest.py).
__all__ = ["reduce_scatter", "EXCLUSIONS", "INPUT_TAGGERS", "SUPPORTED"]
