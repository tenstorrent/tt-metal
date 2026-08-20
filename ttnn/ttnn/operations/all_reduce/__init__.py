# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from .all_reduce import EXCLUSIONS, INPUT_TAGGERS, SUPPORTED, all_reduce

# EXCLUSIONS / INPUT_TAGGERS / SUPPORTED are the op's registry contract (the
# golden/eval harness reads them at the package level — see
# eval/golden_tests/all_reduce/test_golden.py and the runtime xfail-gate in
# eval/golden_tests/conftest.py).
__all__ = ["all_reduce", "EXCLUSIONS", "INPUT_TAGGERS", "SUPPORTED"]
