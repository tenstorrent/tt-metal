# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from .point_to_point import EXCLUSIONS, INPUT_TAGGERS, SUPPORTED, point_to_point

# EXCLUSIONS / INPUT_TAGGERS / SUPPORTED are the op's registry contract (the
# golden/eval harness reads them at the package level — see
# eval/golden_tests/point_to_point/test_golden.py and the runtime xfail-gate in
# eval/golden_tests/conftest.py).
__all__ = ["point_to_point", "EXCLUSIONS", "INPUT_TAGGERS", "SUPPORTED"]
