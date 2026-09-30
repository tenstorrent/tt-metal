# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""pytest entry of the prelude_off_critical_path bench (the implementation lives in bench/prelude_bench.py).

ttnn.operations walks and executes every module under it at `import ttnn`; this shim does nothing then (the walker
names it "mhc_pre.perf_experiments..."), so the bench's pytest / eval imports never ride other users' ttnn import.
"""
if not __name__.startswith("mhc_pre."):
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench"))
    from prelude_bench import test_correctness, test_perf  # noqa: F401
