# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf-tournament experiment artifacts for tilize. NOT part of the op.

Each subdirectory is one idea's isolated micro-benchmark from a perf round —
its own kernels, its own host bench and its own pytest — kept because a measured
NULL is as durable a result as a measured win, and the next round should not
re-run an experiment that has already been done.

`__path__` IS DELIBERATELY EMPTIED BELOW, and that is load-bearing rather than
tidy-up. `ttnn/ttnn/operations/__init__.py` walks this package tree with
`pkgutil.walk_packages(__path__)` and **executes every module it finds**, so
without this line every `import ttnn` anywhere would run these benches'
module-level code — which monkeypatches the op's program descriptor, so a stale
or renamed hook here would break `import ttnn` itself for the whole repo. That
is not hypothetical: it happened, as an `AttributeError: module
'...tilize_program_descriptor' has no attribute '_ablation_defines'` raised out
of `conftest.py` on a tree where the experiment was newer than the op.

Emptying `__path__` makes this a leaf package for `walk_packages` while leaving
the directory perfectly importable by explicit path — which is all the pytest
harnesses in it ever need (`scripts/run_safe_pytest.sh <that file>`).
"""

__path__ = []
