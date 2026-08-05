# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Marker package for the `reduce_at_percall_width_1` isolated perf bench.

Present ONLY so pytest's `--import-mode=importlib` resolves this test as a leaf
module instead of `ttnn.ttnn.operations...` (which would re-execute
`ttnn/ttnn/__init__.py`).  Deliberately empty: `perf_experiments/` itself has NO
`__init__.py`, so `pkgutil.walk_packages` in `ttnn/ttnn/operations/__init__.py`
never reaches this tree (see ../README.md).
"""
