# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Package marker for THIS experiment dir only.  Do NOT add one to the parent
# perf_experiments/ dir: ttnn/ttnn/operations/__init__.py walk_packages-execs
# every importable subpackage below it during `import ttnn`, so a parent marker
# makes every sibling experiment run at ttnn import time (and one broken WIP
# sibling then breaks `import ttnn` repo-wide).  With the marker here only,
# pytest --import-mode=importlib still resolves this test to
# `reduce_pack_to_handoff.test_...` (rooted at perf_experiments/, which is NOT a
# package) instead of `ttnn.ttnn....`, which double-executes ttnn/ttnn/__init__.py.
