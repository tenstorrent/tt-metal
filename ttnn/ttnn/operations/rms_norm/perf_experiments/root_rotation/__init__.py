# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Present ONLY so pytest imports test_root_rotation.py as a package module
# (without it, pytest prepends this dir to sys.path and ttnn gets imported twice).
# Nothing here is imported by the op.  Do NOT add an __init__.py to
# perf_experiments/ itself -- see perf_experiments/README.md.
