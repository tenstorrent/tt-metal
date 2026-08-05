# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Present ONLY so pytest imports test_slot_tree_gather.py as a package module
# (without it, pytest prepends this dir to sys.path and ttnn gets imported twice).
# Nothing here is imported by the op.  See perf_experiments/README.md for why
# perf_experiments/ itself must NOT have one.
