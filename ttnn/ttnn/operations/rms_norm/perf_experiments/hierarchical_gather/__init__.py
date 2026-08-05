# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Present ONLY so pytest imports test_hierarchical_gather.py as a package module
# (without it, pytest prepends this dir to sys.path and ttnn gets imported twice).
# Nothing here is imported by the op.
