# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# Present only so pytest resolves this dir as the package root of
# test_tree_combine.py.  Without it, pytest 9's importlib mode names the module
# from the REPO root and imports the checkout a second time as `ttnn.ttnn`,
# which trips ttnn's "Operation ... is already registered" guard at collection.
