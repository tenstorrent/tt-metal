# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys

import ttnn

__all__ = []

ttnn.register_python_operation(name="ttnn.all_gather")(ttnn._ttnn.operations.ccl.all_gather)
