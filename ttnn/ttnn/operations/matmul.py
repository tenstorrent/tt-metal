# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# TODO(nuked-op matmul): The matmul operation was nuked from the codebase.
# This module previously aliased ttnn._ttnn.operations.matmul.* program-config
# types and attached golden functions to ttnn.matmul / ttnn.linear / ttnn.addmm.
# All of that is gone with the op. The agent recreating matmul should restore
# the program-config aliases, golden functions, and the ttnn.Tensor.__matmul__
# operator here.

__all__ = []
