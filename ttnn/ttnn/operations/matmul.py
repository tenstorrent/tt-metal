# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# TODO(nuked-op matmul): the matmul op (matmul/linear/addmm/sparse_matmul and
# all Matmul*ProgramConfig bindings) was removed. This module previously
# re-exported ttnn._ttnn.operations.matmul.* and attached golden functions for
# ttnn.matmul / ttnn.linear / ttnn.addmm. Restore those re-exports + golden
# functions when the matmul op is recreated.

__all__ = []
