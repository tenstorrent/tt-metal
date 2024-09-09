# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import ttnn._ttnn
import ttnn

matmul = ttnn._ttnn.operations.moreh.moreh_matmul
bmm = ttnn._ttnn.operations.moreh.moreh_bmm
bmm_backward = ttnn._ttnn.operations.moreh.moreh_bmm_backward
