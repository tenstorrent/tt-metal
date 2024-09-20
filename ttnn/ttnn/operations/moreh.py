# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

arange = ttnn._ttnn.operations.moreh.moreh_arange
adam = ttnn._ttnn.operations.moreh.moreh_adam
getitem = ttnn._ttnn.operations.moreh.moreh_getitem
sum = ttnn._ttnn.operations.moreh.moreh_sum
mean = ttnn._ttnn.operations.moreh.moreh_mean
mean_backward = ttnn._ttnn.operations.moreh.moreh_mean_backward
matmul = ttnn._ttnn.operations.moreh.moreh_matmul
