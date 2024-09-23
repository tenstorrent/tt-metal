# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

adam = ttnn._ttnn.operations.moreh.moreh_adam
arange = ttnn._ttnn.operations.moreh.moreh_arange
getitem = ttnn._ttnn.operations.moreh.moreh_getitem
matmul = ttnn._ttnn.operations.moreh.moreh_matmul
mean = ttnn._ttnn.operations.moreh.moreh_mean
mean_backward = ttnn._ttnn.operations.moreh.moreh_mean_backward
norm = ttnn._ttnn.operations.moreh.moreh_norm
norm_backward = ttnn._ttnn.operations.moreh.moreh_norm_backward
sum = ttnn._ttnn.operations.moreh.moreh_sum
