# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Union, List
import ttnn


def run_avg_pool_on_device_wrapper(device):
    def avg_pool2d(x, output_mem_config, output_dtype=None):
        out = ttnn.avg_pool2d(x, memory_config=output_mem_config, dtype=output_dtype)
        return out

    return avg_pool2d
