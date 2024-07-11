# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import tt_lib as ttl

from typing import Union, List


def run_avg_pool_on_device_wrapper(device):
    def average_pool_2d(x, output_mem_config, output_dtype=None):
        out = ttl.tensor.average_pool_2d(x, output_mem_config, output_dtype)
        return out

    return average_pool_2d
