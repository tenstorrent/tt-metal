// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {
    // Compile time args
    constexpr uint32_t shard_width_in_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t shard_height_in_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t batch = get_compile_time_arg_val(2);

    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t shard_size_in_tiles = shard_width_in_tiles * shard_height_in_tiles;

    for (uint32_t b = 0; b < batch; ++b) {
        cb_reserve_back(cb_id_in1, shard_size_in_tiles);
        cb_push_back(cb_id_in1, shard_size_in_tiles);
    }
}
