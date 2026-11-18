// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// t2_raw_kwalk_hxw isolated K-walk bench reader (NCRISC): push-only. cb_act / cb_weight are backed on L1 shards
// holding exactly two K-block slots, so reserve + push re-presents slot (kb % 2) every K-block -- no NoC traffic.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_act = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight = get_compile_time_arg_val(1);
    constexpr uint32_t act_block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t weight_block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(4);
    for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
        cb_reserve_back(cb_act, act_block_tiles);
        cb_push_back(cb_act, act_block_tiles);
        cb_reserve_back(cb_weight, weight_block_tiles);
        cb_push_back(cb_weight, weight_block_tiles);
    }
}
