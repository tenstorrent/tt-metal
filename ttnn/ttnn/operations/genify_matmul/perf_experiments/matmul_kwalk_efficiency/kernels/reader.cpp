// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// K-walk bench reader (NCRISC): push-only. cb_act / cb_weight are backed on L1 shards that hold
// exactly two K-block slots each, so reserve + push without writing re-presents slot (kb % 2) to the
// compute for every K-block. No NoC traffic -- the bench isolates the compute K-walk.
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
