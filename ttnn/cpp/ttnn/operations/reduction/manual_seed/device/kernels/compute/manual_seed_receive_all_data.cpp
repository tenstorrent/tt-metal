// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/rand.h"

#include <cstdint>

namespace NAMESPACE {
void MAIN {
    // Runtime args
    const uint32_t number_of_ids = get_arg_val<uint32_t>(0);

    // Compile time args
    constexpr uint32_t core_id = get_compile_time_arg_val(0);
    constexpr uint32_t user_ids_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t seeds_cb_index = get_compile_time_arg_val(2);

    // Constants
    constexpr uint32_t one_tile = 1;
    constexpr uint32_t metadata_fields = 4;

    // Read user_id from circular buffer
    cb_wait_front(user_ids_cb_index, one_tile);
    uint32_t* user_ids = nullptr;
    cb_get_tile(user_ids_cb_index, 0, &user_ids);
    user_ids += metadata_fields;  // Skip metadata

    // Get info
    for (uint32_t i = 0; i < number_of_ids; i++) {
        if (core_id == user_ids[i]) {
            // Read seed from circular buffer
            cb_wait_front(seeds_cb_index, one_tile);
            uint32_t* seeds = nullptr;
            cb_get_tile(seeds_cb_index, 0, &seeds);
            seeds += metadata_fields;  // Skip metadata
            const uint32_t seed = seeds[i];

            // Set random generator with seed
            rand_tile_init(seed);

            // Remove read token
            cb_pop_front(seeds_cb_index, one_tile);
            break;
        }
    }

    // Remove read token
    cb_pop_front(user_ids_cb_index, one_tile);
}
}  // namespace NAMESPACE
