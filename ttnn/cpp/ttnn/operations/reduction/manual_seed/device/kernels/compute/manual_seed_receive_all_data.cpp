// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/rand.h"
#include "ckernel.h"
#include "ckernel_defs.h"

#include <cstdint>

namespace NAMESPACE {
void MAIN {
    // Compile time args
    constexpr uint32_t kernel_communication_cb_index = get_compile_time_arg_val(0);

    // Constants
    constexpr uint32_t one_tile = 1;

    // Get message from reader
    cb_wait_front(kernel_communication_cb_index, one_tile);

    // Read core ID from message
    const uint32_t message_core_id =
        read_tile_value(kernel_communication_cb_index, /*tile_index=*/0, /*element_offset=*/0);
    const bool is_core_id = (message_core_id != 0) ? true : false;

    if (is_core_id) {
        // Read seed from message
        const uint32_t seed = read_tile_value(kernel_communication_cb_index, /*tile_index=*/0, /*element_offset=*/1);

        // Set random generator with seed
        rand_tile_init(seed);
    }

    // Pop the communication entry
    cb_pop_front(kernel_communication_cb_index, one_tile);
}
}  // namespace NAMESPACE
