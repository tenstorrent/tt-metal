// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/rand.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "ckernel.h"
#include "ckernel_defs.h"

#include <cstdint>

namespace NAMESPACE {
void MAIN {
    // Compile time args
    constexpr uint32_t kernel_communication_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t seed = get_compile_time_arg_val(1);

    // Constants
    constexpr uint32_t one_tile = 1;

    // Get message from reader
    cb_wait_front(kernel_communication_cb_index, one_tile);

    // Get communication entry
    const uint32_t message = read_tile_value(kernel_communication_cb_index, /*tile_index=*/0, /*element_offset=*/0);
    const bool is_core_id = (message != 0) ? true : false;

    // Initialize random generator if core_id matched
    if (is_core_id) {
        rand_tile_init(seed);
    }

    // Pop the communication entry
    cb_pop_front(kernel_communication_cb_index, one_tile);
}
}  // namespace NAMESPACE
