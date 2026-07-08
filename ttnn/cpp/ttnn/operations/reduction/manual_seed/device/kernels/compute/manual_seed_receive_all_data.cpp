// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_unary/rand.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ckernel.h"
#include "ckernel_defs.h"

#include <cstdint>

void kernel_main() {
    // Compile time args
    constexpr uint32_t kernel_communication_dfb_index = get_compile_time_arg_val(0);

    // Constants
    constexpr uint32_t one_tile = 1;

    DataflowBuffer kernel_communication_dfb(kernel_communication_dfb_index);

    // Get message from reader
    kernel_communication_dfb.wait_front(one_tile);

    // Read core ID from message
    const uint32_t message_core_id = kernel_communication_dfb.read_tile_value(/*tile_index=*/0, /*element_offset=*/0);
    const bool is_core_id = (message_core_id != 0) ? true : false;

    if (is_core_id) {
        // Read seed from message
        const uint32_t seed = kernel_communication_dfb.read_tile_value(/*tile_index=*/0, /*element_offset=*/1);

        // A seed value of UINT32_MAX (0xFFFFFFFF) is a special value
        // that skips rand_tile_init, leaving the PRNG state unchanged.
        if (seed != UINT32_MAX) {
            rand_tile_init(seed);
        }
    }

    // Pop the communication entry
    kernel_communication_dfb.pop_front(one_tile);
}
