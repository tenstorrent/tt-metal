// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <vector>
#include "tt-metal/tt_metal.hpp"

// Kernel implementation will be added later

TT_METAL_DEFINE_KERNEL(
    reader_all_reduce_create_qkv_heads,
    TT_METAL_KERNEL_ARGS(
        uint32_t* input_buffer,
        uint32_t* q_output_buffer,
        uint32_t* k_output_buffer,
        uint32_t* v_output_buffer,
        uint32_t batch_size,
        uint32_t head_size,
        uint32_t num_q_heads,
        uint32_t num_kv_heads)) {
    // Reader kernel implementation will be added later
}
