// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"

void kernel_main() {
    // Simplest program possible
    // 1) Only 1 tile
    // 2) float32 elements
    // 3) Along x-axis

    constexpr uint32_t tile_size = 32;  // 'reader' does everythingß

    uint32_t src_dram = get_arg_val<uint32_t>(0);
    uint32_t dst_dram = get_arg_val<uint32_t>(1);

    float* src = (float*)l1_addr_in;
    float* dst = (float*)l1_addr_out;

    float sum = 0.f;
    for (uint32_t i = 0; i < tile_size; i++) {
        sum += src[i];
        dst[i] = sum;
    }
}
