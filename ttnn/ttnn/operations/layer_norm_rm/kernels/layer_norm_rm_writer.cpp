// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Writer Kernel (stub)
//
// Waits for cb_rm_out (Wt tiles = 32 RM sticks per block).
// Writes each stick to DRAM via TensorAccessor.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Compile-time args
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    // TensorAccessorArgs follow at indices 2+

    // Runtime args
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_blocks = get_arg_val<uint32_t>(1);
    uint32_t start_stick_id = get_arg_val<uint32_t>(2);

    // Stub: do nothing -- real implementation will:
    // For each block: wait cb_rm_out, write 32 RM sticks, pop cb_rm_out
}
