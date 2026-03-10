// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Reader Kernel (stub)
//
// Reads RM sticks from DRAM into cb_rm_in (tilize input).
// Prepares reduce scaler (1/W) in cb_reduce_scaler.
// Prepares epsilon tile in cb_eps.
// Optionally reads gamma/beta tiles into cb_gamma/cb_beta.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    // Compile-time args
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    // TensorAccessorArgs follow at indices 1+

    // Runtime args
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_blocks = get_arg_val<uint32_t>(1);
    uint32_t start_stick_id = get_arg_val<uint32_t>(2);
    uint32_t Wt = get_arg_val<uint32_t>(3);
    uint32_t has_gamma = get_arg_val<uint32_t>(4);
    uint32_t has_beta = get_arg_val<uint32_t>(5);
    uint32_t gamma_addr = get_arg_val<uint32_t>(6);
    uint32_t beta_addr = get_arg_val<uint32_t>(7);
    uint32_t eps_packed = get_arg_val<uint32_t>(8);

    // Stub: do nothing -- real implementation will:
    // 1. Prepare reduce scaler (1/W) in cb_reduce_scaler
    // 2. Fill epsilon tile in cb_eps
    // 3. Optionally read gamma/beta tiles
    // 4. For each block: read 32 RM sticks into cb_rm_in (Wt tiles)
}
