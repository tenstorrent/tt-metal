// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"

/*
 * Sharded reader for madd operation.
 * Data is already in L1 (sharded), so we just need to:
 * 1. Signal that input tiles are available in the CBs
 * 2. Generate a zero tile for the compute kernel
 */
void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_srcA_index = get_compile_time_arg_val(0);
    constexpr uint32_t cb_srcB_index = get_compile_time_arg_val(1);
    constexpr uint32_t cb_srcC_index = get_compile_time_arg_val(2);
    constexpr uint32_t cb_zero_index = get_compile_time_arg_val(3);

    // Generate zero tile for compute kernel using optimized helper
    constexpr uint32_t zero = 0;
    cb_reserve_back(cb_zero_index, 1);
    generate_reduce_scaler(cb_zero_index, zero);
    cb_push_back(cb_zero_index, 1);

    // Signal that all input tiles are available (data already in L1 from sharding)
    cb_reserve_back(cb_srcA_index, num_tiles);
    cb_push_back(cb_srcA_index, num_tiles);

    cb_reserve_back(cb_srcB_index, num_tiles);
    cb_push_back(cb_srcB_index, num_tiles);

    cb_reserve_back(cb_srcC_index, num_tiles);
    cb_push_back(cb_srcC_index, num_tiles);
}
