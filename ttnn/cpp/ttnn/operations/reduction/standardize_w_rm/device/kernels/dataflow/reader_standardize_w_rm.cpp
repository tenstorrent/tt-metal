// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"

void kernel_main() {
    // ============================================================
    // Compile-time args
    // ============================================================
    constexpr uint32_t input_stick_size = get_compile_time_arg_val(0);
    constexpr uint32_t packed_scaler_value = get_compile_time_arg_val(1);
    constexpr uint32_t packed_epsilon_value = get_compile_time_arg_val(2);
    constexpr uint32_t Ht = get_compile_time_arg_val(3);
    constexpr uint32_t Wt = get_compile_time_arg_val(4);
    constexpr auto tensor_args = TensorAccessorArgs<5>();

    // ============================================================
    // Runtime args
    // ============================================================
    const uint32_t src_addr = get_arg_val<uint32_t>(0);

    // ============================================================
    // CB IDs (matching kernel_design.md)
    // ============================================================
    constexpr uint32_t cb_in_rm = tt::CBIndex::c_0;    // Input RM sticks
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;   // Scaler tile (1/W)
    constexpr uint32_t cb_epsilon = tt::CBIndex::c_7;  // Epsilon scalar tile

    // ============================================================
    // Phase 0: Initialize scalers (once at program start)
    // ============================================================

    // Generate scaler tile (1/W) for mean and variance calculations
    // The scaler is a tile with 1/W packed into the appropriate positions
    // for reduce operations (bcast19 pattern)
    generate_reduce_scaler(cb_scaler, packed_scaler_value);

    // Generate epsilon scalar tile for numerical stability in rsqrt
    // Using same reduce_scaler pattern as 1/W since we're adding to variance
    // which has Col0 valid (column of values from REDUCE_ROW)
    generate_reduce_scaler(cb_epsilon, packed_epsilon_value);

    // ============================================================
    // Phase 1: Read input sticks (per tile-row)
    // ============================================================
    // Create tensor accessor for input
    const auto accessor = TensorAccessor(tensor_args, src_addr, input_stick_size);

    // Each tile-row consists of 32 sticks
    // CB push/pop uses Wt as the page count (tilize helper expects Wt pages)
    constexpr uint32_t TILE_HEIGHT = 32;
    uint32_t stick_id = 0;

    for (uint32_t ht = 0; ht < Ht; ++ht) {
        // Reserve Wt pages (tilize helper expects Wt pages)
        cb_reserve_back(cb_in_rm, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_in_rm);

        // Read 32 sticks for this tile-row
        for (uint32_t s = 0; s < TILE_HEIGHT; ++s) {
            uint64_t noc_addr = accessor.get_noc_addr(stick_id);
            noc_async_read(noc_addr, l1_write_addr, input_stick_size);
            l1_write_addr += input_stick_size;
            stick_id++;
        }
        noc_async_read_barrier();

        // Signal data ready for compute (push Wt pages)
        cb_push_back(cb_in_rm, Wt);
    }
}
