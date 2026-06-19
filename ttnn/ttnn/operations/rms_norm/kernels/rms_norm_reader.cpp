// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rms_norm reader (Regime A).
//   - prepares the SUM/REDUCE_ROW scaler tile (value 1.0, col-0 matmul fill)
//   - reads gamma once into cb_gamma (held resident across this core's rows)
//   - reads each owned tile-row's Wt tiles into cb_input_resident (read once, P1)

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_row = get_arg_val<uint32_t>(2);
    const uint32_t num_rows = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_input_resident = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(4);
    constexpr auto input_args = TensorAccessorArgs<5>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    using dataflow_kernel_lib::PoolType;
    using dataflow_kernel_lib::ReduceDim;

    // SUM scaler = 1.0, col-0 (matmul) fill for SUM + REDUCE_ROW.
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>();

    const uint32_t tile_bytes = get_tile_size(cb_input_resident);
    const auto input_accessor = TensorAccessor(input_args, input_addr, tile_bytes);

    // gamma read once, held resident.
    if constexpr (has_gamma) {
        const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma);
        const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr, gamma_tile_bytes);
        cb_reserve_back(cb_gamma, Wt);
        uint32_t l1 = get_write_ptr(cb_gamma);
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            noc_async_read_tile(wt, gamma_accessor, l1);
            l1 += gamma_tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_gamma, Wt);
    }

    for (uint32_t row = 0; row < num_rows; ++row) {
        const uint32_t global_row = start_row + row;
        const uint32_t page_base = global_row * Wt;

        cb_reserve_back(cb_input_resident, Wt);
        uint32_t l1 = get_write_ptr(cb_input_resident);
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            noc_async_read_tile(page_base + wt, input_accessor, l1);
            l1 += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_input_resident, Wt);
    }
}
