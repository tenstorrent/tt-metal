// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// RMS Norm - Reader Kernel
// Reads input data from DRAM, generates scaler/eps tiles, double-pushes per row

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_scaler = 1;
constexpr uint32_t cb_eps = 2;
constexpr uint32_t cb_gamma_rm = 3;

// Compile-time args
constexpr uint32_t stick_size = get_compile_time_arg_val(0);
constexpr auto input_accessor_args = TensorAccessorArgs<1>();

void kernel_main() {
    // Runtime args
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);
    const uint32_t start_row_id = get_arg_val<uint32_t>(2);
    const uint32_t Wt = get_arg_val<uint32_t>(3);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(4);
    const uint32_t eps_u32 = get_arg_val<uint32_t>(5);

    if (num_rows == 0) {
        return;
    }

    const auto input_accessor = TensorAccessor(input_accessor_args, src_addr, stick_size);

    // Generate reduce scaler tile: 1/W for mean computation
    const float scaler_val = 1.0f / static_cast<float>(Wt * 32);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(scaler_val);

    // Generate epsilon scalar tile
    float epsilon;
    // Bit-cast uint32 back to float
    static_assert(sizeof(float) == sizeof(uint32_t));
    __builtin_memcpy(&epsilon, &eps_u32, sizeof(float));
    dataflow_kernel_lib::prepare_reduce_scaler<cb_eps>(epsilon);

    for (uint32_t row = 0; row < num_rows; ++row) {
        uint32_t row_id = start_row_id + row;

        // Double-push: read the same row data twice
        // Pass 1: feeds square + reduce (consumed by Phase 1-3)
        // Pass 2: feeds normalize mul_col (consumed by Phase 5-6)
        for (uint32_t pass = 0; pass < 2; ++pass) {
#if IS_INPUT_RM
            // RM path: read 32 sticks (one tile-row), push as Wt tile-sized pages
            cb_reserve_back(cb_in, Wt);
            uint32_t l1_write_addr = get_write_ptr(cb_in);
            uint32_t base_stick = row_id * 32;
            for (uint32_t s = 0; s < 32; ++s) {
                uint64_t noc_addr = input_accessor.get_noc_addr(base_stick + s);
                noc_async_read(noc_addr, l1_write_addr, stick_size);
                l1_write_addr += stick_size;
            }
            noc_async_read_barrier();
            cb_push_back(cb_in, Wt);
#else
            // TILE path: read Wt tiles one at a time
            uint32_t base_tile = row_id * Wt;
            for (uint32_t t = 0; t < Wt; ++t) {
                cb_reserve_back(cb_in, 1);
                uint32_t l1_write_addr = get_write_ptr(cb_in);
                uint64_t noc_addr = input_accessor.get_noc_addr(base_tile + t);
                noc_async_read(noc_addr, l1_write_addr, stick_size);
                noc_async_read_barrier();
                cb_push_back(cb_in, 1);
            }
#endif
        }
    }
}
