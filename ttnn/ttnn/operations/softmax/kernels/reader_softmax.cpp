// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Softmax - Reader Kernel
// Reads input tiles from DRAM into c_0 via TensorAccessor.
// Generates scaler tiles (c_1, c_2) at startup.
// For Stage 1 (data_pipeline): single pass, reads all tiles once.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_mm_scaler.hpp"

constexpr uint32_t cb_input = tt::CBIndex::c_0;
constexpr uint32_t cb_scaler = tt::CBIndex::c_1;
constexpr uint32_t cb_mm_scaler = tt::CBIndex::c_2;

void kernel_main() {
    // Compile-time args
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t num_rows_or_cols = get_compile_time_arg_val(2);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(3);
    constexpr auto tensor_accessor_args = TensorAccessorArgs<4>();

    // Runtime args
    const uint32_t input_addr = get_arg_val<uint32_t>(0);

    // Setup tensor accessor
    const uint32_t page_size = get_tile_size(cb_input);
    const auto input_accessor = TensorAccessor(tensor_accessor_args, input_addr, page_size);

    // Generate constant scaler tiles at startup
    // c_1: reduce scaler (all 1.0 in row 0 of each face) for reduce_tile<MAX>
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(1.0f);

    // c_2: matmul row-reduce scaler (column vector of 1.0 in left faces)
    // Pack 1.0 in bf16 = 0x3F80, double-packed = 0x3F803F80
    constexpr uint32_t packed_bf16_1_0 = 0x3F803F80u;
    generate_mm_scaler(cb_mm_scaler, packed_bf16_1_0);

    // Read all tiles sequentially in row-major order (single pass for stage 1)
    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        cb_reserve_back(cb_input, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_input);
        uint64_t noc_addr = input_accessor.get_noc_addr(tile_id);
        noc_async_read(noc_addr, l1_write_addr, page_size);
        noc_async_read_barrier();
        cb_push_back(cb_input, 1);
    }
}
