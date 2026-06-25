// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Softmax reader kernel (NCRISC/RISCV_1).
//
// Per slab (one (N,C) pair):
//   - Reads Ht×Wt input tiles from DRAM/L1 into cb_input_tiles
//   - Tiles are in row-major tile order within each slab
//
// At kernel start (once):
//   - Prepares cb_scaler_max (1 tile, bf16) via prepare_reduce_scaler<MAX, REDUCE_ROW/COL>
//   - Prepares cb_scaler_sum (1 tile, bf16) via prepare_reduce_scaler<SUM, REDUCE_ROW/COL>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

namespace {
// CB indices — must match program descriptor
constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_scaler_max = 1;
constexpr uint32_t cb_scaler_sum = 2;
}  // namespace

void kernel_main() {
    uint32_t input_buffer_address = get_arg_val<uint32_t>(0);
    uint32_t start_tile_id = get_arg_val<uint32_t>(1);
    uint32_t num_slabs = get_arg_val<uint32_t>(2);

    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    // dim is passed as uint32_t (two's complement); cast to int32_t to recover -1 or -2
    constexpr int32_t dim = static_cast<int32_t>(get_compile_time_arg_val(2));

    // CT args: 3 scalar, then TensorAccessorArgs
    constexpr auto src_args = TensorAccessorArgs<3>();
    const auto src_accessor = TensorAccessor(src_args, input_buffer_address);

    CircularBuffer input_cb(cb_input_tiles);
    Noc noc;
    const uint32_t tile_bytes = get_tile_size(cb_input_tiles);

    // Prepare scaler tiles once at kernel start
    if constexpr (dim == -1) {
        dataflow_kernel_lib::
            prepare_reduce_scaler<cb_scaler_max, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>(1.0f);
        dataflow_kernel_lib::
            prepare_reduce_scaler<cb_scaler_sum, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(1.0f);
    } else {
        dataflow_kernel_lib::
            prepare_reduce_scaler<cb_scaler_max, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_COL>(1.0f);
        dataflow_kernel_lib::
            prepare_reduce_scaler<cb_scaler_sum, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_COL>(1.0f);
    }

    uint32_t tiles_per_slab = Ht * Wt;
    uint32_t tile_id = start_tile_id;

    for (uint32_t slab = 0; slab < num_slabs; ++slab) {
        // Read Ht×Wt tiles for this slab in row-major tile order
        for (uint32_t i = 0; i < tiles_per_slab; ++i) {
            input_cb.reserve_back(1);
            noc.async_read(src_accessor, input_cb, tile_bytes, {.page_id = tile_id}, {.offset_bytes = 0});
            noc.async_read_barrier();
            input_cb.push_back(1);
            tile_id++;
        }
    }
}
