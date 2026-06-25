// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Softmax reader kernel (NCRISC/RISCV_1).
//
// Per slab (one (N,C) pair):
//   TILE path:
//     - Reads Ht×Wt input tiles from DRAM/L1 into cb_input_tiles
//     - Tiles are in row-major tile order within each slab
//
//   ROW_MAJOR path:
//     - Reads Ht×Wt row-major sticks from DRAM/L1 into cb_rm_in
//       via read_sticks_for_tilize (TILE granularity)
//     - Compute kernel will tilize cb_rm_in → cb_input_tiles
//
// At kernel start (once):
//   - Prepares cb_scaler_max (1 tile, bf16) via prepare_reduce_scaler<MAX, REDUCE_ROW/COL>
//   - Prepares cb_scaler_sum (1 tile, bf16) via prepare_reduce_scaler<SUM, REDUCE_ROW/COL>
//
// CT args: Ht, Wt, dim(u32), is_rm, then TensorAccessorArgs starting at index 4
// RT args: input_buffer_address, start_id, num_slabs
//   TILE: start_id = starting tile index
//   RM:   start_id = starting stick (page) index

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

namespace {
// CB indices — must match program descriptor
constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_scaler_max = 1;
constexpr uint32_t cb_scaler_sum = 2;
constexpr uint32_t cb_rm_in = 3;
}  // namespace

void kernel_main() {
    uint32_t input_buffer_address = get_arg_val<uint32_t>(0);
    uint32_t start_id = get_arg_val<uint32_t>(1);
    uint32_t num_slabs = get_arg_val<uint32_t>(2);

    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr int32_t dim = static_cast<int32_t>(get_compile_time_arg_val(2));
    constexpr uint32_t is_rm = get_compile_time_arg_val(3);

    // CT args: 4 scalar, then TensorAccessorArgs
    constexpr auto src_args = TensorAccessorArgs<4>();
    const auto src_accessor = TensorAccessor(src_args, input_buffer_address);

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

    if constexpr (is_rm) {
        // ===== ROW_MAJOR path: read sticks into cb_rm_in =====
        // read_sticks_for_tilize (TILE granularity) reads 32 rows per call,
        // producing Wt tile-sized pages in cb_rm_in.
        // For tile-aligned shapes: row_bytes = W * elem_size = Wt * tile_size / 32.
        constexpr uint32_t tile_h = 32;
        const uint32_t tile_size = get_tile_size(cb_rm_in);
        const uint32_t row_bytes = Wt * tile_size / 32;

        // start_id is a stick (page) index in the RM tensor.
        // Each slab has H = Ht * 32 sticks.
        uint32_t stick_id = start_id;

        for (uint32_t slab = 0; slab < num_slabs; ++slab) {
            // Read Ht tile-rows, each with tile_h (32) sticks
            for (uint32_t tr = 0; tr < Ht; ++tr) {
                dataflow_kernel_lib::read_sticks_for_tilize<cb_rm_in>(
                    src_accessor,
                    tile_h,                  // total_num_rows = 32
                    row_bytes,               // bytes per stick
                    stick_id + tr * tile_h,  // start_page (stick index)
                    0                        // byte_offset_within_page
                );
            }
            stick_id += Ht * tile_h;  // H = Ht * 32 sticks per slab
        }
    } else {
        // ===== TILE path: read tiles directly into cb_input_tiles =====
        CircularBuffer input_cb(cb_input_tiles);
        Noc noc;
        const uint32_t tile_bytes = get_tile_size(cb_input_tiles);

        uint32_t tiles_per_slab = Ht * Wt;
        uint32_t tile_id = start_id;

        for (uint32_t slab = 0; slab < num_slabs; ++slab) {
            for (uint32_t i = 0; i < tiles_per_slab; ++i) {
                input_cb.reserve_back(1);
                noc.async_read(src_accessor, input_cb, tile_bytes, {.page_id = tile_id}, {.offset_bytes = 0});
                noc.async_read_barrier();
                input_cb.push_back(1);
                tile_id++;
            }
        }
    }
}
