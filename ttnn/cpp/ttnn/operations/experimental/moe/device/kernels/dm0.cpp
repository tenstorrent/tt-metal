// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Compile time arguments
    constexpr auto in_args = TensorAccessorArgs<0>();
    constexpr auto w0_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr auto w1_args = TensorAccessorArgs<w0_args.next_compile_time_args_offset()>();
    constexpr auto w2_args = TensorAccessorArgs<w1_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<w2_args.next_compile_time_args_offset()>();

    // Run-time arguments
    uint32_t argidx = 0;
    const auto core_id = get_arg_val<uint32_t>(argidx++);
    const auto in_addr = get_arg_val<uint32_t>(argidx++);
    const auto w0_addr = get_arg_val<uint32_t>(argidx++);
    const auto w1_addr = get_arg_val<uint32_t>(argidx++);
    const auto w2_addr = get_arg_val<uint32_t>(argidx++);
    const auto out_addr = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_r2c_w0 = tt::CBIndex::c_0;
    constexpr auto cb_s2c_in = tt::CBIndex::c_1;
    constexpr auto cb_c2c_mm0 = tt::CBIndex::c_2;
    constexpr auto cb_c2c_mm1 = tt::CBIndex::c_3;
    constexpr auto cb_c2w_elt = tt::CBIndex::c_4;
    constexpr auto cb_r2c_in2 = tt::CBIndex::c_5;
    constexpr auto cb_c2w_mm2 = tt::CBIndex::c_6;

    // CB Aliases
    constexpr auto cb_r2c_w1 = tt::CBIndex::c_0;
    constexpr auto cb_r2c_w2 = tt::CBIndex::c_0;

    // Tile sizes
    constexpr uint32_t in_tile_size = get_tile_size(cb_s2c_in);
    constexpr uint32_t w0_tile_size = get_tile_size(cb_r2c_w0);
    constexpr uint32_t w1_tile_size = get_tile_size(cb_r2c_w1);
    constexpr uint32_t w2_tile_size = get_tile_size(cb_r2c_w2);
    constexpr uint32_t out_tile_size = get_tile_size(cb_c2w_elt);

    // Tensor accessors
    const auto in_accessor = TensorAccessor(in_args, in_addr, in_tile_size);
    const auto w0_accessor = TensorAccessor(w0_args, w0_addr, w0_tile_size);
    const auto w1_accessor = TensorAccessor(w1_args, w1_addr, w1_tile_size);
    const auto w2_accessor = TensorAccessor(w2_args, w2_addr, w2_tile_size);
    const auto out_accessor = TensorAccessor(out_args, out_addr, out_tile_size);

    // Constants for MoE
    constexpr uint32_t num_w0_w1_tiles = 224;
    constexpr uint32_t num_w2_tiles_h = 64;
    const uint32_t num_w2_tiles_w = core_id < 32 ? 4 : 3;
    const uint32_t num_mm2_tiles = core_id < 32 ? 4 : 3;

    constexpr uint32_t num_elt_tiles = 1;

    constexpr uint32_t w0_w1_stride = 64;
    constexpr uint32_t w2_stride_w = 1;
    constexpr uint32_t w2_stride_h = 224;

    const uint32_t w0_tile_id_start = core_id;
    const uint32_t w1_tile_id_start = core_id;
    const uint32_t w2_tile_id_start = core_id < 32 ? 4 * core_id : 4 * 32 + 3 * (core_id - 32);

    // // Read W0 and W1 from DRAM into CB
    uint32_t w0_tile_id = w0_tile_id_start;
    uint32_t w1_tile_id = w1_tile_id_start;
    for (uint32_t i = 0; i < num_w0_w1_tiles; ++i) {
        cb_reserve_back(cb_r2c_w0, 1);
        uint32_t cb_r2c_w0_addr = get_write_ptr(cb_r2c_w0);
        // noc_async_read_tile(w0_tile_id, w0_accessor, cb_r2c_w0_addr);
        // noc_async_read_barrier();
        cb_push_back(cb_r2c_w0, 1);
        w0_tile_id += w0_w1_stride;
    }
    for (uint32_t i = 0; i < num_w0_w1_tiles; ++i) {
        cb_reserve_back(cb_r2c_w1, 1);
        uint32_t cb_r2c_w1_addr = get_write_ptr(cb_r2c_w1);
        // noc_async_read_tile(w1_tile_id, w1_accessor, cb_r2c_w1_addr);
        // noc_async_read_barrier();
        cb_push_back(cb_r2c_w1, 1);
        w1_tile_id += w0_w1_stride;
    }

    // Read W2 from DRAM into CB
    uint32_t w2_tile_id_h_start = w2_tile_id_start;
    for (uint32_t i = 0; i < num_w2_tiles_h; ++i) {
        uint32_t w2_tile_id = w2_tile_id_h_start;
        for (uint32_t j = 0; j < num_w2_tiles_w; ++j) {
            cb_reserve_back(cb_r2c_w2, 1);
            uint32_t cb_r2c_w2_addr = get_write_ptr(cb_r2c_w2);
            // noc_async_read_tile(w2_tile_id, w2_accessor, cb_r2c_w2_addr);
            // noc_async_read_barrier();
            cb_push_back(cb_r2c_w2, 1);
            w2_tile_id += w2_stride_w;
        }
        w2_tile_id_h_start += w2_stride_h;
    }
}
