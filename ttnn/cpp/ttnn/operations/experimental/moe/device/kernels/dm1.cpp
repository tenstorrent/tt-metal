// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Compile time arguments
    constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");
    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");

    constexpr auto in_args = TensorAccessorArgs<0>();
    constexpr auto w0_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr auto w1_args = TensorAccessorArgs<w0_args.next_compile_time_args_offset()>();
    constexpr auto w2_args = TensorAccessorArgs<w1_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<w2_args.next_compile_time_args_offset()>();

    // Run-time arguments
    uint32_t argidx = 0;
    const auto core_id = get_arg_val<uint32_t>(argidx++);
    const auto vchannel = get_arg_val<uint32_t>(argidx++);
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
    constexpr uint32_t num_w0_w1_tiles_h = 224;
    constexpr uint32_t num_w2_tiles_h = 64;

    const uint32_t num_w0_w1_tiles_w = (core_id < 8) ? 5 : 6;
    const uint32_t num_w2_tiles_w = (core_id < 8) ? 18 : 20;

    const uint32_t num_elt_tiles = num_w0_w1_tiles_w;
    const uint32_t num_in2_tiles = num_w2_tiles_w;
    const uint32_t num_mm2_tiles = num_w2_tiles_w;

    for (uint32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        // Write from cb_c2w_elt
        for (uint32_t i = 0; i < num_elt_tiles; ++i) {
            cb_wait_front(cb_c2w_elt, 1);
            cb_pop_front(cb_c2w_elt, 1);
        }

        // Read to cb_r2c_in2
        for (uint32_t i = 0; i < num_w2_tiles_h; ++i) {
            // cb_reserve_back(cb_r2c_in2, 1);
            // cb_push_back(cb_r2c_in2, 1);
        }

        // Write from cb_c2w_mm2
        for (uint32_t i = 0; i < (num_mm2_tiles / 2); ++i) {
            cb_wait_front(cb_c2w_mm2, 2);
            cb_pop_front(cb_c2w_mm2, 2);
        }
    }
}
