// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <tools/profiler/kernel_profiler.hpp>
#include "compute_common.hpp"

namespace NAMESPACE {

void MAIN {
    // Extract compile-time arguments
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);             // inner block size in tiles
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(1);     // out_subblock_h*in0_block_w
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(2);     // out_subblock_w*in0_block_w
    constexpr uint32_t in1_block_w = get_compile_time_arg_val(3);             // out_subblock_w
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(4);          // inner row block size in tiles
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(5);          // inner column block size in tiles
    constexpr uint32_t out_subblock_num_tiles = get_compile_time_arg_val(6);  // out_subblock_h * out_subblock_w
    constexpr bool untilize_out = get_compile_time_arg_val(7);                // untilize output

    // Define circular buffer indices
    // constexpr uint32_t in0_cb_id = tt::CBIndex::c_0;
    // constexpr uint32_t in1_cb_id = tt::CBIndex::c_1;
    // constexpr uint32_t out_cb_id = tt::CBIndex::c_4;
    // constexpr uint32_t mm_partials_cb_id = tt::CBIndex::c_5;

    constexpr uint32_t in0_cb_id = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t bias_cb_id = tt::CBIndex::c_2;
    constexpr uint32_t out_cb_id = tt::CBIndex::c_4;
    constexpr uint32_t mm_partials_cb_id = tt::CBIndex::c_5;
    constexpr uint32_t mm_partials2_cb_id = tt::CBIndex::c_6;

    // Profiling scope for the matmul computation
    {
        DeviceZoneScopedN("router_compute_sigmoid");

        // Call the templated router compute function
        // router_compute_sigmoid<
        //     in0_cb_id,
        //     in1_cb_id,
        //     out_cb_id,
        //     mm_partials_cb_id,
        //     in0_block_w,
        //     in0_block_num_tiles,
        //     in1_block_num_tiles,
        //     in1_block_w,
        //     out_subblock_h,
        //     out_subblock_w,
        //     out_subblock_num_tiles,
        //     untilize_out>();
        router_compute_sigmoid_bias<
            in0_cb_id,
            in1_cb_id,
            bias_cb_id,
            out_cb_id,
            mm_partials_cb_id,
            mm_partials2_cb_id,
            in0_block_w,
            in0_block_num_tiles,
            in1_block_num_tiles,
            in1_block_w,
            out_subblock_h,
            out_subblock_w,
            out_subblock_num_tiles,
            1,
            untilize_out>();
    }
}
}  // namespace NAMESPACE
