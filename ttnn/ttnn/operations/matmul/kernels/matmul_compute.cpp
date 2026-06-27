// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// matmul compute (TRISC). One matmul_block per output block; the helper loops
// the K-blocks internally (waits/pops one K-block of in0/in1 per iteration,
// spills/reloads via cb_interm, packs the final SubblockMajor block to cb_out).
// All output blocks are identical work, so they are just called in lock-step
// with the reader/writer.

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"

using namespace compute_kernel_lib;

constexpr uint32_t CB_IN0_ACT = 0;
constexpr uint32_t CB_IN1_WEIGHT = 1;
constexpr uint32_t CB_OUT = 16;
constexpr uint32_t CB_INTERM = 24;

void kernel_main() {
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(0);
    constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(1);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(2);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(3);
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(4);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(5);
    constexpr uint32_t total_blocks = get_compile_time_arg_val(6);

    CircularBuffer in0_buf(CB_IN0_ACT);
    CircularBuffer in1_buf(CB_IN1_WEIGHT);
    CircularBuffer out_buf(CB_OUT);
    CircularBuffer interm_buf(CB_INTERM);

    // Boot once (hw_configure-bearing init is the caller's one-time job).
    compute_kernel_hw_startup(CB_IN0_ACT, CB_IN1_WEIGHT, CB_OUT);
    mm_block_init(CB_IN0_ACT, CB_IN1_WEIGHT, CB_OUT, 0 /*transpose*/, out_subblock_w, out_subblock_h, in0_block_w);

    const auto shape = MatmulBlockShape::of(
        in0_num_subblocks,
        in1_num_subblocks,
        out_subblock_h,
        out_subblock_w,
        in0_block_w,
        num_k_blocks,
        1 /*batch — outer loop is here*/);

    for (uint32_t i = 0; i < total_blocks; i++) {
        matmul_block<>(in0_buf, in1_buf, out_buf, interm_buf, shape);
    }
}
