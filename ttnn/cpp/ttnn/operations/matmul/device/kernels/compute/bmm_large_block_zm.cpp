// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul/matmul.hpp"

void kernel_main() {
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);        // inner block size in tiles
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(1);  // outer row block size (in inner row blocks)
    constexpr uint32_t in1_num_subblocks =
        get_compile_time_arg_val(4);                              // outer column block size (in inner column blocks)
    constexpr uint32_t num_blocks = get_compile_time_arg_val(7);  // outer inner dim (in inner dim blocks)
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(8);  // inner row block size in tiles
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(9);  // inner column block size in tiles
    constexpr uint32_t batch = get_compile_time_arg_val(11);          // batch dim

    constexpr uint32_t dfb_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t dfb_in1 = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t dfb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t dfb_intermed0 = get_named_compile_time_arg_val("cb_intermed0");

    // Initialize once and retain matmul state across batches; reinitialization can corrupt
    // heterogeneous-tile DRAM-sharded configurations.
    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb_in0, dfb_in1, dfb_intermed0);
    matmul_block_init(dfb_in0, dfb_in1, /*transpose_in1=*/false, out_subblock_w, out_subblock_h, in0_block_w);

    compute_kernel_lib::matmul<
        /*transpose_in1=*/false,
        /*packer_l1_acc=*/false,
        compute_kernel_lib::matmul_config::InitMode::AssumeInitialized,
        compute_kernel_lib::matmul_config::InputPolicy::WaitAndPopPerKBlock,
        compute_kernel_lib::matmul_config::DataFormatReconfig::None>(
        dfb_in0,
        dfb_in1,
        dfb_out,
        dfb_intermed0,
        compute_kernel_lib::StaticMatmulShape<
            in0_num_subblocks,
            in1_num_subblocks,
            out_subblock_h,
            out_subblock_w,
            in0_block_w,
            num_blocks,
            batch>{});
}
