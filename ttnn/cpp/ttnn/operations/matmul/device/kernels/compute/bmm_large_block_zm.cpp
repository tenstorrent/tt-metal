// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"

void kernel_main() {
    uint32_t in0_block_w = get_compile_time_arg_val(0);
    uint32_t in0_num_subblocks = get_compile_time_arg_val(1);
    uint32_t in1_num_subblocks = get_compile_time_arg_val(4);
    uint32_t num_k_blocks = get_compile_time_arg_val(7);
    uint32_t out_subblock_h = get_compile_time_arg_val(8);
    uint32_t out_subblock_w = get_compile_time_arg_val(9);
    uint32_t batch = get_compile_time_arg_val(11);

    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t cb_in1 = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t cb_intermed0 = get_named_compile_time_arg_val("cb_intermed0");

    // Factories that emit ROW_MAJOR_OUTPUT want absolute-offset packing so writers
    // read tiles in row-major order. Multicast factories (no define) use sequential pack.
    constexpr compute_kernel_lib::OutputLayout output_layout =
#ifdef ROW_MAJOR_OUTPUT
        compute_kernel_lib::OutputLayout::RowMajor;
#else
        compute_kernel_lib::OutputLayout::SubblockMajor;
#endif

    mm_block_init(cb_in0, cb_in1, cb_intermed0, false, out_subblock_w, out_subblock_h, in0_block_w);

    for (uint32_t b = 0; b < batch; b++) {
        compute_kernel_lib::matmul_block<
            /*transpose=*/false,
            /*packer_l1_acc=*/false,
            /*pack_last_to_interm=*/false,
            /*pack_relu=*/false,
            output_layout>(
            cb_in0,
            cb_in1,
            cb_out,
            cb_intermed0,
            compute_kernel_lib::MatmulBlockShape::of(
                in0_num_subblocks,
                in1_num_subblocks,
                out_subblock_h,
                out_subblock_w,
                in0_block_w,
                num_k_blocks,
                /*batch=*/1));
    }
}
