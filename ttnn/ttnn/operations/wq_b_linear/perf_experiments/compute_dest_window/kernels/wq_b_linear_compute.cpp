// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// wq_b_linear worker compute: one matmul_block over num_k_blocks K-steps, packer L1
// accumulation into cb_interm_acc, last K-step packed into cb_out_block (TileRowMajor or per subblock).

#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

// Zones (permanent): compute_wait_x0 = unpack waiting for x round 0 + W block 0 (hoisted out of the helper so
// pipeline fill is not charged to the matmul); compute_matmul = the whole matmul_block (its later per-K-step waits
// included: occupancy on TRISC_0/2, wait+work on TRISC_1). Ablation: SKIP_COMPUTE (helper switch).

void kernel_main() {
    constexpr uint32_t cb_x_resident = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w_block = get_compile_time_arg_val(1);
    constexpr uint32_t cb_interm_acc = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out_block = get_compile_time_arg_val(3);
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(4);
    constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(5);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(6);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(7);
    constexpr uint32_t block_k_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(9);
    // out_subblock_major: last K-step pushes cb_out_block per subblock (writer stores each as it lands,
    // overlapping the store with the rest of the last K-step); else one TileRowMajor push per row group.
    constexpr bool out_subblock_major = get_compile_time_arg_val(10) != 0;
    constexpr auto out_layout = out_subblock_major ? compute_kernel_lib::OutputCBLayout::SubblockMajor
                                                   : compute_kernel_lib::OutputCBLayout::TileRowMajor;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x_resident, cb_w_block, cb_out_block);

    CircularBuffer x_buf(cb_x_resident), w_buf(cb_w_block), out_buf(cb_out_block), interm_buf(cb_interm_acc);
    {
        MaybeDeviceZoneScope("compute_wait_x0");
        cb_wait_front(cb_x_resident, in0_num_subblocks * out_subblock_h * block_k_tiles);  // x round 0
        cb_wait_front(cb_w_block, block_k_tiles * in1_num_subblocks * out_subblock_w);     // W block 0
    }
    MaybeDeviceZoneScope("compute_matmul");
    compute_kernel_lib::
        matmul_block<false /*transpose*/, true /*packer_l1_acc*/, compute_kernel_lib::LastBlockTarget::Out, out_layout>(
            x_buf,
            w_buf,
            out_buf,
            interm_buf,
            compute_kernel_lib::MatmulBlockShape::of(
                in0_num_subblocks, in1_num_subblocks, out_subblock_h, out_subblock_w, block_k_tiles, num_k_blocks));
}
