// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// wq_b_linear worker compute, helper-only alternative to the raw DEST-window kernel (perf experiment
// compute_dest_window): kernel_lib matmul_block with packer L1 acc straight into cb_out_block
// (last_block_target = Interm with interm_buf = out_buf, TileRowMajor, caller_owns_pack_target). The helper
// then has no per-K-block interm push + drain wait (the pack -> unpack sync that costs ~3.2 us in the op's
// current call) and no last-block reload. Cost: one push of the whole block at the end (no per-subblock
// store overlap); the writer must use the TileRowMajor path (OUT_SUBBLOCK_MAJOR = False).

#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    constexpr uint32_t cb_x_resident = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w_block = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out_block = get_compile_time_arg_val(3);
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(4);
    constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(5);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(6);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(7);
    constexpr uint32_t block_k_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(9);
    constexpr uint32_t out_block_tiles = in0_num_subblocks * in1_num_subblocks * out_subblock_h * out_subblock_w;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x_resident, cb_w_block, cb_out_block);

    CircularBuffer x_buf(cb_x_resident), w_buf(cb_w_block), out_buf(cb_out_block);
    {
        MaybeDeviceZoneScope("compute_wait_x0");
        cb_wait_front(cb_x_resident, in0_num_subblocks * out_subblock_h * block_k_tiles);
        cb_wait_front(cb_w_block, block_k_tiles * in1_num_subblocks * out_subblock_w);
    }
    MaybeDeviceZoneScope("compute_matmul");
    out_buf.reserve_back(out_block_tiles);
    using namespace compute_kernel_lib;
    matmul_block<
        false,
        true,
        LastBlockTarget::Interm,
        OutputCBLayout::TileRowMajor,
        matmul_config::InitMode::Short,
        InputPolicy::WaitAndPopPerKBlock,
        InputPolicy::WaitAndPopPerKBlock,
        NoPostCompute,
        NoPreKBlock,
        NoPostKBlock,
        0,
        NoKBlockInnerDimFn,
        NoIn0Source,
        NoIn1BaseOffset,
        true /*caller_owns_pack_target*/>(
        x_buf,
        w_buf,
        out_buf,
        out_buf,
        MatmulBlockShape::of(
            in0_num_subblocks, in1_num_subblocks, out_subblock_h, out_subblock_w, block_k_tiles, num_k_blocks));
    out_buf.push_back(out_block_tiles);
}
