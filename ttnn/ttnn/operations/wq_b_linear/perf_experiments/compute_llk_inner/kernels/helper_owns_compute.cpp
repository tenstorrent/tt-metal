// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// compute_llk_inner option "helper_owns": the same kernel_lib matmul_block, configured so that every K-step
// packs with L1 accumulation straight into cb_out_block (caller-owned pack target), instead of
// LastBlockTarget::Out's spill-to-interm + per-K-step drain + last-K-step reload.
// Why it is faster: on the Out path the helper drains cb_interm_acc on the UNPACK thread after every K-step
// (wait_front/pop_front of the spilled partials), so unpack of K-step k+1 cannot start until the packer
// finished K-step k: the pack latency of the last subblock is exposed 12 times. caller_owns_pack_target
// removes the drain (and the reload), so unpack runs ahead of pack as usual.
// Requires OUT_SUBBLOCK_MAJOR = False in the descriptor (TileRowMajor output, one push of the whole block).

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
    constexpr uint32_t out_block_tiles = in0_num_subblocks * out_subblock_h * in1_num_subblocks * out_subblock_w;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x_resident, cb_w_block, cb_out_block);

    CircularBuffer x_buf(cb_x_resident), w_buf(cb_w_block), out_buf(cb_out_block);
    {
        MaybeDeviceZoneScope("compute_wait_x0");
        cb_wait_front(cb_x_resident, in0_num_subblocks * out_subblock_h * block_k_tiles);
        cb_wait_front(cb_w_block, block_k_tiles * in1_num_subblocks * out_subblock_w);
    }
    MaybeDeviceZoneScope("compute_matmul");
    out_buf.reserve_back(out_block_tiles);
    compute_kernel_lib::matmul_block<
        false /*transpose*/,
        true /*packer_l1_acc*/,
        compute_kernel_lib::LastBlockTarget::Interm,
        compute_kernel_lib::OutputCBLayout::TileRowMajor,
        compute_kernel_lib::matmul_config::InitMode::Short,
        compute_kernel_lib::InputPolicy::WaitAndPopPerKBlock,
        compute_kernel_lib::InputPolicy::WaitAndPopPerKBlock,
        compute_kernel_lib::NoPostCompute,
        compute_kernel_lib::NoPreKBlock,
        compute_kernel_lib::NoPostKBlock,
        0,
        compute_kernel_lib::NoKBlockInnerDimFn,
        compute_kernel_lib::NoIn0Source,
        compute_kernel_lib::NoIn1BaseOffset,
        true /*caller_owns_pack_target*/>(
        x_buf,
        w_buf,
        out_buf,
        out_buf /* pack target = the output CB itself */,
        compute_kernel_lib::MatmulBlockShape::of(
            in0_num_subblocks, in1_num_subblocks, out_subblock_h, out_subblock_w, block_k_tiles, num_k_blocks));
    out_buf.push_back(out_block_tiles);
    PACK((llk_pack_reconfig_l1_acc(0)));
}
