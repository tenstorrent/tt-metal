// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
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

    CircularBuffer in0_buf(cb_in0);
    CircularBuffer in1_buf(cb_in1);
    CircularBuffer out_buf(cb_out);
    CircularBuffer interm_buf(cb_intermed0);

    // Factories that emit TILE_PACK_ROW_MAJOR want absolute-offset packing so writers
    // read tiles in row-major order. Multicast factories (no define) use sequential pack.
    constexpr compute_kernel_lib::OutputCBLayout output_layout =
#ifdef TILE_PACK_ROW_MAJOR
        compute_kernel_lib::OutputCBLayout::TileRowMajor;
#else
        compute_kernel_lib::OutputCBLayout::SubblockMajor;
#endif

    // Boot-time matmul init. mm_block_init/mm_block_init_short are deprecated: boot with
    // compute_kernel_hw_startup (the single hw_configure MMIO, first compute API call at
    // kernel start) then matmul_block_init (unpack/math matmul init). The helper is then
    // invoked with InitMode::None so it reuses LLK state across the batch loop. Per-batch
    // re-init was found to corrupt state on heterogeneous-tile-shape DRAM-sharded configs —
    // see commit 76e99730d2e for the analogous fix in
    // bmm_large_block_zm_fused_bias_activation.cpp, which preserves a kernel-side batch loop
    // because it interleaves bias-add / untilize phases per batch. This simpler bmm has no
    // per-batch phase work, so helper-batched is the right shape.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_intermed0);
    matmul_block_init(cb_in0, cb_in1, /*transpose=*/false, out_subblock_w, out_subblock_h, in0_block_w);

    compute_kernel_lib::matmul_block<
        /*transpose=*/false,
        /*packer_l1_acc=*/false,
        compute_kernel_lib::LastBlockTarget::Out,
        output_layout,
        compute_kernel_lib::matmul_config::InitMode::None>(
        in0_buf,
        in1_buf,
        out_buf,
        interm_buf,
        compute_kernel_lib::MatmulBlockShape::of(
            in0_num_subblocks, in1_num_subblocks, out_subblock_h, out_subblock_w, in0_block_w, num_k_blocks, batch));
}
