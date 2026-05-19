// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

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
    constexpr compute_kernel_lib::OutputLayout output_layout =
#ifdef TILE_PACK_ROW_MAJOR
        compute_kernel_lib::OutputLayout::RowMajor;
#else
        compute_kernel_lib::OutputLayout::SubblockMajor;
#endif

    // Precondition for the matmul_block helper's default init_mode=ReconfigAndShort:
    // compute_kernel_hw_startup runs once at boot to program pack_init / pack_dest_init /
    // math_pack_sync_init and the initial hw_configure. The helper then re-establishes
    // data formats + matmul-mode MOPs per call without re-issuing _hw_configure MMIO.
    compute_kernel_hw_startup(cb_in0, cb_in1, cb_out);

    // Hand the batch loop to the helper: the helper's batch loop runs init once at
    // entry (init_mode=ReconfigAndShort default) and reuses LLK state across batches.
    // Per-batch re-init was found to corrupt state on heterogeneous-tile-shape DRAM-
    // sharded configs — see commit 76e99730d2e for the analogous fix in
    // bmm_large_block_zm_fused_bias_activation.cpp, which preserves a kernel-side
    // batch loop because it interleaves bias-add / untilize phases per batch. This
    // simpler bmm has no per-batch phase work, so helper-batched is the right shape.
    compute_kernel_lib::matmul_block<
        /*transpose=*/false,
        /*packer_l1_acc=*/false,
        compute_kernel_lib::LastBlockTarget::Out,
        output_layout>(
        in0_buf,
        in1_buf,
        out_buf,
        interm_buf,
        compute_kernel_lib::MatmulBlockShape::of(
            in0_num_subblocks, in1_num_subblocks, out_subblock_h, out_subblock_w, in0_block_w, num_k_blocks, batch));
}
