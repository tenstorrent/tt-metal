// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// genify_matmul compute — block operation `matmul_accumulate_block` of regime R1 (epilogue none).
//
// One compute_kernel_lib::matmul_block call per block of this core's region. The helper walks the
// num_k_blocks K-blocks (waiting / popping cb_act_tiles and cb_weight_tiles per K-block),
// accumulates the partial sums of the whole block in cb_partials with packer L1 accumulation
// (TileRowMajor spill, exact one-block capacity), reloads them on the last K-block and packs the
// result row group by row group into cb_output_tiles. With a single K-block there is no
// accumulator: cb_output_tiles is passed as the helper's same-format interm placeholder.
// The ragged last K-block is handled by LastKBlockInnerDim (only last_block_k_tiles FMA steps).

#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"

namespace ckl = compute_kernel_lib;

constexpr uint32_t cb_act_tiles = get_compile_time_arg_val(0);
constexpr uint32_t cb_weight_tiles = get_compile_time_arg_val(1);
constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(2);
constexpr uint32_t cb_partials = get_compile_time_arg_val(3);
constexpr uint32_t block_batches = get_compile_time_arg_val(4);
constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(5);
constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(6);
constexpr uint32_t sb_h = get_compile_time_arg_val(7);
constexpr uint32_t sb_w = get_compile_time_arg_val(8);
constexpr uint32_t block_k_tiles = get_compile_time_arg_val(9);
constexpr uint32_t num_k_blocks = get_compile_time_arg_val(10);
constexpr uint32_t last_block_k_tiles = get_compile_time_arg_val(11);
constexpr uint32_t block_n_tiles = get_compile_time_arg_val(12);

// FMA step count per K-block: the balanced split leaves the last K-block narrower; the reader
// pushes the nominal block_k pages, so the garbage tail must never be summed.
struct LastKBlockInnerDim {
    ALWI uint32_t operator()(uint32_t block, uint32_t block_k) const {
        return (block == num_k_blocks - 1) ? last_block_k_tiles : block_k;
    }
};

// No accumulator CB when there is a single K-block: the helper only needs a same-format placeholder.
constexpr uint32_t cb_interm = (num_k_blocks > 1) ? cb_partials : cb_output_tiles;

void kernel_main() {
    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    // matmul maps in0 -> SrcB, in1 -> SrcA: Reverse is mandatory.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_act_tiles, cb_weight_tiles, cb_output_tiles);

    CircularBuffer act_buf(cb_act_tiles), weight_buf(cb_weight_tiles), out_buf(cb_output_tiles), interm_buf(cb_interm);

    const auto shape = ckl::MatmulBlockShape::of(
        in0_num_subblocks, in1_num_subblocks, sb_h, sb_w, block_k_tiles, num_k_blocks, block_batches);

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        ckl::matmul_block<
            /*transpose=*/false,
            /*packer_l1_acc=*/true,
            ckl::LastBlockTarget::Out,
            ckl::OutputCBLayout::TileRowMajor,
            ckl::matmul_config::InitMode::Short,
            ckl::InputPolicy::WaitAndPopPerKBlock,
            ckl::InputPolicy::WaitAndPopPerKBlock,
            ckl::NoPostCompute,
            ckl::NoPreKBlock,
            ckl::NoPostKBlock,
            /*untilize_block_ct_dim=*/0,
            LastKBlockInnerDim>(
            act_buf,
            weight_buf,
            out_buf,
            interm_buf,
            shape,
            ckl::NoPostCompute{},
            ckl::NoPreKBlock{},
            /*in1_per_core_w=*/block_n_tiles,
            /*out_row_width=*/block_n_tiles,
            ckl::NoPostKBlock{},
            LastKBlockInnerDim{});
    }
}
