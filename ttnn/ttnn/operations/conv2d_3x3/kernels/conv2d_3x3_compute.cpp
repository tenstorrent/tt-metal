// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// conv2d_3x3 compute: per M-block, tilize_matmul_block (matmul_block with tilize as its PreKBlockFn,
// fp32 packer-L1 accumulation over the K walk) then finalize_block (row-broadcast bias add).
// No bias: the last K-block packs straight to the output CB and finalize_block is empty.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp"

namespace ckl = compute_kernel_lib;

namespace {

// Tilizes one K-block (block_m_tiles tile-rows x block_k_tiles) of gathered im2col rows. The matmul
// restores its own state afterwards (InitMode::ShortAfterPreKBlock).
template <uint32_t block_k_tiles, uint32_t block_m_tiles, uint32_t cb_im2col_rm, uint32_t cb_im2col_tiles>
struct TilizeIm2colKBlock {
    DataflowBuffer* rm;
    DataflowBuffer* tiles;

    ALWI void operator()(uint32_t /*block*/, uint32_t /*num_k_blocks*/, bool /*is_last*/) const {
        // The previous K-block's pack left packer L1 accumulation on; the tilize pack must overwrite.
        pack_reconfig_l1_acc(0);
        ckl::tilize<block_k_tiles, cb_im2col_rm, cb_im2col_tiles>(*rm, *tiles, block_m_tiles);
    }
};

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_im2col_rm = get_compile_time_arg_val(0);
    constexpr uint32_t cb_im2col_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t cb_weights = get_compile_time_arg_val(2);
    constexpr uint32_t cb_bias = get_compile_time_arg_val(3);
    constexpr uint32_t cb_matmul_partials = get_compile_time_arg_val(4);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t block_m_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t block_k_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t block_n_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(9);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(10);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(11);
    constexpr bool has_bias = get_compile_time_arg_val(12) != 0;

    const uint32_t num_m_blocks_this_core = get_arg_val<uint32_t>(0);

    constexpr uint32_t in0_num_subblocks = block_m_tiles / out_subblock_h;
    constexpr uint32_t in1_num_subblocks = block_n_tiles / out_subblock_w;
    static_assert(in0_num_subblocks * out_subblock_h == block_m_tiles);
    static_assert(in1_num_subblocks * out_subblock_w == block_n_tiles);
    constexpr auto mm_shape = ckl::MatmulBlockShape::of(
        in0_num_subblocks, in1_num_subblocks, out_subblock_h, out_subblock_w, block_k_tiles, num_k_blocks);
    constexpr auto bias_shape =
        ckl::BiasAddShape::of(in0_num_subblocks, in1_num_subblocks, out_subblock_h, out_subblock_w, block_n_tiles);
    constexpr ckl::LastBlockTarget last_block_target =
        has_bias ? ckl::LastBlockTarget::Interm : ckl::LastBlockTarget::Out;

    // The first op is the tilize, so startup matches its operands; every later phase reconfigures itself.
    compute_kernel_hw_startup(cb_im2col_rm, cb_im2col_tiles);

    DataflowBuffer im2col_rm_dfb(cb_im2col_rm);
    DataflowBuffer im2col_tiles_dfb(cb_im2col_tiles);
    CircularBuffer im2col_tiles(cb_im2col_tiles);
    CircularBuffer weights(cb_weights);
    CircularBuffer partials(cb_matmul_partials);
    CircularBuffer output_tiles(cb_output_tiles);
    CircularBuffer bias(cb_bias);

    using TilizeFn = TilizeIm2colKBlock<block_k_tiles, block_m_tiles, cb_im2col_rm, cb_im2col_tiles>;
    const TilizeFn tilize_k_block{&im2col_rm_dfb, &im2col_tiles_dfb};

    if constexpr (has_bias) {
        bias.wait_front(block_n_tiles);
    }

    for (uint32_t m_block_idx = 0; m_block_idx < num_m_blocks_this_core; ++m_block_idx) {
        ckl::matmul_block<
            /*transpose=*/false,
            /*packer_l1_acc=*/true,
            last_block_target,
            ckl::OutputCBLayout::TileRowMajor,
            ckl::matmul_config::InitMode::ShortAfterPreKBlock,
            ckl::InputPolicy::WaitAndPopPerKBlock,
            ckl::InputPolicy::WaitAndPopPerKBlock,
            ckl::NoPostCompute,
            TilizeFn>(
            im2col_tiles,
            weights,
            output_tiles,
            partials,
            mm_shape,
            {},
            tilize_k_block,
            /*in1_per_core_w=*/block_n_tiles,
            /*out_row_width=*/block_n_tiles);

        if constexpr (has_bias) {
            // matmul_block leaves packer L1 accumulation on after its last K-block.
            pack_reconfig_l1_acc(0);
            ckl::add_bias_bcast_rows<ckl::BiasBroadcast::RowBroadcast, ckl::OutputCBLayout::TileRowMajor>(
                partials, bias, output_tiles, bias_shape);
        }
    }

    if constexpr (has_bias) {
        bias.pop_front(block_n_tiles);
    }
}
