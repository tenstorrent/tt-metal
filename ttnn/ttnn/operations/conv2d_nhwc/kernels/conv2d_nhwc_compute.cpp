// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// conv2d_nhwc compute — regime R0.
//
// Per (m-block, n-block):
//   conv_matmul_block : matmul_block (packer-L1 accumulation into cb_partials, TileRowMajor,
//                       caller-owned pack target), with tilize_kblock (cb_act_rm -> cb_act_tiles,
//                       bm x bk) as its PreKBlockFn.
//   bias_block        : add_bias_bcast_rows in place (cb_partials -> cb_biased alias).
//   untilize_store    : untilize<bn> (bm tile-rows) into cb_out_rm for the writer.
//
// Raw API: PACK((llk_pack_reconfig_l1_acc(0))) before each tilize and after the matmul — the
// matmul helper leaves packer L1 accumulation on after K-block >= 1 (op_design.md Key Risks).

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp"

constexpr uint32_t cb_act_rm = get_compile_time_arg_val(0);
constexpr uint32_t cb_weight_tiles = get_compile_time_arg_val(1);
constexpr uint32_t cb_bias_tiles = get_compile_time_arg_val(2);
constexpr uint32_t cb_act_tiles = get_compile_time_arg_val(3);
constexpr uint32_t cb_partials = get_compile_time_arg_val(4);
constexpr uint32_t cb_biased = get_compile_time_arg_val(5);
constexpr uint32_t cb_out_rm = get_compile_time_arg_val(6);
constexpr uint32_t block_m_tiles = get_compile_time_arg_val(7);
constexpr uint32_t block_k_tiles = get_compile_time_arg_val(8);
constexpr uint32_t block_n_tiles = get_compile_time_arg_val(9);
constexpr uint32_t num_k_blocks = get_compile_time_arg_val(10);
constexpr uint32_t num_n_blocks = get_compile_time_arg_val(11);
constexpr uint32_t out_subblock_h = get_compile_time_arg_val(12);
constexpr uint32_t out_subblock_w = get_compile_time_arg_val(13);
constexpr bool has_bias = get_compile_time_arg_val(14) != 0;
constexpr uint32_t core_n_tiles = get_compile_time_arg_val(15);

constexpr uint32_t in0_num_subblocks = block_m_tiles / out_subblock_h;
constexpr uint32_t in1_num_subblocks = block_n_tiles / out_subblock_w;
constexpr uint32_t block_out_tiles = block_m_tiles * block_n_tiles;

using namespace compute_kernel_lib;

// tilize_kblock: bm tile-rows x bk tiles of RM im2col -> in0 tiles of this K-block.
struct TilizeKBlock {
    ALWI void operator()(uint32_t, uint32_t, bool) const {
        PACK((llk_pack_reconfig_l1_acc(0)));
        tilize<block_k_tiles, cb_act_rm, cb_act_tiles>(block_m_tiles);
    }
};

template <bool with_bias>
ALWI void bias_and_untilize(CircularBuffer& partials, CircularBuffer& bias, CircularBuffer& biased, uint32_t n_block) {
    if constexpr (with_bias) {
        add_bias_bcast_rows<BiasBroadcast::RowBroadcast, OutputCBLayout::TileRowMajor>(
            partials,
            bias,
            biased,
            BiasAddShape::of(in0_num_subblocks, in1_num_subblocks, out_subblock_h, out_subblock_w, block_n_tiles),
            {},
            n_block * block_n_tiles);
        untilize<block_n_tiles, cb_biased, cb_out_rm>(block_m_tiles);
    } else {
        untilize<block_n_tiles, cb_partials, cb_out_rm>(block_m_tiles);
    }
}

void kernel_main() {
    const uint32_t num_m_blocks = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_act_tiles, cb_weight_tiles, cb_partials);

    CircularBuffer act_tiles(cb_act_tiles);
    CircularBuffer weight_tiles(cb_weight_tiles);
    CircularBuffer partials(cb_partials);
    CircularBuffer bias(cb_bias_tiles);
    CircularBuffer biased(cb_biased);

    // load_bias_slice (consumer side): fronted for the kernel's life.
    if constexpr (has_bias) {
        bias.wait_front(core_n_tiles);
    }

    const auto shape = MatmulBlockShape::of(
        in0_num_subblocks, in1_num_subblocks, out_subblock_h, out_subblock_w, block_k_tiles, num_k_blocks);

    for (uint32_t m_block = 0; m_block < num_m_blocks; ++m_block) {
        for (uint32_t n_block = 0; n_block < num_n_blocks; ++n_block) {
            // conv_matmul_block: caller owns the accumulator's single reserve / push.
            partials.reserve_back(block_out_tiles);
            matmul_block<
                false,
                true,
                LastBlockTarget::Interm,
                OutputCBLayout::TileRowMajor,
                matmul_config::InitMode::ShortAfterPreKBlock,
                InputPolicy::WaitAndPopPerKBlock,
                InputPolicy::WaitAndPopPerKBlock,
                NoPostCompute,
                TilizeKBlock,
                NoPostKBlock,
                0,
                NoKBlockInnerDimFn,
                NoIn0Source,
                NoIn1BaseOffset,
                true>(act_tiles, weight_tiles, partials, partials, shape, {}, {}, block_n_tiles, block_n_tiles);
            partials.push_back(block_out_tiles);
            PACK((llk_pack_reconfig_l1_acc(0)));

            bias_and_untilize<has_bias>(partials, bias, biased, n_block);
        }
    }

    if constexpr (has_bias) {
        bias.pop_front(core_n_tiles);
    }
}
