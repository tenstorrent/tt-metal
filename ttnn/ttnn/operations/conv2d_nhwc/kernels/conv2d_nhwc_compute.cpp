// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// conv2d_nhwc compute — regimes R0 and R4/R5.
//
// Per (m-block, n-block):
//   conv_matmul_block : matmul_block (packer-L1 accumulation into cb_partials, TileRowMajor,
//                       caller-owned pack target), with tilize_kblock as its PreKBlockFn:
//                       R0    — every K-block: cb_act_rm -> cb_act_tiles (bm x bk), the in0.
//                       R4/R5 — only this core's own channel-slice K-blocks: cb_act_rm ->
//                               cb_act_send, one K-block AHEAD of the matmul (so the owner's
//                               tilize overlaps the row's matmul of the previous block); the
//                               writer multicasts cb_act_send into every row core's cb_act_tiles.
//   bias_block        : add_bias_bcast_rows in place (cb_partials -> cb_biased alias).
//   (relu)            : pack_relu_config(zero) around the final output pack, none() after.
//   untilize_store    : untilize<bn> (bm tile-rows) into cb_out_rm for the writer (R0), or
//                       straight into the output shard (R4/R5, zero-copy); a WIDTH shard's
//                       partial last tile-row goes to the cb_out_rm tail for the writer.
//
// R11 depthwise (no matmul): per (m-block, n-block), per tile-row: tilize taps x bn tiles, then
//   dw_tap_accumulate (eltwise_chain: Mul with row-broadcast tap weights, WholeShape DEST
//   accumulation over the taps) into cb_partials; then the same bias_and_untilize.
//
// Raw API: PACK((llk_pack_reconfig_l1_acc(0))) before each tilize and after the matmul — the
// matmul helper leaves packer L1 accumulation on after K-block >= 1 (op_design.md Key Risks).

#include <cstdint>
#include <type_traits>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"

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
constexpr bool k_exchange = get_compile_time_arg_val(16) != 0;
constexpr uint32_t cb_act_send = get_compile_time_arg_val(17);
constexpr uint32_t kblocks_per_slice = get_compile_time_arg_val(18);
constexpr uint32_t cb_out_shard = get_compile_time_arg_val(19);
constexpr bool out_tail = get_compile_time_arg_val(20) != 0;
constexpr bool act_mcast = get_compile_time_arg_val(21) != 0;  // R4/R5 with > 1 channel slice
// R4/R5: cb_act_send depth in own K-blocks. The owner tilizes own block kb right after own block
// kb - act_send_depth was multicast; >= kblocks_per_slice tilizes the whole own slice up front.
constexpr uint32_t act_send_depth = get_compile_time_arg_val(22);
// Fused ReLU (technique 4): packer ReLU on the final output pack only — the bias pack, or the
// untilize pack when there is no bias — and cleared right after, so the next block's tilize
// and partial (L1-accumulated) packs never see it.
constexpr bool fuse_relu = get_compile_time_arg_val(23) != 0;
// R11 depthwise: per tile-row, cb_act_tiles holds taps x bn tilized tiles ([tap][ct], the reader's
// im2col order) and cb_weight_tiles taps x bn row-0 weight tiles for the n-block; output tile ct is
// sum_tap act[tap][ct] * row_bcast(w[tap][ct]), accumulated in DEST (no matmul).
constexpr bool depthwise = get_compile_time_arg_val(24) != 0;
constexpr uint32_t taps = get_compile_time_arg_val(25);
// Dense tilize width in tiles: bk ([pixel][K-block] RM rows), or one tap (tap-major K-blocks: the
// reader lays each tap's 32 pixels out as a contiguous 32 x tap_width block). Either way the tiles
// come out in K order, tile-row by tile-row.
constexpr uint32_t tilize_w = get_compile_time_arg_val(26);
static_assert(block_k_tiles % tilize_w == 0, "a K-block is whole tilize blocks");
// R3 (HEIGHT): all num_k_blocks weight K-blocks are resident in cb_weight_tiles (the writer pushes
// them once); the matmul reads K-block j at tile offset j * bk * bn (in1 NoWaitNoPop + base offset).
constexpr bool weights_resident = get_compile_time_arg_val(27) != 0;
constexpr uint32_t weight_kblock_tiles = block_k_tiles * block_n_tiles;
constexpr uint32_t resident_weight_tiles = num_k_blocks * weight_kblock_tiles;
// Split gather (R3): odd m-blocks' RM im2col arrives in cb_act_rm2 (gathered by the writer RISC).
constexpr uint32_t cb_act_rm2 = get_compile_time_arg_val(28);
constexpr uint32_t num_gather_roles = get_compile_time_arg_val(29);
constexpr uint32_t tilize_blocks_per_row = block_k_tiles / tilize_w;

constexpr uint32_t in0_num_subblocks = block_m_tiles / out_subblock_h;
constexpr uint32_t in1_num_subblocks = block_n_tiles / out_subblock_w;
constexpr uint32_t block_out_tiles = block_m_tiles * block_n_tiles;
// R4/R5 K order interleaves the slices: K-block j belongs to slice j % num_slices and is that
// slice's own K-block j / num_slices (the writer maps weight rows the same way).
constexpr uint32_t num_slices = num_k_blocks / kblocks_per_slice;

using namespace compute_kernel_lib;

// tilize_kblock: bm tile-rows x bk tiles of RM im2col -> tiles.
//   R0:    this K-block's in0 (cb_act_tiles).
//   R4/R5 (act_mcast): own blocks are j = kb * num_slices + my_slice. Own blocks [0, depth) are
//          tilized into cb_act_send at block 0; own block kb + depth right after own block kb was
//          multicast (block kb * num_slices + my_slice + 1), so the owner's tilize (and the
//          reader's gather feeding it) overlaps the other slices' turns. One slice: as R0.
struct TilizeKBlock {
    uint32_t my_slice;
    uint32_t m_block = 0;
    ALWI void tilize_own() const {
        PACK((llk_pack_reconfig_l1_acc(0)));
        tilize<tilize_w, cb_act_rm, cb_act_send>(block_m_tiles * tilize_blocks_per_row);
    }
    ALWI void operator()(uint32_t block, uint32_t, bool) const {
        tilize_kblock(block);
        if constexpr (weights_resident) {
            // in1 is NoWaitNoPop: K-blocks 0..block of the resident weights must have landed (the
            // writer pushes them chunk by chunk; idempotent once all are in).
            cb_wait_front(cb_weight_tiles, (block + 1) * weight_kblock_tiles);
        }
    }
    ALWI void tilize_kblock(uint32_t block) const {
        if constexpr (!act_mcast) {
#ifdef ABL_TILIZE
            for (uint32_t t = 0; t < block_m_tiles; ++t) {
                cb_wait_front(cb_act_rm, block_k_tiles);
                cb_pop_front(cb_act_rm, block_k_tiles);
            }
            cb_reserve_back(cb_act_tiles, block_m_tiles * block_k_tiles);
            cb_push_back(cb_act_tiles, block_m_tiles * block_k_tiles);
            return;
#endif
            PACK((llk_pack_reconfig_l1_acc(0)));
            if (num_gather_roles > 1 && (m_block & 1)) {
                tilize<tilize_w, cb_act_rm2, cb_act_tiles>(block_m_tiles * tilize_blocks_per_row);
            } else {
                tilize<tilize_w, cb_act_rm, cb_act_tiles>(block_m_tiles * tilize_blocks_per_row);
            }
        } else {
            constexpr uint32_t first = act_send_depth < kblocks_per_slice ? act_send_depth : kblocks_per_slice;
            if (block == 0) {
                for (uint32_t kb = 0; kb < first; ++kb) {
                    tilize_own();
                }
            } else if (block > my_slice && (block - my_slice - 1) % num_slices == 0) {
                if ((block - my_slice - 1) / num_slices + act_send_depth < kblocks_per_slice) {
                    tilize_own();
                }
            }
        }
    }
};

// Resident weights: K-block j's in1 starts j * bk * bn tiles into the fronted region.
struct ResidentIn1Offset {
    ALWI uint32_t operator()(uint32_t block) const { return block * weight_kblock_tiles; }
};
using In1Offset = std::conditional_t<weights_resident, ResidentIn1Offset, NoIn1BaseOffset>;
constexpr InputPolicy in1_policy = weights_resident ? InputPolicy::NoWaitNoPop : InputPolicy::WaitAndPopPerKBlock;

// untilize_store: `rows` tile-rows of cb_src -> the output (shard, or cb_out_rm), with a
// WIDTH shard's partial last tile-row staged in cb_out_rm (tail) for the writer.
template <uint32_t cb_src>
ALWI void untilize_store(bool last_block) {
    if constexpr (!k_exchange) {
        untilize<block_n_tiles, cb_src, cb_out_rm>(block_m_tiles);
    } else if constexpr (out_tail) {
        if (last_block) {
            if constexpr (block_m_tiles > 1) {
                untilize<block_n_tiles, cb_src, cb_out_shard>(block_m_tiles - 1);
            }
            untilize<block_n_tiles, cb_src, cb_out_rm>(1);
        } else {
            untilize<block_n_tiles, cb_src, cb_out_shard>(block_m_tiles);
        }
    } else {
        untilize<block_n_tiles, cb_src, cb_out_shard>(block_m_tiles);
    }
}

template <bool with_bias>
ALWI void bias_and_untilize(
    CircularBuffer& partials, CircularBuffer& bias, CircularBuffer& biased, uint32_t n_block, bool last_block) {
    if constexpr (with_bias) {
        if constexpr (fuse_relu) {
            pack_relu_config(ReluConfig::zero());
        }
        add_bias_bcast_rows<BiasBroadcast::RowBroadcast, OutputCBLayout::TileRowMajor>(
            partials,
            bias,
            biased,
            BiasAddShape::of(in0_num_subblocks, in1_num_subblocks, out_subblock_h, out_subblock_w, block_n_tiles),
            {},
            n_block * block_n_tiles);
        if constexpr (fuse_relu) {
            pack_relu_config(ReluConfig::none());
        }
        untilize_store<cb_biased>(last_block);
    } else {
        if constexpr (fuse_relu) {
            pack_relu_config(ReluConfig::zero());
        }
        untilize_store<cb_partials>(last_block);
        if constexpr (fuse_relu) {
            pack_relu_config(ReluConfig::none());
        }
    }
}

// dw_tap_accumulate: one output tile-row of the n-block — for each of its bn output tiles, one
// eltwise_chain walking the taps (grid(taps, 1), Col mapping + Strided {ct, bn}: tile tap*bn + ct
// of both operands) with the products accumulated in one sticky DEST tile (WholeShape), packed
// once into cb_partials. Inputs are caller-managed: fronted for the whole tile-row / n-block.
ALWI void dw_tap_accumulate(DataflowBuffer& act, DataflowBuffer& weights, DataflowBuffer& partials) {
    using TapMul = BinaryFpu<
        BinaryFpuOp::Mul,
        input(
            cb_act_tiles,
            WaitPolicy::None,
            PopPolicy::None,
            InputTileMapping::Col,
            DataFormatReconfig::Enabled,
            TileAddressing::Strided),
        input(
            cb_weight_tiles,
            BroadcastDim::Row,
            WaitPolicy::None,
            PopPolicy::None,
            InputTileMapping::Col,
            DataFormatReconfig::Enabled,
            TileAddressing::Strided),
        Dst::D0,
        DestAccumulation::WholeShape>;
    using PackAcc = PackTile<output(
        cb_partials,
        ReservePolicy::OneUpfront,
        PushPolicy::OneAtEnd,
        DataFormatReconfig::Enabled,
        TileAddressing::Direct,
        DestAccumulation::WholeShape)>;
    for (uint32_t ct = 0; ct < block_n_tiles; ++ct) {
        const StridedTileRange range{ct, block_n_tiles};
        eltwise_chain(IterationShape::grid(taps, 1), TapMul{act, weights, range, range}, PackAcc{partials});
    }
}

void kernel_main() {
    const uint32_t num_m_blocks = get_arg_val<uint32_t>(0);
    const uint32_t my_slice = get_arg_val<uint32_t>(1);  // R4/R5: this core's channel slice

    if constexpr (depthwise) {
        compute_kernel_hw_startup(cb_act_tiles, cb_weight_tiles, cb_partials);
    } else {
        compute_kernel_hw_startup<SrcOrder::Reverse>(cb_act_tiles, cb_weight_tiles, cb_partials);
    }

    CircularBuffer act_tiles(cb_act_tiles);
    CircularBuffer weight_tiles(cb_weight_tiles);
    CircularBuffer partials(cb_partials);
    CircularBuffer bias(cb_bias_tiles);
    CircularBuffer biased(cb_biased);

    // load_bias_slice (consumer side): fronted for the kernel's life.
    if constexpr (has_bias) {
        bias.wait_front(core_n_tiles);
    }

    if constexpr (depthwise) {
        constexpr uint32_t row_tiles = taps * block_n_tiles;  // = block_k_tiles
        static_assert(!depthwise || row_tiles == block_k_tiles, "depthwise K-block = taps x bn");
        DataflowBuffer dw_act(cb_act_tiles);
        DataflowBuffer dw_weights(cb_weight_tiles);
        DataflowBuffer dw_partials(cb_partials);
        for (uint32_t m_block = 0; m_block < num_m_blocks; ++m_block) {
            for (uint32_t n_block = 0; n_block < num_n_blocks; ++n_block) {
                weight_tiles.wait_front(row_tiles);
                for (uint32_t t = 0; t < block_m_tiles; ++t) {
                    // tilize one im2col tile-row: taps x bn per-tap tiles.
                    tilize<row_tiles, cb_act_rm, cb_act_tiles>(1);
                    act_tiles.wait_front(row_tiles);
                    dw_tap_accumulate(dw_act, dw_weights, dw_partials);
                    act_tiles.pop_front(row_tiles);
                }
                weight_tiles.pop_front(row_tiles);
                bias_and_untilize<has_bias>(partials, bias, biased, n_block, m_block + 1 == num_m_blocks);
            }
        }
    } else {
        const auto shape = MatmulBlockShape::of(
            in0_num_subblocks, in1_num_subblocks, out_subblock_h, out_subblock_w, block_k_tiles, num_k_blocks);
        for (uint32_t m_block = 0; m_block < num_m_blocks; ++m_block) {
            for (uint32_t n_block = 0; n_block < num_n_blocks; ++n_block) {
                // conv_matmul_block: caller owns the accumulator's single reserve / push.
                partials.reserve_back(block_out_tiles);
#ifdef ABL_MATMUL
                for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
                    TilizeKBlock{my_slice, m_block}(kb, 0, false);
                    cb_wait_front(cb_act_tiles, block_m_tiles * block_k_tiles);
                    cb_pop_front(cb_act_tiles, block_m_tiles * block_k_tiles);
                    if constexpr (!weights_resident) {
                        cb_wait_front(cb_weight_tiles, block_k_tiles * block_n_tiles);
                        cb_pop_front(cb_weight_tiles, block_k_tiles * block_n_tiles);
                    }
                }
                if (false)
#endif
                    matmul_block<
                        false,
                        true,
                        LastBlockTarget::Interm,
                        OutputCBLayout::TileRowMajor,
                        matmul_config::InitMode::ShortAfterPreKBlock,
                        InputPolicy::WaitAndPopPerKBlock,
                        in1_policy,
                        NoPostCompute,
                        TilizeKBlock,
                        NoPostKBlock,
                        0,
                        NoKBlockInnerDimFn,
                        NoIn0Source,
                        In1Offset,
                        true>(
                        act_tiles,
                        weight_tiles,
                        partials,
                        partials,
                        shape,
                        {},
                        TilizeKBlock{my_slice, m_block},
                        block_n_tiles,
                        block_n_tiles,
                        {},
                        {},
                        {},
                        In1Offset{});
                partials.push_back(block_out_tiles);
                PACK((llk_pack_reconfig_l1_acc(0)));

                bias_and_untilize<has_bias>(partials, bias, biased, n_block, m_block + 1 == num_m_blocks);
            }
        }
    }

    if constexpr (has_bias) {
        bias.pop_front(core_n_tiles);
    }
    if constexpr (weights_resident) {
        weight_tiles.pop_front(resident_weight_tiles);
    }
}
