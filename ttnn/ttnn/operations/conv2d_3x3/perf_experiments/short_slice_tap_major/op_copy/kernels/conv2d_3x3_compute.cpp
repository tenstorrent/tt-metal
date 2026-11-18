// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// conv2d_3x3 compute: per M-block, [untilize_footprint (TILE input)] then tilize_matmul_block
// (matmul_block with tilize as its PreKBlockFn, fp32 packer-L1 accumulation over the K walk) then
// finalize_block (row-broadcast bias add). No bias: the last K-block packs straight to the output CB
// and finalize_block is empty. ReLU applies only to the final value: the bias-add pack, or the last
// K-block's pack without bias.
//
// Face-direct im2col: the reader writes the in0 tiles itself, so the matmul runs without the tilize hook. With one
// K-block per M-block the reader pushes in0 in row groups and compute runs one matmul call per group as it lands,
// keeping the weights fronted across the calls, and adds the bias once per bias_stream_groups groups, so only the
// last chunk's bias add trails the gather.
//
// matmul_block leaves packer L1 accumulation on and (OutWithRelu) PACK_RELU on; every later pack
// (untilize, tilize, combine, bias add) first restores the state it needs.
//
// K split (BLOCK_SHARDED, k_cols > 1): the matmul accumulates this core's channel slice into fp32
// partials [bm, all C_out] that the WRITER consumes (it scatters N-slice y to grid column y of the row),
// so compute owns that CB's reserve/push once per M-block instead of letting the helper drain it per
// K-block (the helper's drain would make compute a second consumer). The K-blocks still serialize on the
// packer: each one's tilize pushes cb_im2col_tiles between two L1-accumulating passes. Then, on a core
// that owns output columns, combine_block sums the k_cols slices the row sent it (pairwise FPU adds into
// fp32 DEST), adds the bias and packs into the output shard; ReLU is armed only for that last pack.

#include <cstdint>

#ifdef CONV2D_NO_ZONES  // short_slice_tap_major experiment: unzoned timing
#define MaybeDeviceZoneScope(name)
#endif
#include <type_traits>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace ckl = compute_kernel_lib;

namespace {

// Tilizes one K-block (block_m_tiles tile-rows x block_k_tiles) of gathered im2col rows. The matmul
// restores its own state afterwards (InitMode::ShortAfterPreKBlock).
// The reader lays a tile-row out either as one [32, block_k_tiles * 32] row block or tap-major, as
// block_k_tiles / tilize_width_tiles blocks of [32, tilize_width_tiles * 32]; tilizing those blocks in order
// yields the same tiles in the same [m][k] order.
template <
    uint32_t block_k_tiles,
    uint32_t tilize_width_tiles,
    uint32_t block_m_tiles,
    uint32_t cb_im2col_rm,
    uint32_t cb_im2col_tiles,
    bool relu_on_last_matmul_pack>
struct TilizeIm2colKBlock {
    DataflowBuffer* rm;
    DataflowBuffer* tiles;

    ALWI void operator()(uint32_t /*block*/, uint32_t /*num_k_blocks*/, bool is_last) const {
        // The previous K-block's pack left packer L1 accumulation on; the tilize pack must overwrite.
        pack_reconfig_l1_acc(0);
        if constexpr (relu_on_last_matmul_pack) {
            // OutWithRelu turns PACK_RELU on before this hook; activations must not be clamped.
            pack_relu_config(ReluConfig::none());
        }
        static_assert(block_k_tiles % tilize_width_tiles == 0);
        {
            // Includes the unpacker's per-tile-row wait on the reader's gather.
            MaybeDeviceZoneScope("compute_tilize");
            ckl::tilize<tilize_width_tiles, cb_im2col_rm, cb_im2col_tiles>(
                *rm, *tiles, block_m_tiles * (block_k_tiles / tilize_width_tiles));
        }
        if constexpr (relu_on_last_matmul_pack) {
            if (is_last) {
                pack_relu_config(ReluConfig::zero());
            }
        }
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
    constexpr bool apply_relu = get_compile_time_arg_val(13) != 0;
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(14);
    constexpr uint32_t cb_halo_slab = get_compile_time_arg_val(15);
    constexpr bool tile_input = get_compile_time_arg_val(16) != 0;
    constexpr uint32_t c_in_tiles = get_compile_time_arg_val(17);
    constexpr uint32_t slab_tile_rows = get_compile_time_arg_val(18);
    constexpr bool k_split = get_compile_time_arg_val(19) != 0;
    constexpr uint32_t k_cols = get_compile_time_arg_val(20);
    constexpr uint32_t out_n_tiles = get_compile_time_arg_val(21);
    constexpr uint32_t cb_combine_slots = get_compile_time_arg_val(22);
    constexpr uint32_t cb_combined = get_compile_time_arg_val(23);
    constexpr uint32_t combine_subblock_h = get_compile_time_arg_val(24);
    constexpr uint32_t combine_subblock_w = get_compile_time_arg_val(25);
    constexpr uint32_t im2col_tilize_width_tiles = get_compile_time_arg_val(26);
    constexpr bool face_direct = get_compile_time_arg_val(27) != 0;
    constexpr uint32_t in0_stream_rows = get_compile_time_arg_val(28);
    constexpr uint32_t bias_stream_groups = get_compile_time_arg_val(29);

    const uint32_t num_m_blocks_this_core = get_arg_val<uint32_t>(0);
    [[maybe_unused]] const bool owns_output_columns = get_arg_val<uint32_t>(1) != 0;

    constexpr uint32_t in0_num_subblocks = block_m_tiles / out_subblock_h;
    constexpr uint32_t in1_num_subblocks = block_n_tiles / out_subblock_w;
    static_assert(in0_num_subblocks * out_subblock_h == block_m_tiles);
    static_assert(in1_num_subblocks * out_subblock_w == block_n_tiles);
    constexpr auto mm_shape = ckl::MatmulBlockShape::of(
        in0_num_subblocks, in1_num_subblocks, out_subblock_h, out_subblock_w, block_k_tiles, num_k_blocks);
    // One matmul call per in0 row group (face-direct, one K-block): the group's subblock rows x all of N.
    // The last group may be shorter; every group is whole subblocks.
    constexpr uint32_t in0_groups = (block_m_tiles + in0_stream_rows - 1) / in0_stream_rows;
    static_assert(in0_stream_rows % out_subblock_h == 0);
    static_assert(in0_groups == 1 || (face_direct && num_k_blocks == 1 && !k_split));
    static_assert(bias_stream_groups > 0);
    constexpr auto bias_shape =
        ckl::BiasAddShape::of(in0_num_subblocks, in1_num_subblocks, out_subblock_h, out_subblock_w, block_n_tiles);
    static_assert(block_m_tiles % combine_subblock_h == 0 && out_n_tiles % combine_subblock_w == 0);
    constexpr auto combine_bias_shape = ckl::BiasAddShape::of(
        block_m_tiles / combine_subblock_h,
        out_n_tiles / combine_subblock_w,
        combine_subblock_h,
        combine_subblock_w,
        out_n_tiles);
    constexpr uint32_t partials_tiles = block_m_tiles * block_n_tiles;
    // The k_cols slots hold the row's contributions to this core's [bm, out_n_tiles] output block, one
    // slot after another: a k_cols-row x slot-wide block whose column sums are the outputs.
    constexpr uint32_t slot_tiles = block_m_tiles * out_n_tiles;
    // kernel_main is not a template, so the TILE-only untilize is instantiated on the ROW_MAJOR path too.
    constexpr uint32_t untilize_width_tiles = tile_input ? c_in_tiles : 1;
    // Under a K split the matmul packs partials, which ReLU must never touch.
    constexpr bool relu_on_last_matmul_pack = apply_relu && !has_bias && !k_split;
    constexpr ckl::LastBlockTarget last_block_target =
        has_bias ? ckl::LastBlockTarget::Interm
                 : (apply_relu ? ckl::LastBlockTarget::OutWithRelu : ckl::LastBlockTarget::Out);

    // Startup matches the first op (untilize for TILE input, the matmul for face-direct, else tilize); every
    // later phase reconfigures itself.
    if constexpr (tile_input) {
        compute_kernel_hw_startup(cb_input_tiles, cb_halo_slab);
    } else if constexpr (face_direct) {
        compute_kernel_hw_startup<SrcOrder::Reverse>(
            cb_im2col_tiles, cb_weights, (has_bias || k_split) ? cb_matmul_partials : cb_output_tiles);
    } else {
        compute_kernel_hw_startup(cb_im2col_rm, cb_im2col_tiles);
    }

    DataflowBuffer input_tiles_dfb(cb_input_tiles);
    DataflowBuffer halo_slab_dfb(cb_halo_slab);
    DataflowBuffer im2col_rm_dfb(cb_im2col_rm);
    DataflowBuffer im2col_tiles_dfb(cb_im2col_tiles);
    CircularBuffer im2col_tiles(cb_im2col_tiles);
    CircularBuffer weights(cb_weights);
    CircularBuffer partials(cb_matmul_partials);
    CircularBuffer output_tiles(cb_output_tiles);
    CircularBuffer bias(cb_bias);
    CircularBuffer combined(cb_combined);
    DataflowBuffer combine_slots_dfb(cb_combine_slots);
    DataflowBuffer combined_dfb(cb_combined);
    DataflowBuffer output_tiles_dfb(cb_output_tiles);

    using TilizeFn = TilizeIm2colKBlock<
        block_k_tiles,
        im2col_tilize_width_tiles,
        block_m_tiles,
        cb_im2col_rm,
        cb_im2col_tiles,
        relu_on_last_matmul_pack>;
    auto make_pre_k_block = [&]() {
        if constexpr (face_direct) {
            return ckl::NoPreKBlock{};
        } else {
            return TilizeFn{&im2col_rm_dfb, &im2col_tiles_dfb};
        }
    };
    const auto pre_k_block = make_pre_k_block();
    using PreKBlockFn = std::decay_t<decltype(pre_k_block)>;

    if constexpr (has_bias) {
        bias.wait_front(out_n_tiles);
    }

    // finalize_block over `shape`'s rows of the matmul partials: + bias, ReLU on this pack only.
    auto add_bias = [&](const ckl::BiasAddShape& shape) {
        MaybeDeviceZoneScope("compute_bias_add");
        // matmul_block leaves packer L1 accumulation on after its last K-block.
        pack_reconfig_l1_acc(0);
        if constexpr (apply_relu) {
            pack_relu_config(ReluConfig::zero());
        }
        ckl::add_bias_bcast_rows<ckl::BiasBroadcast::RowBroadcast, ckl::OutputCBLayout::TileRowMajor>(
            partials, bias, output_tiles, shape);
    };

    // Sum the row's k_cols contributions per output tile and pack them; with a bias the fp32 sum goes
    // through cb_combined into the bias add, otherwise straight to the output shard.
    auto combine_block = [&]() {
        {
            // The row's slices arrive over the NoC; waiting here keeps that wait out of the sum's zone.
            MaybeDeviceZoneScope("compute_combine_wait");
            combine_slots_dfb.wait_front(k_cols * slot_tiles);
        }
        MaybeDeviceZoneScope("compute_combine");
        pack_reconfig_l1_acc(0);
        if constexpr (apply_relu && !has_bias) {
            pack_relu_config(ReluConfig::zero());
        }
        constexpr uint32_t sum_cb = has_bias ? cb_combined : cb_output_tiles;
        // The cross-core inputs are raw partial sums, so only the cross-tile adds run (Skip). No scaler
        // tile is read on this path; the scaler slot names the input CB.
        ckl::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_COL,
            cb_combine_slots,
            cb_combine_slots,
            sum_cb,
            ckl::ReduceInputPolicy::BulkWaitBulkPop,
            ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
            ReduceFp32Mode::Fast,
            ckl::ReduceAlgorithm::AccumulateViaAdd,
            ckl::ReduceWithinTile::Skip>(
            combine_slots_dfb,
            combine_slots_dfb,
            has_bias ? combined_dfb : output_tiles_dfb,
            ckl::ReduceInputBlockShape::of(k_cols, slot_tiles));
        if constexpr (has_bias) {
            if constexpr (apply_relu) {
                pack_relu_config(ReluConfig::zero());
            }
            ckl::add_bias_bcast_rows<ckl::BiasBroadcast::RowBroadcast, ckl::OutputCBLayout::TileRowMajor>(
                combined, bias, output_tiles, combine_bias_shape);
        }
    };

    for (uint32_t m_block_idx = 0; m_block_idx < num_m_blocks_this_core; ++m_block_idx) {
        if constexpr (tile_input) {
            // untilize_footprint: the reader streams slab_tile_rows input tile-rows in slab order.
            MaybeDeviceZoneScope("compute_untilize");
            pack_reconfig_l1_acc(0);
            ckl::untilize<untilize_width_tiles, cb_input_tiles, cb_halo_slab>(
                input_tiles_dfb, halo_slab_dfb, slab_tile_rows);
        }

        if constexpr (k_split) {
            {
                // The whole K walk, tilize hooks included (compute_tilize is the tilize share).
                MaybeDeviceZoneScope("compute_matmul_walk");
                partials.reserve_back(partials_tiles);
                ckl::matmul_block<
                    /*transpose=*/false,
                    /*packer_l1_acc=*/true,
                    ckl::LastBlockTarget::Interm,
                    ckl::OutputCBLayout::TileRowMajor,
                    ckl::matmul_config::InitMode::ShortAfterPreKBlock,
                    ckl::InputPolicy::WaitAndPopPerKBlock,
                    ckl::InputPolicy::WaitAndPopPerKBlock,
                    ckl::NoPostCompute,
                    PreKBlockFn,
                    ckl::NoPostKBlock,
                    /*untilize_block_ct_dim=*/0,
                    ckl::NoKBlockInnerDimFn,
                    ckl::NoIn0Source,
                    ckl::NoIn1BaseOffset,
                    /*caller_owns_pack_target=*/true>(
                    im2col_tiles,
                    weights,
                    output_tiles,
                    partials,
                    mm_shape,
                    {},
                    pre_k_block,
                    /*in1_per_core_w=*/block_n_tiles,
                    /*out_row_width=*/block_n_tiles);
                partials.push_back(partials_tiles);
            }
            if (owns_output_columns) {
                combine_block();
            }
        } else if constexpr (in0_groups > 1) {
            // Matmul and bias calls interleave with the streamed in0 groups; one zone for the whole block.
            MaybeDeviceZoneScope("compute_matmul_stream");
            for (uint32_t group = 0; group < in0_groups; ++group) {
                const uint32_t group_end =
                    (group + 1) * in0_stream_rows < block_m_tiles ? (group + 1) * in0_stream_rows : block_m_tiles;
                ckl::matmul_block<
                    /*transpose=*/false,
                    /*packer_l1_acc=*/true,
                    last_block_target,
                    ckl::OutputCBLayout::TileRowMajor,
                    ckl::matmul_config::InitMode::Short,
                    ckl::InputPolicy::WaitAndPopPerKBlock,
                    ckl::InputPolicy::WaitAndRetainOnLastBlock>(
                    im2col_tiles,
                    weights,
                    output_tiles,
                    partials,
                    ckl::MatmulBlockShape::of(
                        (group_end - group * in0_stream_rows) / out_subblock_h,
                        in1_num_subblocks,
                        out_subblock_h,
                        out_subblock_w,
                        block_k_tiles,
                        1),
                    {},
                    {},
                    /*in1_per_core_w=*/block_n_tiles,
                    /*out_row_width=*/block_n_tiles);
                if constexpr (has_bias) {
                    const uint32_t groups_done = group + 1;
                    if (groups_done % bias_stream_groups == 0 || groups_done == in0_groups) {
                        const uint32_t chunk_rows =
                            group_end - (group / bias_stream_groups) * bias_stream_groups * in0_stream_rows;
                        add_bias(ckl::BiasAddShape::of(
                            chunk_rows / out_subblock_h,
                            in1_num_subblocks,
                            out_subblock_h,
                            out_subblock_w,
                            block_n_tiles));
                        if constexpr (apply_relu) {
                            pack_relu_config(ReluConfig::none());
                        }
                    }
                }
            }
            weights.pop_front(block_k_tiles * block_n_tiles);
        } else {
            MaybeDeviceZoneScope("compute_matmul_walk");
            ckl::matmul_block<
                /*transpose=*/false,
                /*packer_l1_acc=*/true,
                last_block_target,
                ckl::OutputCBLayout::TileRowMajor,
                ckl::matmul_config::InitMode::ShortAfterPreKBlock,
                ckl::InputPolicy::WaitAndPopPerKBlock,
                ckl::InputPolicy::WaitAndPopPerKBlock,
                ckl::NoPostCompute,
                PreKBlockFn>(
                im2col_tiles,
                weights,
                output_tiles,
                partials,
                mm_shape,
                {},
                pre_k_block,
                /*in1_per_core_w=*/block_n_tiles,
                /*out_row_width=*/block_n_tiles);
        }

        if constexpr (has_bias && !k_split && in0_groups == 1) {
            add_bias(bias_shape);
        }
        if constexpr (apply_relu) {
            pack_relu_config(ReluConfig::none());
        }
    }

    if constexpr (has_bias) {
        bias.pop_front(out_n_tiles);
    }
}
