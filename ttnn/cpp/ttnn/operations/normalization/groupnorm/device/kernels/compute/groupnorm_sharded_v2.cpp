// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "api/compute/matmul.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "api/dataflow/dataflow_buffer.h"

// SPLIT REDUCE across Cores
void kernel_main() {
    constexpr uint32_t is_mcast_sender = get_compile_time_arg_val(0);
    constexpr uint32_t do_gamma = get_compile_time_arg_val(1);
    constexpr uint32_t do_beta = get_compile_time_arg_val(2);
    constexpr uint32_t num_cores_per_mcast_group = get_compile_time_arg_val(3);
    // True when a reconfig-relevant operand is fp32: the per-group reconfig_data_format calls below
    // are then required. All-bf16 compiles them out (no-ops). See program factory.
    constexpr bool enable_fp32_reconfig = get_named_compile_time_arg_val("enable_fp32_reconfig") != 0;

    constexpr uint32_t batch = get_compile_time_arg_val(4);
    constexpr uint32_t group = get_compile_time_arg_val(5);

    constexpr uint32_t num_cols_per_group = get_compile_time_arg_val(6);

    volatile uint32_t block_h = get_compile_time_arg_val(7);
    constexpr uint32_t block_w = get_compile_time_arg_val(8);
    constexpr uint32_t block_hw = get_compile_time_arg_val(9);

    constexpr uint32_t subblock_w = get_compile_time_arg_val(10);
    constexpr uint32_t num_subblocks_w = get_compile_time_arg_val(11);

    constexpr uint32_t per_core_M = get_compile_time_arg_val(12);
    constexpr uint32_t per_core_N = get_compile_time_arg_val(13);
    constexpr uint32_t per_core_MN = get_compile_time_arg_val(14);

    constexpr uint32_t per_core_N_tile_bytes = get_compile_time_arg_val(15);
    constexpr uint32_t num_groups_per_reset = get_compile_time_arg_val(16);

    constexpr uint32_t single_tile_size_bytes = get_compile_time_arg_val(17);
    constexpr uint32_t num_tiles_per_batch = get_compile_time_arg_val(18);

    constexpr uint32_t num_tiles_input_mask = get_compile_time_arg_val(19);
    constexpr uint32_t block_w_last = get_compile_time_arg_val(20);
    constexpr uint32_t GROUP_SIZE_IS_POWER_OF_2 = get_compile_time_arg_val(21);
    constexpr uint32_t GROUP_SIZE_SMALLER_THAN_TILE_W = get_compile_time_arg_val(22);
    constexpr uint32_t group_row_offset = get_compile_time_arg_val(23);
    constexpr uint32_t tile_width = get_compile_time_arg_val(24);

    // Non-tile-aligned H*W; see compute/groupnorm.cpp. logical_hw / padded_hw only keep two shapes
    // padding to the same size out of one cached program; has_row_mask is what this branches on.
    constexpr uint32_t logical_hw [[maybe_unused]] = get_compile_time_arg_val(25);
    constexpr uint32_t padded_hw [[maybe_unused]] = get_compile_time_arg_val(26);
    constexpr bool has_row_mask = get_compile_time_arg_val(27) == 1;
    // Composed-mask protocol: the writer ships ONE row-0-only column-selector set per (batch,
    // group) unconditionally; under has_row_mask this kernel composes the batch's final
    // row-tile's mask on device as rowvalid (c_18) x column selector -> c_19.
    constexpr uint32_t mask_tiles_per_group = block_w;

    constexpr uint32_t block_w_minus_one = block_w - 1;
    constexpr uint32_t block_w_minus_two = block_w - 2;
    constexpr uint32_t tile_w_minux_group_size = tile_width - num_cols_per_group;

    // dst regs
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t scaler0 = 0;

    // input cbs
    constexpr uint32_t dfb_in0_id = tt::CBIndex::c_0;
    constexpr uint32_t dfb_in_id = tt::CBIndex::c_1;
    constexpr uint32_t dfb_scaler_id = tt::CBIndex::c_2;
    constexpr uint32_t dfb_scaler_global_id = tt::CBIndex::c_4;
    constexpr uint32_t dfb_eps_id = tt::CBIndex::c_3;
    constexpr uint32_t dfb_gamma_id = tt::CBIndex::c_5;
    constexpr uint32_t dfb_beta_id = tt::CBIndex::c_6;
    constexpr uint32_t dfb_input_mask_id = tt::CBIndex::c_7;
    // Pass-1 operand source: tilized-input CB when the reader tilizes, raw input CB otherwise.
    // (The init calls keep using dfb_in0_id, as before -- only the ops switch.)
#ifdef TILIZE_IN
    constexpr uint32_t dfb_pass1_src_id = dfb_in_id;
#else
    constexpr uint32_t dfb_pass1_src_id = dfb_in0_id;
#endif

    // interm cbs
    constexpr uint32_t dfb_repack_id = tt::CBIndex::c_11;
    constexpr uint32_t dfb_repack_out_id = tt::CBIndex::c_12;
    constexpr uint32_t dfb_x_id = tt::CBIndex::c_13;
    constexpr uint32_t dfb_ex_partial_id = tt::CBIndex::c_8;
    constexpr uint32_t dfb_ex_id = tt::CBIndex::c_9;
    constexpr uint32_t dfb_ex_external_id = tt::CBIndex::c_10;
    constexpr uint32_t dfb_ex_global_id = num_cores_per_mcast_group == 1 ? dfb_ex_partial_id : tt::CBIndex::c_15;
    constexpr uint32_t dfb_ex2pe_id = tt::CBIndex::c_17;
    constexpr uint32_t dfb_ones_id = tt::CBIndex::c_26;
    // Composed-mask CBs, created only under pad correction (has_row_mask); aliased to
    // always-present CBs otherwise.
    constexpr uint32_t dfb_rowvalid_id = has_row_mask ? tt::CBIndex::c_18 : tt::CBIndex::c_26;
    constexpr uint32_t dfb_mask_last_id = has_row_mask ? tt::CBIndex::c_19 : tt::CBIndex::c_7;

    // output cb
    constexpr uint32_t dfb_out0_id = tt::CBIndex::c_16;
#ifdef UNTILIZE_OUT
    // not used in cases of negative mask
    constexpr uint32_t dfb_out_id = tt::CBIndex::c_30;
#else
    constexpr uint32_t dfb_out_id =
        (do_gamma or do_beta) ? (((do_gamma and not do_beta) or (not do_gamma and do_beta)) ? dfb_in_id : dfb_out0_id)
                              : dfb_out0_id;
#endif

    // tile offset
    uint32_t index_subblock_w_offset = 0;
    uint32_t index_h_offset = 0;
    uint32_t index_w_offset = 0;
    uint32_t index_b_offset = 0;
    uint32_t index_g_offset = 0;
    // data offset
    uint32_t num_datum_per_row_offeset = 0;
    // inplace out cbs
    bool copy_or_add = true;
    uint32_t group_reset_index = 0;
    uint32_t index_block_w = 0;
    uint32_t row_offset = num_cols_per_group;

#ifdef UNTILIZE_OUT
#ifndef FUSE_NEGATIVE_MASK
    constexpr int dfb_outgamma_id = dfb_in_id;
    constexpr int dfb_inbeta_id = do_gamma ? dfb_outgamma_id : dfb_out_id;
    constexpr int dfb_outbeta_id = do_gamma ? dfb_out_id : dfb_in_id;
    constexpr int dfb_untilize_in_id = (do_gamma and not do_beta) ? dfb_outgamma_id
                                       : do_beta                  ? dfb_outbeta_id
                                                                  : dfb_out_id;
    constexpr int dfb_untilize_out_id =
#ifdef READER_REPACK
        dfb_repack_out_id;
#else
        dfb_out0_id;
#endif
#else
    constexpr int dfb_outgamma_id = dfb_in_id;
    constexpr int dfb_inbeta_id = dfb_in_id;
    constexpr int dfb_outbeta_id = dfb_in_id;
    constexpr int dfb_untilize_in_id = dfb_in_id;
    constexpr int dfb_untilize_out_id =
#ifdef READER_REPACK
        dfb_repack_out_id;
#else
        dfb_out0_id;
#endif
#endif
#else
    constexpr int dfb_outgamma_id = do_beta ? dfb_in_id : dfb_out0_id;
    constexpr int dfb_inbeta_id = do_gamma ? dfb_outgamma_id : dfb_out_id;
    constexpr int dfb_outbeta_id = dfb_out0_id;
#endif

    // Used in cases of negative mask provided
    constexpr uint32_t dfb_in_negative_mask_id = tt::CBIndex::c_14;

#ifdef FUSE_NEGATIVE_MASK
    constexpr bool use_negative_mask = true;
#else
    constexpr bool use_negative_mask = false;
#endif

    DataflowBuffer dfb_beta(dfb_beta_id);
    DataflowBuffer dfb_eps(dfb_eps_id);
    DataflowBuffer dfb_ex(dfb_ex_id);
    DataflowBuffer dfb_ex2pe(dfb_ex2pe_id);
    DataflowBuffer dfb_ex_external(dfb_ex_external_id);
    DataflowBuffer dfb_ex_global(dfb_ex_global_id);
    DataflowBuffer dfb_ex_partial(dfb_ex_partial_id);
    DataflowBuffer dfb_gamma(dfb_gamma_id);
    DataflowBuffer dfb_in(dfb_in_id);
    DataflowBuffer dfb_in_negative_mask(dfb_in_negative_mask_id);
    DataflowBuffer dfb_inbeta(dfb_inbeta_id);
    DataflowBuffer dfb_input_mask(dfb_input_mask_id);
    DataflowBuffer dfb_mask_last(dfb_mask_last_id);
    DataflowBuffer dfb_ones(dfb_ones_id);
    DataflowBuffer dfb_out(dfb_out_id);
    DataflowBuffer dfb_outbeta(dfb_outbeta_id);
    DataflowBuffer dfb_outgamma(dfb_outgamma_id);
    DataflowBuffer dfb_rowvalid(dfb_rowvalid_id);
    DataflowBuffer dfb_scaler(dfb_scaler_id);
    DataflowBuffer dfb_scaler_global(dfb_scaler_global_id);
    DataflowBuffer dfb_x(dfb_x_id);

// tilize input from RM to tile layout
#ifdef TILIZE_IN
    compute_kernel_hw_startup(dfb_in0_id, dfb_in0_id, dfb_in_id);
// Tilize in0 -> in (row-major to tiled)
#ifdef READER_REPACK
    constexpr uint32_t dfb_in_rm_id = dfb_repack_id;
    compute_kernel_lib::tilize<
        per_core_N,
        dfb_in_rm_id,
        dfb_in_id,
        compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(per_core_M);
#else
    constexpr uint32_t dfb_in_rm_id = dfb_in0_id;
    compute_kernel_lib::tilize<
        per_core_N,
        dfb_in_rm_id,
        dfb_in_id,
        compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::tilize_config::WaitMode::NoWait,
        compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(per_core_M);
#endif
    dfb_in.wait_front(per_core_MN);
#else
    compute_kernel_hw_startup(dfb_in0_id, dfb_input_mask_id, dfb_x_id);
#endif

    if constexpr (has_row_mask) {
        dfb_rowvalid.wait_front(1);
    }

    index_b_offset = 0;
    for (uint32_t b = 0; b < batch; ++b) {
        index_g_offset = 0;
        for (uint32_t g = 0; g < group; ++g) {
            // mask input
            index_h_offset = index_b_offset + index_g_offset;
            reconfig_data_format_srcb(dfb_in0_id, dfb_input_mask_id);
            dfb_input_mask.wait_front(mask_tiles_per_group);
            // Compose the final row-tile's mask: rowvalid[r] * colsel[c] -> dfb_mask_last. The
            // column selector's row 0 is broadcast down the rowvalid tile, so the product
            // varies down the rows.
            if constexpr (has_row_mask) {
                if constexpr (enable_fp32_reconfig) {
                    reconfig_data_format_srca(dfb_rowvalid_id);
                    pack_reconfig_data_format(dfb_mask_last_id);
                }
                mul_bcast_rows_init(dfb_rowvalid_id, dfb_input_mask_id);
                dfb_mask_last.reserve_back(block_w);
                index_subblock_w_offset = 0;
                for (uint32_t j = 0; j < num_subblocks_w; ++j) {
                    tile_regs_acquire();
                    for (uint32_t w = 0; w < subblock_w; ++w) {
                        mul_tiles_bcast_rows(dfb_rowvalid_id, dfb_input_mask_id, 0, w + index_subblock_w_offset, w);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t w = 0; w < subblock_w; ++w) {
                        pack_tile(w, dfb_mask_last_id);
                    }
                    tile_regs_release();
                    index_subblock_w_offset += subblock_w;
                }
                dfb_mask_last.push_back(block_w);
                if constexpr (enable_fp32_reconfig) {
                    reconfig_data_format_srca(dfb_in0_id);
                    pack_reconfig_data_format(dfb_x_id);
                }
            }
            dfb_x.reserve_back(block_hw);
            const uint32_t bcast_row_tiles = has_row_mask ? block_h - 1 : block_h;
            mul_bcast_rows_init(dfb_in0_id, dfb_input_mask_id);
            for (uint32_t i = 0; i < bcast_row_tiles; ++i) {
                index_subblock_w_offset = 0;
                for (uint32_t j = 0; j < num_subblocks_w; ++j) {
                    tile_regs_acquire();
                    for (uint32_t w = 0; w < subblock_w; ++w) {
                        uint32_t index = w + index_subblock_w_offset + index_h_offset;
                        // When the last group spans fewer than block_w tiles, the index can
                        // exceed the CB tile count. Clamp it so the read stays in bounds;
                        // the input mask guarantees the result from the clamped tile is zeroed.
                        if (index >= per_core_MN) {
                            index = per_core_MN - 1;
                        }
                        mul_tiles_bcast_rows(
                            dfb_pass1_src_id, dfb_input_mask_id, index, w + index_subblock_w_offset, w);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < subblock_w; ++i) {
                        pack_tile(i, dfb_x_id);
                    }
                    tile_regs_release();
                    index_subblock_w_offset += subblock_w;
                }
                index_h_offset += per_core_N;
            }
            // Peeled final row-tile: full-tile multiply with the composed mask.
            if constexpr (has_row_mask) {
                dfb_mask_last.wait_front(block_w);
                reconfig_data_format_srcb(dfb_input_mask_id, dfb_mask_last_id);
                mul_init(dfb_in0_id, dfb_mask_last_id);
                index_subblock_w_offset = 0;
                for (uint32_t j = 0; j < num_subblocks_w; ++j) {
                    tile_regs_acquire();
                    for (uint32_t w = 0; w < subblock_w; ++w) {
                        uint32_t index = w + index_subblock_w_offset + index_h_offset;
                        if (index >= per_core_MN) {
                            index = per_core_MN - 1;
                        }
                        mul_tiles(dfb_pass1_src_id, dfb_mask_last_id, index, w + index_subblock_w_offset, w);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < subblock_w; ++i) {
                        pack_tile(i, dfb_x_id);
                    }
                    tile_regs_release();
                    index_subblock_w_offset += subblock_w;
                }
                index_h_offset += per_core_N;
            }
            dfb_x.push_back(block_hw);
            reconfig_data_format_srcb(has_row_mask ? dfb_mask_last_id : dfb_input_mask_id, dfb_ones_id);

            // Partial-E[x]
            index_h_offset = 0;
            mul_init(dfb_x_id, dfb_ones_id);
            dfb_ex2pe.reserve_back(1);
            tile_regs_acquire();
            dfb_x.wait_front(block_hw);
            dfb_ones.wait_front(1);

            index_h_offset = 0;
            // Accumulate into dest directly by using mul_tiles (tile * 1 is accumulated into dest)
            // Alternative is to use reduce_tile multiple times, but this showed to be more precise and faster.
            for (uint32_t h = 0; h < block_h; ++h) {
                for (uint32_t w = 0; w < block_w; ++w) {
                    uint32_t index = index_h_offset + w;
                    mul_tiles(dfb_x_id, dfb_ones_id, index, 0, dst0);
                }
                index_h_offset += block_w;
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, dfb_ex2pe_id);
            tile_regs_release();
            dfb_ex2pe.push_back(1);

            // reduce only one final tile
            //
            // Note that reader_mcast_sender_unary_sharded_gn_v2.cpp depends on the
            // documented behavior of REDUCE_SCALAR's packer to set every
            // non-result datum of dfb_ex_partial to zero.
            // If this `reduce<…, REDUCE_SCALAR>` pack into dfb_ex_partial is
            // ever replaced by something that does not have the same
            // packer-zero contract (e.g. a `pack_tile` / `pack_tile_block`
            // path like welford_groupnorm_sharded_v2.cpp uses), the sharded
            // reader's "single-tile-overwrite trick" must be adjusted accordingly
            // (e.g. use `zero_whole_cb` from groupnorm_zero_fill.hpp, mirroring the
            // mcast reader). Same applies to the second REDUCE_SCALAR pack into
            // dfb_ex_partial later in this kernel (variance).
            compute_kernel_lib::
                reduce<PoolType::SUM, ReduceDim::REDUCE_SCALAR, dfb_ex2pe_id, dfb_scaler_id, dfb_ex_partial_id>(
                    compute_kernel_lib::ReduceInputBlockShape::single());

            if constexpr (is_mcast_sender and num_cores_per_mcast_group > 1) {
                compute_kernel_lib::reduce<
                    PoolType::SUM,
                    ReduceDim::REDUCE_SCALAR,
                    dfb_ex_external_id,
                    dfb_scaler_global_id,
                    dfb_ex_global_id,
                    compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
                    compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(
                    compute_kernel_lib::ReduceInputBlockShape::single());
                dfb_ex.reserve_back(1);
                dfb_ex.push_back(1);
            }
            dfb_ex_global.wait_front(1);

            // x - E[x]
            // fp32: reset both srcs so fp32 x/mean aren't read through the partial-E[x] bf16 dfb_ones format.
            // The reconfig has to precede the init: the init's LLK assert checks that the unpack config
            // registers already describe these operands. (The MOP is built from the init's static
            // arguments; the registers themselves are consumed later, by UNPACR.)
            if constexpr (enable_fp32_reconfig) {
                reconfig_data_format_srca(dfb_x_id);
                reconfig_data_format_srcb(dfb_ex_global_id);
            }
            sub_bcast_scalar_init(dfb_x_id, dfb_ex_global_id);
            for (uint32_t i = 0; i < block_h; i++) {
                index_subblock_w_offset = 0;
                for (uint32_t j = 0; j < num_subblocks_w; j++) {
                    tile_regs_acquire();
                    for (uint32_t w = 0; w < subblock_w; w++) {
                        uint32_t index = w + index_subblock_w_offset;
                        sub_tiles_bcast_scalar(dfb_x_id, dfb_ex_global_id, index, 0, w);
                    }
                    tile_regs_commit();
                    dfb_x.pop_front(subblock_w);
                    dfb_x.reserve_back(subblock_w);
                    tile_regs_wait();
                    for (uint32_t k = 0; k < subblock_w; k++) {
                        pack_tile(k, dfb_x_id);
                    }
                    dfb_x.push_back(subblock_w);
                    tile_regs_release();
                    dfb_x.wait_front(block_hw);
                }
            }
            dfb_ex_global.pop_front(1);

            reconfig_data_format_srcb(dfb_ex_global_id, dfb_input_mask_id);
            // Re-mask (x - E[x]); otherwise each padding row is centered to (garbage - E[x])
            // and squared into the variance.
            mul_bcast_rows_init(dfb_x_id, dfb_input_mask_id);
            dfb_x.wait_front(block_hw);

            for (uint32_t i = 0; i < bcast_row_tiles; i++) {
                index_subblock_w_offset = 0;
                for (uint32_t j = 0; j < num_subblocks_w; ++j) {
                    tile_regs_acquire();
                    for (uint32_t w = 0; w < subblock_w; ++w) {
                        uint32_t index = w + index_subblock_w_offset;
                        mul_tiles_bcast_rows(dfb_x_id, dfb_input_mask_id, index, index, w);
                    }
                    tile_regs_commit();

                    dfb_x.pop_front(subblock_w);
                    dfb_x.reserve_back(subblock_w);

                    tile_regs_wait();
                    for (uint32_t i = 0; i < subblock_w; ++i) {
                        pack_tile(i, dfb_x_id);
                    }
                    dfb_x.push_back(subblock_w);
                    tile_regs_release();
                }
            }
            // Peeled final row-tile: full-tile multiply with the composed mask.
            if constexpr (has_row_mask) {
                reconfig_data_format_srcb(dfb_input_mask_id, dfb_mask_last_id);
                mul_init(dfb_x_id, dfb_mask_last_id);
                index_subblock_w_offset = 0;
                for (uint32_t j = 0; j < num_subblocks_w; ++j) {
                    tile_regs_acquire();
                    for (uint32_t w = 0; w < subblock_w; ++w) {
                        uint32_t index = w + index_subblock_w_offset;
                        mul_tiles(dfb_x_id, dfb_mask_last_id, index, index, w);
                    }
                    tile_regs_commit();

                    dfb_x.pop_front(subblock_w);
                    dfb_x.reserve_back(subblock_w);

                    tile_regs_wait();
                    for (uint32_t i = 0; i < subblock_w; ++i) {
                        pack_tile(i, dfb_x_id);
                    }
                    dfb_x.push_back(subblock_w);
                    tile_regs_release();
                }
            }
            dfb_input_mask.pop_front(mask_tiles_per_group);
            if constexpr (has_row_mask) {
                dfb_mask_last.pop_front(block_w);
            }
            reconfig_data_format_srcb(has_row_mask ? dfb_mask_last_id : dfb_input_mask_id, dfb_x_id);

            // (x - E[x])^2
            index_h_offset = 0;
            mul_init(dfb_x_id, dfb_x_id);
            dfb_x.wait_front(block_hw);

            tile_regs_acquire();
            dfb_ex2pe.reserve_back(1);

            for (uint32_t i = 0; i < block_h; i++) {
                index_subblock_w_offset = 0;
                for (uint32_t j = 0; j < num_subblocks_w; j++) {
                    for (uint32_t w = 0; w < subblock_w; w++) {
                        uint32_t index = w + index_subblock_w_offset + index_h_offset;
                        mul_tiles(dfb_x_id, dfb_x_id, index, index, dst0);
                    }

                    index_subblock_w_offset += subblock_w;
                }
                index_h_offset += block_w;
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, dfb_ex2pe_id);
            tile_regs_release();
            dfb_ex2pe.push_back(1);

            // If modifying this code, see the long comment at the first REDUCE_SCALAR
            // pack into dfb_ex_partial earlier in this kernel.
            // The sharded reader's "single-tile-overwrite trick" depends on
            // this pack also clearing every non-result datum of dfb_ex_partial
            // to exact zero (documented packer behavior for REDUCE_SCALAR).
            compute_kernel_lib::
                reduce<PoolType::SUM, ReduceDim::REDUCE_SCALAR, dfb_ex2pe_id, dfb_scaler_id, dfb_ex_partial_id>(
                    compute_kernel_lib::ReduceInputBlockShape::single());

            dfb_ex_partial.wait_front(1);
            if constexpr (is_mcast_sender and num_cores_per_mcast_group > 1) {
                compute_kernel_lib::reduce<
                    PoolType::SUM,
                    ReduceDim::REDUCE_SCALAR,
                    dfb_ex_external_id,
                    dfb_scaler_global_id,
                    dfb_ex_global_id,
                    compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
                    compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(
                    compute_kernel_lib::ReduceInputBlockShape::single());
                dfb_ex.reserve_back(1);
                dfb_ex.push_back(1);
            }

            // global reduce results

            dfb_eps.wait_front(1);
            dfb_ex_global.wait_front(1);
            dfb_ex2pe.reserve_back(1);

            // The row mask keeps the padding out of both sums, so this is already the variance over
            // the real rows; no back-correction needed.
            // (Var + eps)
            tile_regs_acquire();
            // fp32: reset both srcs so bf16 eps isn't read through the (x-Ex)^2 fp32 format (else garbage var+eps).
            if constexpr (enable_fp32_reconfig) {
                reconfig_data_format_srca(dfb_ex_global_id);
                reconfig_data_format_srcb(dfb_eps_id);
            }
            add_init(dfb_ex_global_id, dfb_eps_id);
            add_tiles(dfb_ex_global_id, dfb_eps_id, 0, 0, dst0);
            // 1/[sqrt(Var + eps)]
            rsqrt_tile_init<true>();
            rsqrt_tile<true>(dst0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, dfb_ex2pe_id);
            tile_regs_release();
            dfb_ex2pe.push_back(1);
            dfb_ex_global.pop_front(1);
            //  (x - Ex) * 1/[sqrt(Var + eps)]
            index_h_offset = 0;
            // fp32: reset both srcs so fp32 x/rstd aren't read through the (var+eps) bf16 dfb_eps format.
            if constexpr (enable_fp32_reconfig) {
                reconfig_data_format_srca(dfb_x_id);
                reconfig_data_format_srcb(dfb_ex2pe_id);
            }
            mul_bcast_scalar_init(dfb_x_id, dfb_ex2pe_id);

            dfb_ex2pe.wait_front(1);
            for (uint32_t i = 0; i < block_h; i++) {
                index_subblock_w_offset = 0;
                for (uint32_t j = 0; j < num_subblocks_w; j++) {
                    tile_regs_acquire();
                    for (uint32_t w = 0; w < subblock_w; w++) {
                        uint32_t index = w + index_subblock_w_offset;
                        mul_tiles_bcast_scalar(dfb_x_id, dfb_ex2pe_id, index, 0, w);
                    }
                    tile_regs_commit();
                    dfb_x.pop_front(subblock_w);
                    dfb_x.reserve_back(subblock_w);
                    tile_regs_wait();
                    for (uint32_t i = 0; i < subblock_w; i++) {
                        pack_tile(i, dfb_x_id);
                    }
                    dfb_x.push_back(subblock_w);
                    tile_regs_release();
                }
            }
            dfb_ex2pe.pop_front(1);
            dfb_x.wait_front(block_hw);
            //  add or copy with previous output results
            uint32_t block_w_curr = index_g_offset == (per_core_N - block_w_last) ? block_w_last : block_w;

            // if we are using negative mask, we are overlapping tilized in and out, otherwise they are 2 separate
            // buffers.
            if constexpr (use_negative_mask == false) {
                for (uint32_t w = 0; w < block_w_curr; ++w) {
                    index_h_offset = index_b_offset + index_g_offset;
                    uint32_t index_h1_offset = 0;

                    if (copy_or_add == true) {
                        copy_init(dfb_x_id);
                    } else {
                        add_init(dfb_out_id, dfb_x_id);
                    }

                    for (uint32_t i = 0; i < block_h; ++i) {
                        tile_regs_acquire();
                        uint32_t index_x = w + index_h1_offset;
                        uint32_t index = w + index_h_offset;

                        if (copy_or_add == true) {
                            copy_tile(dfb_x_id, index_x, dst0);
                        } else {
                            add_tiles(dfb_out_id, dfb_x_id, index, index_x, dst0);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_tile<true>(dst0, dfb_out_id, index);
                        tile_regs_release();

                        index_h_offset += per_core_N;
                        index_h1_offset += block_w;
                    }

                    // update group tile offset
                    if (index_block_w >= block_w_curr - 1) {
                        index_block_w = 0;

                        if (group_reset_index == num_groups_per_reset - 1) {
                            copy_or_add = true;

                            group_reset_index = 0;
                        } else {
                            copy_or_add = false;

                            group_reset_index += 1;
                        }
                    } else {
                        copy_or_add = true;
                        index_block_w += 1;
                    }
                }
            } else {
                // zero out values in cb_tilized_in input by multiplying with negative mask for the current group
                dfb_in_negative_mask.wait_front(block_w);
                reconfig_data_format_srcb(dfb_x_id, dfb_in_negative_mask_id);
                mul_bcast_rows_init(dfb_in_id, dfb_in_negative_mask_id);

                for (uint32_t w = 0; w < block_w_curr; w++) {
                    index_h_offset = index_b_offset + index_g_offset;
                    uint32_t index_h1_offset = 0;

                    for (uint32_t i = 0; i < block_h; i++) {
                        tile_regs_acquire();
                        uint32_t index_in = w + index_h_offset;
                        uint32_t index_mask = w;

                        mul_tiles_bcast_rows(dfb_in_id, dfb_in_negative_mask_id, index_in, index_mask, dst0);
                        tile_regs_commit();

                        tile_regs_wait();
                        pack_tile<true>(dst0, dfb_in_id, index_in);
                        tile_regs_release();

                        index_h_offset += per_core_N;
                    }
                }

                reconfig_data_format_srcb(dfb_in_negative_mask_id, dfb_x_id);
                add_init(dfb_in_id, dfb_x_id);
                // data in cb_x_id has valid data only for current group
                // cb_in_id has cleared data for that group
                // just add them together
                for (uint32_t w = 0; w < block_w_curr; ++w) {
                    index_h_offset = index_b_offset + index_g_offset;
                    uint32_t index_h1_offset = 0;

                    for (uint32_t i = 0; i < block_h; ++i) {
                        tile_regs_acquire();
                        uint32_t index_x = w + index_h1_offset;
                        uint32_t index = w + index_h_offset;

                        add_tiles(dfb_in_id, dfb_x_id, index, index_x, dst0);
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_tile<true>(dst0, dfb_in_id, index);
                        tile_regs_release();

                        index_h_offset += per_core_N;
                        index_h1_offset += block_w;
                    }
                }
                dfb_in_negative_mask.pop_front(block_w);
            }

            dfb_x.pop_front(block_hw);

            if constexpr (GROUP_SIZE_IS_POWER_OF_2) {
                if (row_offset == tile_width) {
                    index_g_offset += block_w;
                    row_offset = num_cols_per_group;

                } else {
                    index_g_offset += block_w_minus_one;
                    row_offset += num_cols_per_group;
                }
            } else if constexpr (GROUP_SIZE_SMALLER_THAN_TILE_W) {
                if (row_offset == tile_width) {
                    index_g_offset += block_w_minus_one;
                    row_offset = num_cols_per_group;

                } else if (row_offset > tile_width) {
                    index_g_offset += block_w_minus_one;
                    row_offset = row_offset + group_row_offset;

                } else {
                    row_offset += num_cols_per_group;
                }
            } else {
                if (row_offset > tile_width) {
                    index_g_offset += block_w_minus_one;
                    row_offset = row_offset - tile_w_minux_group_size;
                } else {
                    row_offset += num_cols_per_group;
                    index_g_offset += block_w_minus_two;
                }
            }
        }
        index_b_offset += num_tiles_per_batch;
    }

    if constexpr (use_negative_mask == false) {
        dfb_out.push_back(per_core_MN);
        dfb_in.pop_front(per_core_MN);

    } else {
        // nothing, for the negative mask implementation, cb_in_id is the only cb in use, and it already has the data
        // required for the rest of kernel.
    }

    if constexpr (do_gamma) {
        index_h_offset = 0;
        if constexpr (use_negative_mask == false) {
            // fp32: reset both srcs so bf16 gamma isn't read through the normalization loop's fp32 format.
            if constexpr (enable_fp32_reconfig) {
                reconfig_data_format_srca(dfb_out_id);
                reconfig_data_format_srcb(dfb_gamma_id);
            }
            mul_bcast_rows_init(dfb_out_id, dfb_gamma_id);
            dfb_outgamma.reserve_back(per_core_MN);
            dfb_gamma.wait_front(per_core_N);
            for (uint32_t i = 0; i < per_core_M; ++i) {
                for (uint32_t j = 0; j < per_core_N; ++j) {
                    tile_regs_acquire();
                    uint32_t index = j + index_h_offset;
                    mul_tiles_bcast_rows(dfb_out_id, dfb_gamma_id, index, j, dst0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(dst0, dfb_outgamma_id);
                    tile_regs_release();
                }
                index_h_offset += per_core_N;
            }

            dfb_outgamma.push_back(per_core_MN);
            dfb_out.pop_front(per_core_MN);
            dfb_outgamma.wait_front(per_core_MN);
        } else {
            // cb in has data required for gamma, so we do it inplace
            // fp32: see non-negative-mask branch above.
            if constexpr (enable_fp32_reconfig) {
                reconfig_data_format_srca(dfb_in_id);
                reconfig_data_format_srcb(dfb_gamma_id);
            }
            mul_bcast_rows_init(dfb_in_id, dfb_gamma_id);
            dfb_gamma.wait_front(per_core_N);
            dfb_in.wait_front(per_core_MN);
            for (uint32_t i = 0; i < per_core_M; i++) {
                for (uint32_t j = 0; j < per_core_N; j++) {
                    tile_regs_acquire();
                    mul_tiles_bcast_rows(dfb_in_id, dfb_gamma_id, 0, j, dst0);
                    tile_regs_commit();
                    dfb_in.pop_front(1);
                    dfb_in.reserve_back(1);
                    tile_regs_wait();
                    pack_tile(dst0, dfb_in_id);
                    dfb_in.push_back(1);
                    tile_regs_release();
                }
            }
        }
    }

    if constexpr (do_beta) {
        if constexpr (use_negative_mask == false) {
            index_h_offset = 0;
            // fp32: reset both srcs so bf16 beta isn't read as fp32 (matters especially when do_gamma=false).
            if constexpr (enable_fp32_reconfig) {
                reconfig_data_format_srca(dfb_inbeta_id);
                reconfig_data_format_srcb(dfb_beta_id);
            }
            add_bcast_rows_init(dfb_inbeta_id, dfb_beta_id);
            dfb_outbeta.reserve_back(per_core_MN);
            dfb_beta.wait_front(per_core_N);
            for (uint32_t i = 0; i < per_core_M; ++i) {
                for (uint32_t j = 0; j < per_core_N; ++j) {
                    tile_regs_acquire();
                    uint32_t index = j + index_h_offset;
                    add_tiles_bcast_rows(dfb_inbeta_id, dfb_beta_id, index, j, dst0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(dst0, dfb_outbeta_id);
                    tile_regs_release();
                }
                index_h_offset += per_core_N;
            }
            dfb_outbeta.push_back(per_core_MN);
            dfb_inbeta.pop_front(per_core_MN);
            dfb_outbeta.wait_front(per_core_MN);
        } else {
            // cb_in_id has data required for beta, so we do it inplace
            // fp32: see non-negative-mask branch above.
            if constexpr (enable_fp32_reconfig) {
                reconfig_data_format_srca(dfb_in_id);
                reconfig_data_format_srcb(dfb_beta_id);
            }
            add_bcast_rows_init(dfb_in_id, dfb_beta_id);
            dfb_beta.wait_front(per_core_N);
            dfb_in.wait_front(per_core_MN);
            for (uint32_t i = 0; i < per_core_M; i++) {
                for (uint32_t j = 0; j < per_core_N; j++) {
                    tile_regs_acquire();
                    add_tiles_bcast_rows(dfb_in_id, dfb_beta_id, 0, j, dst0);
                    tile_regs_commit();
                    dfb_in.pop_front(1);
                    dfb_in.reserve_back(1);
                    tile_regs_wait();
                    pack_tile(dst0, dfb_in_id);
                    dfb_in.push_back(1);
                    tile_regs_release();
                }
            }
        }
    }

#ifdef UNTILIZE_OUT
    // untilize - DEST capacity auto-detected
    compute_kernel_lib::untilize<
        per_core_N,
        dfb_untilize_in_id,
        dfb_untilize_out_id,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitUpfront,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(per_core_M);
#endif
}
