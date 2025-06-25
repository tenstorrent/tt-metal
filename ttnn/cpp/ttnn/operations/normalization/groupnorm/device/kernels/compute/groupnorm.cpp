// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_SCALAR

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/layernorm.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/matmul.h"

namespace NAMESPACE {
void MAIN {
    // clang-format off
    // Definitions
    //   block_h: This the length of the row we wish to processes in terms of tiles
    //
    //   out_block_...: This is the length of our Circular Buffer, sometimes the length of out tensors(block_h) are larger than L1 space, so we
    //   have to process chunks of this data at a time
    //   this chunk is called an out_block
    //
    //   num_out_blocks: This is the number of chunks specified by the use, such that a CBs (length defined by out_block) fit in L1
    //   (Users should minimize the number of num_out_blocks for better perf)
    //
    //   ...normal:  If num_out_blocks evenly divides block_h, then all chunks are the size normal
    //
    //   ...last: If num_out_blocks does not divides block_h, the leftovers are put into a chunk of length last
    //
    //   sender: This refers to a core that does aggregation calculations
    //   for the group of cores
    //
    //   receiver: This the cores that receive the aggregated results from sender, they only do
    //   local computations that they send to the sender for final aggregation
    //
    // This is a high level desciption of the stages of this kernel, tags will be added to show where in the code each
    // stage starts and ends
    //
    // Batch Loop:
    //   Group Loop:
    //     This is the process which repeats for every group
    //     Average Calc: E[x]
    //       Local Reduce:
    //           First we apply an input mask
    //           This is where we sum up our core's subtensor
    //           After summing up, we pass our scalar tile to cb_ex_partial
    //           The reader kernels then aggregate all of the local scalars into a single tile
    //       Global Reduce:
    //           This single tile (cb_ex_external) is a tile that contains each partial reduce from all the other cores
    //           Only the core designated as the sender reduces this tile to produce the global scalar reduce value.
    //           It's reader core the sends this data out to all other cores as cb_ex_global
    //
    //     Variance Calc: ∑(x-E[x])^2
    //     This follows the same pattern as the average calculation
    //       Local Reduce:
    //           First we subtract each value from our core's subtensor by the average value
    //           We next apply our input mask to zero our the values we wish to ignore
    //           Next we square our residuals to obtain the squared residuals
    //           After summing up, we pass our scalar tile to cb_ex2_partial
    //           The reader kernels then aggregate all of the local scalars into a single tile
    //       Global Reduce:
    //           This single tile (cb_ex_external) is a tile that contains each partial reduce from all the other cores
    //           Only the core designated as the sender reduces this tile to produce the global scalar reduce value.
    //           It's reader core the sends this data out to all other cores as cb_ex2_global
    //
    //     cb_ex2pe Calculation:
    //       First we add cb_ex2_global with cb_eps
    //       Then we take the sqrt
    //       Lastly we take the reciprocal and he have the denominator of our calculation
    //     Final Val Calc:
    //       First we subtract each value from our core's subtensor by the average value
    //       We next apply our input mask to zero our the values we wish to ignore
    //       Next we multiply our residual with our denominator
    //       Optional Gamma:
    //           We multiply this value to gamma
    //       Optional Beta:
    //           We add beta to this value
    //
    // We are now done! Nice
    //   To look at where the code starts and stops seach for
    //   Start LABEL or End Label
    //   Ex: Start Local Reduce or End Local Reduce
    // clang-format on
    constexpr uint32_t is_mcast_sender = get_compile_time_arg_val(0);
    constexpr uint32_t do_gamma = get_compile_time_arg_val(1);
    constexpr uint32_t do_beta = get_compile_time_arg_val(2);
    constexpr uint32_t num_cores_per_mcast_group = get_compile_time_arg_val(3);

    constexpr uint32_t batch = get_compile_time_arg_val(4);
    constexpr uint32_t group = get_compile_time_arg_val(5);

    constexpr uint32_t block_h = get_compile_time_arg_val(6);
    constexpr uint32_t block_w = get_compile_time_arg_val(7);
    constexpr uint32_t block_hw = get_compile_time_arg_val(8);

    constexpr uint32_t subblock_w = get_compile_time_arg_val(9);
    constexpr uint32_t num_subblocks_w = get_compile_time_arg_val(10);

    constexpr uint32_t per_core_M = get_compile_time_arg_val(11);
    constexpr uint32_t per_core_N = get_compile_time_arg_val(12);
    constexpr uint32_t per_core_MN = get_compile_time_arg_val(13);

    constexpr uint32_t per_core_N_tile_bytes = get_compile_time_arg_val(14);
    constexpr uint32_t num_groups_per_reset = get_compile_time_arg_val(15);

    constexpr uint32_t single_tile_size_bytes = get_compile_time_arg_val(16);
    constexpr uint32_t num_tiles_per_batch = get_compile_time_arg_val(17);

    constexpr uint32_t num_tiles_input_mask = get_compile_time_arg_val(18);
    constexpr uint32_t num_cols_per_group = get_compile_time_arg_val(19);

    constexpr uint32_t block_w_last = get_compile_time_arg_val(20);
    constexpr uint32_t GROUP_SIZE_IS_POWER_OF_2 = get_compile_time_arg_val(21);
    constexpr uint32_t GROUP_SIZE_SMALLER_THAN_TILE_W = get_compile_time_arg_val(22);
    constexpr uint32_t group_row_offset = get_compile_time_arg_val(23);
    constexpr uint32_t num_out_blocks = get_compile_time_arg_val(24);

    constexpr uint32_t block_w_minus_one = block_w - 1;
    constexpr uint32_t block_w_minus_two = block_w - 2;
    constexpr uint32_t tile_w_minux_group_size = TILE_WIDTH - num_cols_per_group;

    // dst regs
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t scaler0 = 0;

    // input cbs
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in = tt::CBIndex::c_29;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;
    constexpr uint32_t cb_scaler_global = tt::CBIndex::c_4;
    constexpr uint32_t cb_eps = tt::CBIndex::c_3;
    constexpr uint32_t cb_gamma = tt::CBIndex::c_5;
    constexpr uint32_t cb_beta = tt::CBIndex::c_6;
    constexpr uint32_t cb_input_mask = tt::CBIndex::c_28;

    // interm cbs
    constexpr uint32_t cb_repack = tt::CBIndex::c_26;
    constexpr uint32_t cb_repack_out = tt::CBIndex::c_31;
    constexpr uint32_t cb_x = tt::CBIndex::c_24;
    constexpr uint32_t cb_xmm = tt::CBIndex::c_25;
    constexpr uint32_t cb_ex_partial = tt::CBIndex::c_8;
    constexpr uint32_t cb_ex2_partial = tt::CBIndex::c_21;
    constexpr uint32_t cb_ex = tt::CBIndex::c_9;
    constexpr uint32_t cb_ex2 = tt::CBIndex::c_13;
    constexpr uint32_t cb_ex_external = tt::CBIndex::c_10;
    constexpr uint32_t cb_ex_global = tt::CBIndex::c_15;
    constexpr uint32_t cb_ex2_global = tt::CBIndex::c_14;
    constexpr uint32_t cb_ex2pe = tt::CBIndex::c_27;

    // interm cbs reuse
    constexpr uint32_t cb_fusion = cb_xmm;
    constexpr uint32_t cb_reread_out = tt::CBIndex::c_23;
    constexpr uint32_t cb_reread_write_out = tt::CBIndex::c_22;

    // output cb
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
#ifdef UNTILIZE_OUT
    constexpr uint32_t cb_out = tt::CBIndex::c_30;
#else
    constexpr uint32_t cb_out = (do_gamma or do_beta)
                                    ? (((do_gamma and not do_beta) or (not do_gamma and do_beta)) ? cb_in : cb_out0)
                                    : cb_out0;
#endif

    // tile offset
    uint32_t index_subblock_w_offset = 0;
    uint32_t index_h_offset = 0;
    uint32_t index_w_offset = 0;
    uint32_t index_b_offset = 0;
    uint32_t index_g_offset = 0;
    uint32_t row_offset = num_cols_per_group;
    // data offset
    uint32_t num_datum_per_row_offeset = 0;
    // inplace out cbs
    bool copy_or_add = true;
    bool reset_index = false;
    uint32_t group_reset_index = 0;
    uint32_t index_block_w = 0;
    uint32_t output_tile_index = 0;
    bool apply_gamma_beta[block_w];
    constexpr uint32_t data_per_core_N_per_group = (per_core_N * TILE_WIDTH / group);

#ifdef UNTILIZE_OUT
    constexpr int cb_outgamma = cb_in;
    constexpr int cb_inbeta = do_gamma ? cb_outgamma : cb_out;
    constexpr int cb_outbeta = do_gamma ? cb_out : cb_in;
    constexpr int cb_untilize_in = (do_gamma and not do_beta) ? cb_outgamma : do_beta ? cb_outbeta : cb_out;
    constexpr int cb_untilize_out =
#ifdef READER_REPACK
        cb_repack_out;
#else
        cb_out0;
#endif
#else
    constexpr int cb_outgamma = do_beta ? cb_in : cb_out0;
    constexpr int cb_inbeta = do_gamma ? cb_outgamma : cb_reread_write_out;
    constexpr int cb_outbeta = cb_out0;
#endif

// tilize input from RM to tile layout
#ifdef TILIZE_IN
    binary_op_init_common(cb_in0, cb_in0, cb_in);
// tilize in0 -> in
#ifdef READER_REPACK
    constexpr uint32_t cb_in_rm = cb_repack;
#else
    constexpr uint32_t cb_in_rm = cb_in0;
#endif
    tilize_init(cb_in_rm, per_core_N, cb_in);
    for (uint32_t m = 0; m < per_core_M; ++m) {
#ifdef READER_REPACK
        cb_wait_front(cb_in_rm, per_core_N);
#endif
        cb_reserve_back(cb_in, per_core_N);
        tilize_block(cb_in_rm, per_core_N, cb_in);
        cb_push_back(cb_in, per_core_N);
        cb_pop_front(cb_in_rm, per_core_N);
    }
    tilize_uninit(cb_in_rm, cb_in);
    cb_wait_front(cb_in, per_core_MN);
#else
    binary_op_init_common(cb_in0, cb_input_mask, cb_x);
#endif

    index_b_offset = 0;
    constexpr uint32_t out_block_h_normal = block_h / num_out_blocks;
    uint32_t out_block_hw_normal = out_block_h_normal * block_w;
    uint32_t num_out_blocks_padded = num_out_blocks;
    uint32_t extra_out_block = false;
    uint32_t out_block_h_last = out_block_h_normal;
    uint32_t out_block_hw_last = out_block_hw_normal;
    if constexpr (block_h % num_out_blocks != 0) {
        extra_out_block = true;
        num_out_blocks_padded++;
        out_block_h_last = (block_h % num_out_blocks);
        out_block_hw_last = out_block_h_last * block_w;
    }
    uint32_t cb_ex_external_tiles_required =
        num_out_blocks_padded * num_cores_per_mcast_group * 16 / single_tile_size_bytes;
    if ((num_out_blocks_padded * num_cores_per_mcast_group * 16) % single_tile_size_bytes) {
        cb_ex_external_tiles_required++;
    }

    // Start Batch Loop
    for (uint32_t b = 0; b < batch; ++b) {
        index_g_offset = 0;

        row_offset = num_cols_per_group;
        copy_or_add = true;
        reset_index = false;
        group_reset_index = 0;
        index_block_w = 0;
        output_tile_index = 0;

        // Start Group Loop
        for (uint32_t g = 0; g < group; ++g) {
            // Start Average Calc
            // Start Local Reduce
            cb_wait_front(cb_input_mask, block_w);
            for (uint32_t out_block_index = 0; out_block_index < num_out_blocks_padded; out_block_index++) {
                uint32_t out_block_h_actual, out_block_hw_actual;
                if (extra_out_block && (out_block_index == (num_out_blocks_padded - 1))) {
                    out_block_h_actual = out_block_h_last;
                    out_block_hw_actual = out_block_hw_last;
                } else {
                    out_block_h_actual = out_block_h_normal;
                    out_block_hw_actual = out_block_hw_normal;
                }
                cb_wait_front(cb_in0, out_block_hw_normal);

                index_h_offset = 0;
                reconfig_data_format_srcb(cb_in0, cb_input_mask);
                // mask input
                mul_tiles_init(cb_in0, cb_input_mask);
                cb_reserve_back(cb_x, out_block_hw_normal);
                for (uint32_t i = 0; i < out_block_h_actual; ++i) {
                    index_subblock_w_offset = 0;
                    for (uint32_t j = 0; j < num_subblocks_w; ++j) {
                        tile_regs_acquire();
                        for (uint32_t w = 0; w < subblock_w; ++w) {
                            uint32_t index = w + index_subblock_w_offset + index_h_offset;
                            uint32_t index_mask = w + index_subblock_w_offset;
#ifdef TILIZE_IN
                        mul_tiles(cb_in, cb_input_mask, index, index_mask, w);
#else
                            mul_tiles(cb_in0, cb_input_mask, index, index_mask, w);
#endif
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t i = 0; i < subblock_w; ++i) {
                            pack_tile(i, cb_x);
                        }
                        tile_regs_release();
                        index_subblock_w_offset += subblock_w;
                    }
                    index_h_offset += block_w;
                }
#ifdef TILIZE_IN
                cb_pop_front(cb_in, out_block_hw_actual);
#else
                cb_pop_front(cb_in0, out_block_hw_normal);
#endif
                cb_push_back(cb_x, out_block_hw_normal);
                reconfig_data_format_srcb(cb_input_mask, cb_scaler);

                // Partial/E[x]
                index_h_offset = 0;
                reduce_init(cb_x, cb_scaler, cb_ex_partial);
                cb_reserve_back(cb_ex_partial, 1);
                tile_regs_acquire();
                cb_wait_front(cb_scaler, 1);
                cb_wait_front(cb_x, out_block_hw_normal);

                for (uint32_t h = 0; h < out_block_h_actual; ++h) {
                    for (uint32_t w = 0; w < block_w; ++w) {
                        uint32_t index = index_h_offset + w;
                        reduce_tile(cb_x, cb_scaler, index, scaler0, dst0);
                    }
                    index_h_offset += block_w;
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(dst0, cb_ex_partial);
                tile_regs_release();
                cb_pop_front(cb_x, out_block_hw_normal);
                cb_push_back(cb_ex_partial, 1);
                reduce_uninit();

                cb_wait_front(cb_ex_partial, 1);
            }
            // End Local Redcue
            // Start Global Reduce
            if constexpr (is_mcast_sender) {
                reduce_init(cb_ex_external, cb_scaler_global, cb_ex_global);
                cb_reserve_back(cb_ex_global, 1);
                if (num_cores_per_mcast_group > 1) {
                    cb_reserve_back(cb_ex, 1);
                }
                tile_regs_acquire();
                cb_wait_front(cb_scaler_global, 1);
                cb_wait_front(cb_ex_external, cb_ex_external_tiles_required);
                for (uint32_t external_i = 0; external_i < cb_ex_external_tiles_required; external_i++) {
                    reduce_tile(cb_ex_external, cb_scaler_global, external_i, scaler0, dst0);
                }
                cb_pop_front(cb_ex_external, cb_ex_external_tiles_required);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(dst0, cb_ex_global);
                tile_regs_release();
                reduce_uninit();
                cb_push_back(cb_ex_global, 1);
                if (num_cores_per_mcast_group > 1) {
                    cb_push_back(cb_ex, 1);
                }
            }
            // End Global Reduce
            // End Average Calc

            // Start Variance Calc
            // Start Local Reduce
            for (uint32_t out_block_index = 0; out_block_index < num_out_blocks_padded; out_block_index++) {
                uint32_t out_block_h_actual, out_block_hw_actual;
                if (extra_out_block && (out_block_index == (num_out_blocks_padded - 1))) {
                    out_block_h_actual = out_block_h_last;
                    out_block_hw_actual = out_block_hw_last;
                } else {
                    out_block_h_actual = out_block_h_normal;
                    out_block_hw_actual = out_block_hw_normal;
                }

                cb_wait_front(cb_in0, out_block_hw_normal);
                // x - E[x]
                sub_tiles_bcast_scalar_init_short(cb_in0, cb_ex_global);

                cb_reserve_back(cb_xmm, out_block_hw_normal);
                cb_wait_front(cb_ex_global, 1);
                for (uint32_t i = 0; i < out_block_h_actual; i++) {
                    index_subblock_w_offset = 0;
                    for (uint32_t j = 0; j < num_subblocks_w; j++) {
                        tile_regs_acquire();
                        for (uint32_t w = 0; w < subblock_w; w++) {
                            uint32_t index = w + index_subblock_w_offset;
                            sub_tiles_bcast_scalar(cb_in0, cb_ex_global, index, 0, w);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t i = 0; i < subblock_w; i++) {
                            pack_tile(i, cb_xmm);
                        }
                        tile_regs_release();
                        index_subblock_w_offset += subblock_w;
                    }
                    cb_pop_front(cb_in0, block_w);
                }
                if (extra_out_block && (out_block_index == (num_out_blocks_padded - 1))) {
                    cb_pop_front(cb_in0, out_block_hw_normal - out_block_hw_last);
                }
                cb_push_back(cb_xmm, out_block_hw_normal);

                // zero out the garbage values by mult mask again
                reconfig_data_format_srcb(cb_ex_global, cb_input_mask);
                mul_tiles_init(cb_xmm, cb_input_mask);
                cb_reserve_back(cb_x, out_block_hw_normal);
                cb_wait_front(cb_xmm, out_block_hw_normal);
                for (uint32_t i = 0; i < out_block_h_actual; i++) {
                    index_subblock_w_offset = 0;
                    for (uint32_t j = 0; j < num_subblocks_w; ++j) {
                        tile_regs_acquire();
                        for (uint32_t w = 0; w < subblock_w; ++w) {
                            uint32_t index = w + index_subblock_w_offset;
                            uint32_t index_mask = index;
                            mul_tiles(cb_xmm, cb_input_mask, index, index_mask, w);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t i = 0; i < subblock_w; ++i) {
                            pack_tile(i, cb_x);
                        }
                        tile_regs_release();
                        index_subblock_w_offset += subblock_w;
                    }
                    cb_pop_front(cb_xmm, block_w);
                }
                if (extra_out_block && (out_block_index == (num_out_blocks_padded - 1))) {
                    cb_pop_front(cb_xmm, out_block_hw_normal - out_block_hw_last);
                }
                cb_push_back(cb_x, out_block_hw_normal);

                reconfig_data_format_srcb(cb_input_mask, cb_x);
                // (x - E[x])^2
                index_h_offset = 0;
                mul_tiles_init(cb_x, cb_x);
                cb_reserve_back(cb_xmm, out_block_hw_normal);
                cb_wait_front(cb_x, out_block_hw_normal);
                for (uint32_t i = 0; i < out_block_h_actual; i++) {
                    index_subblock_w_offset = 0;
                    for (uint32_t j = 0; j < num_subblocks_w; j++) {
                        tile_regs_acquire();
                        for (uint32_t w = 0; w < subblock_w; w++) {
                            uint32_t index = w + index_subblock_w_offset + index_h_offset;
                            mul_tiles(cb_x, cb_x, index, index, w);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t i = 0; i < subblock_w; i++) {
                            pack_tile(i, cb_xmm);
                        }
                        tile_regs_release();
                        index_subblock_w_offset += subblock_w;
                    }
                    index_h_offset += block_w;
                }
                cb_pop_front(cb_x, out_block_hw_normal);
                cb_push_back(cb_xmm, out_block_hw_normal);

                // Partial-Var(x)
                index_h_offset = 0;
                reduce_init(cb_xmm, cb_scaler, cb_ex2_partial);
                cb_reserve_back(cb_ex2_partial, 1);
                tile_regs_acquire();
                cb_wait_front(cb_xmm, out_block_hw_normal);
                cb_wait_front(cb_scaler, 1);  // TODO DELETE THIS
                for (uint32_t h = 0; h < out_block_h_actual; ++h) {
                    for (uint32_t w = 0; w < block_w; ++w) {
                        uint32_t index = index_h_offset + w;
                        reduce_tile(cb_xmm, cb_scaler, index, scaler0, dst0);
                    }
                    index_h_offset += block_w;
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(dst0, cb_ex2_partial);
                tile_regs_release();
                cb_push_back(cb_ex2_partial, 1);
                cb_pop_front(cb_xmm, out_block_hw_normal);
                reduce_uninit();
            }
            // End Local Reduce
            // Start Global Reduce
            if constexpr (is_mcast_sender) {
                reduce_init(cb_ex_external, cb_scaler_global, cb_ex2_global);
                cb_reserve_back(cb_ex2_global, 1);
                if (num_cores_per_mcast_group > 1) {
                    cb_reserve_back(cb_ex2, 1);
                }
                tile_regs_acquire();
                cb_wait_front(cb_scaler_global, 1);
                cb_wait_front(cb_ex_external, cb_ex_external_tiles_required);  // TODO DELETE THIS AND ADD POP
                for (uint32_t external_i = 0; external_i < cb_ex_external_tiles_required; external_i++) {
                    reduce_tile(cb_ex_external, cb_scaler_global, external_i, scaler0, dst0);
                }
                cb_pop_front(cb_ex_external, cb_ex_external_tiles_required);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(dst0, cb_ex2_global);
                tile_regs_release();
                reduce_uninit();
                cb_push_back(cb_ex2_global, 1);
                if (num_cores_per_mcast_group > 1) {
                    cb_push_back(cb_ex2, 1);
                }
            }
            // End Global Reduce

            // Start Variance Calc
            //  global reduce results
            cb_wait_front(cb_eps, 1);
            cb_wait_front(cb_ex2_global, 1);
            cb_reserve_back(cb_ex2pe, 1);
            // (Var + eps)
            tile_regs_acquire();
            add_tiles_init(cb_ex2_global, cb_eps);
            add_tiles(cb_ex2_global, cb_eps, 0, 0, dst0);
            tile_regs_wait();
            // sqrt(Var + eps)
            sqrt_tile_init();
            sqrt_tile(dst0);
            tile_regs_wait();
            // 1/[sqrt(Var + eps)]
            recip_tile_init();
            recip_tile(dst0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_ex2pe);
            tile_regs_release();
            cb_push_back(cb_ex2pe, 1);
            cb_pop_front(cb_ex2_global, 1);
            // End Variance Calc

            bool start_copy_or_add = copy_or_add;
            uint32_t start_group_reset_index = group_reset_index;
            uint32_t start_index_block_w = index_block_w;

            uint32_t out_block_h_offset = 0;
            // Start Final Val Calc
            for (uint32_t out_block_index = 0; out_block_index < num_out_blocks_padded; out_block_index++) {
                uint32_t out_block_h_actual, out_block_hw_actual;
                if (extra_out_block && (out_block_index == (num_out_blocks_padded - 1))) {
                    out_block_h_actual = out_block_h_last;
                    out_block_hw_actual = out_block_hw_last;
                } else {
                    out_block_h_actual = out_block_h_normal;
                    out_block_hw_actual = out_block_hw_normal;
                }

                cb_wait_front(cb_in0, out_block_hw_normal);
                // x - E[x]
                sub_tiles_bcast_scalar_init_short(cb_in0, cb_ex_global);
                cb_reserve_back(cb_xmm, out_block_hw_normal);
                cb_wait_front(cb_ex_global, 1);
                for (uint32_t i = 0; i < out_block_h_actual; i++) {
                    index_subblock_w_offset = 0;
                    for (uint32_t j = 0; j < num_subblocks_w; j++) {
                        tile_regs_acquire();
                        for (uint32_t w = 0; w < subblock_w; w++) {
                            uint32_t index = w + index_subblock_w_offset;
                            sub_tiles_bcast_scalar(cb_in0, cb_ex_global, index, 0, w);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t i = 0; i < subblock_w; i++) {
                            pack_tile(i, cb_xmm);
                        }
                        tile_regs_release();
                        index_subblock_w_offset += subblock_w;
                    }
                    cb_pop_front(cb_in0, block_w);
                }
                if (extra_out_block && (out_block_index == (num_out_blocks_padded - 1))) {
                    cb_pop_front(cb_in0, out_block_hw_normal - out_block_hw_last);
                }
                cb_push_back(cb_xmm, out_block_hw_normal);

                // zero out the garbage values by mult mask again
                reconfig_data_format_srcb(cb_ex_global, cb_input_mask);
                mul_tiles_init(cb_xmm, cb_input_mask);
                cb_reserve_back(cb_x, out_block_hw_normal);
                cb_wait_front(cb_xmm, out_block_hw_normal);
                for (uint32_t i = 0; i < out_block_h_actual; i++) {
                    index_subblock_w_offset = 0;
                    for (uint32_t j = 0; j < num_subblocks_w; ++j) {
                        tile_regs_acquire();
                        for (uint32_t w = 0; w < subblock_w; ++w) {
                            uint32_t index = w + index_subblock_w_offset;
                            uint32_t index_mask = index;
                            mul_tiles(cb_xmm, cb_input_mask, index, index_mask, w);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t i = 0; i < subblock_w; ++i) {
                            pack_tile(i, cb_x);
                        }
                        tile_regs_release();
                        index_subblock_w_offset += subblock_w;
                    }
                    cb_pop_front(cb_xmm, block_w);
                }
                if (extra_out_block && (out_block_index == (num_out_blocks_padded - 1))) {
                    cb_pop_front(cb_xmm, out_block_hw_normal - out_block_hw_last);
                }
                cb_push_back(cb_x, out_block_hw_normal);
                reconfig_data_format_srcb(cb_input_mask, cb_x);

                // (x - Ex) * 1/[sqrt(Var + eps)]
                index_h_offset = 0;
                mul_tiles_bcast_scalar_init_short(cb_x, cb_ex2pe);
                cb_reserve_back(cb_xmm, out_block_hw_normal);
                cb_wait_front(cb_ex2pe, 1);
                cb_wait_front(cb_x, out_block_hw_normal);
                for (uint32_t i = 0; i < out_block_h_actual; i++) {
                    index_subblock_w_offset = 0;
                    for (uint32_t j = 0; j < num_subblocks_w; j++) {
                        tile_regs_acquire();
                        for (uint32_t w = 0; w < subblock_w; w++) {
                            uint32_t index = w + index_subblock_w_offset + index_h_offset;
                            mul_tiles_bcast_scalar(cb_x, cb_ex2pe, index, 0, w);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t i = 0; i < subblock_w; i++) {
                            pack_tile(i, cb_xmm);
                        }
                        tile_regs_release();
                        index_subblock_w_offset += subblock_w;
                    }
                    index_h_offset += block_w;
                }
                cb_pop_front(cb_x, out_block_hw_normal);
                cb_push_back(cb_xmm, out_block_hw_normal);
                cb_wait_front(cb_xmm, out_block_hw_normal);

                copy_or_add = start_copy_or_add;
                group_reset_index = start_group_reset_index;
                index_block_w = start_index_block_w;

                // add or copy with previous output results
                uint32_t block_w_curr = index_g_offset == (per_core_N - block_w_last) ? block_w_last : block_w;

                cb_wait_front(cb_reread_out, out_block_hw_normal);
                cb_reserve_back(cb_reread_write_out, out_block_hw_normal);
                for (uint32_t w = 0; w < block_w_curr; ++w) {
                    uint32_t index_h_offset = 0;
                    uint32_t index_h1_offset = 0;

                    if (copy_or_add == true) {
                        copy_tile_init(cb_xmm);
                    } else {
                        add_tiles_init(cb_reread_out, cb_xmm);
                    }

                    for (uint32_t i = 0; i < out_block_h_actual; ++i) {
                        tile_regs_acquire();
                        uint32_t index_reread_out = w + index_h_offset;
                        uint32_t index_xmm = w + index_h1_offset;

                        if (copy_or_add == true) {
                            copy_tile(cb_xmm, index_xmm, dst0);
                        } else {
                            add_tiles(cb_reread_out, cb_xmm, index_reread_out, index_xmm, dst0);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_tile<true>(dst0, cb_reread_write_out, index_reread_out);
                        tile_regs_release();

                        index_h_offset += block_w_curr;
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

                    bool is_past_end_of_group =
                        (((w + index_g_offset) + 1) * TILE_WIDTH) > ((g + 1) * data_per_core_N_per_group);
                    apply_gamma_beta[w] = !is_past_end_of_group;
                }
                cb_pop_front(cb_xmm, out_block_hw_normal);
                cb_pop_front(cb_reread_out, out_block_hw_normal);
                cb_push_back(cb_reread_write_out, out_block_hw_normal);

                // Start Optional Gamma:
                if constexpr (do_gamma) {
                    index_h_offset = 0;
                    cb_reserve_back(cb_outgamma, out_block_hw_normal);
                    cb_wait_front(cb_gamma, per_core_N);
                    cb_wait_front(cb_reread_write_out, out_block_hw_normal);
                    for (uint32_t i = 0; i < out_block_h_actual; ++i) {
                        for (uint32_t j = 0; j < block_w_curr; ++j) {
                            if (apply_gamma_beta[j]) {
                                mul_bcast_rows_init_short(cb_reread_write_out, cb_gamma);
                            } else {
                                copy_tile_init(cb_reread_write_out);
                            }
                            tile_regs_acquire();
                            uint32_t index = j + index_h_offset;
                            uint32_t index_gamma = j + index_g_offset;
                            if (apply_gamma_beta[j]) {
                                mul_tiles_bcast_rows(cb_reread_write_out, cb_gamma, index, index_gamma, dst0);
                            } else {
                                copy_tile(cb_reread_write_out, index, dst0);
                            }
                            tile_regs_commit();
                            tile_regs_wait();
                            pack_tile(dst0, cb_outgamma);
                            tile_regs_release();
                        }
                        index_h_offset += block_w_curr;
                    }
                    cb_push_back(cb_outgamma, out_block_hw_normal);
                    cb_pop_front(cb_reread_write_out, out_block_hw_normal);
                    cb_wait_front(cb_outgamma, out_block_hw_normal);
                }
                // End Optional Gamma
                //
                // Start Optional Beta
                if constexpr (do_beta) {
                    index_h_offset = 0;
                    cb_reserve_back(cb_outbeta, out_block_hw_normal);
                    cb_wait_front(cb_beta, per_core_N);
                    for (uint32_t i = 0; i < out_block_h_actual; ++i) {
                        for (uint32_t j = 0; j < block_w_curr; ++j) {
                            if (apply_gamma_beta[j]) {
                                add_bcast_rows_init_short(cb_inbeta, cb_beta);
                            } else {
                                copy_tile_init(cb_inbeta);
                            }
                            tile_regs_acquire();
                            uint32_t index = j + index_h_offset;
                            uint32_t index_beta = j + index_g_offset;
                            if (apply_gamma_beta[j]) {
                                add_tiles_bcast_rows(cb_inbeta, cb_beta, index, index_beta, dst0);
                            } else {
                                copy_tile(cb_inbeta, index, dst0);
                            }
                            tile_regs_commit();
                            tile_regs_wait();
                            pack_tile(dst0, cb_outbeta);
                            tile_regs_release();
                        }
                        index_h_offset += block_w_curr;
                    }
                    cb_push_back(cb_outbeta, out_block_hw_normal);
                    cb_pop_front(cb_inbeta, out_block_hw_normal);
                    cb_wait_front(cb_outbeta, out_block_hw_normal);
                }
                // End Optional Beta

#ifdef UNTILIZE_OUT
                // untilize
                untilize_init_short(cb_untilize_in);
                cb_wait_front(cb_untilize_in, per_core_MN);
                for (uint32_t m = 0; m < per_core_M; ++m) {
                    cb_reserve_back(cb_untilize_out, per_core_N);
                    untilize_block(cb_untilize_in, per_core_N, cb_untilize_out);
                    cb_push_back(cb_untilize_out, per_core_N);
                    cb_pop_front(cb_untilize_in, per_core_N);
                }
                untilize_uninit(cb_untilize_in);
#endif
            }
            // End Final Val Calc
            if constexpr (GROUP_SIZE_IS_POWER_OF_2) {
                if (row_offset == TILE_WIDTH) {
                    index_g_offset += block_w;
                    row_offset = num_cols_per_group;

                } else {
                    index_g_offset += block_w_minus_one;
                    row_offset += num_cols_per_group;
                }
            } else if constexpr (GROUP_SIZE_SMALLER_THAN_TILE_W) {
                if (row_offset == TILE_WIDTH) {
                    index_g_offset += block_w_minus_one;
                    row_offset = num_cols_per_group;

                } else if (row_offset > TILE_WIDTH) {
                    index_g_offset += block_w_minus_one;
                    row_offset = row_offset + group_row_offset;

                } else {
                    row_offset += num_cols_per_group;
                }
            } else {
                if (row_offset > TILE_WIDTH) {
                    index_g_offset += block_w_minus_one;
                    row_offset = row_offset - tile_w_minux_group_size;
                } else {
                    row_offset += num_cols_per_group;
                    index_g_offset += block_w_minus_two;
                }
            }
            cb_pop_front(cb_ex_global, 1);
            cb_pop_front(cb_ex2pe, 1);
            cb_pop_front(cb_input_mask, block_w);
        }
        // End Group Loop
        index_b_offset += num_tiles_per_batch;
    }
    // End Batch Loop
}
}  // namespace NAMESPACE
