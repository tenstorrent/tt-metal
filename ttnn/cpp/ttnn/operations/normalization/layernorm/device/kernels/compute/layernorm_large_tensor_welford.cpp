// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "compute_kernel_api.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/welford.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/rsqrt.h"
#include "compute_kernel_api/transpose_wh.h"
#include "ttnn/operations/normalization/kernel_util/compute/memory.h"

namespace NAMESPACE {

template <
    tt::CBIndex cb_in,
    tt::CBIndex cb_inb,
    tt::CBIndex cb_interm_pre_add,
    tt::CBIndex cb_ex,
    tt::CBIndex cb_ex2,
    uint32_t input_dst,
    uint32_t mean_dst,
    uint32_t var_dst,
    uint32_t Wt,
    uint32_t tile_width,
    uint32_t W,
    uint32_t blk>
void welford_fuse_pre_add(const std::array<uint32_t, W>& reciprocal_lut) {
    // The number of valid rows in the last tile in width dimension
    constexpr uint32_t last_tile_rows = W % tile_width;
    constexpr bool is_last_tile_full = (last_tile_rows == 0);

    uint32_t sample_idx = 0;

    tile_regs_acquire();
    welford_init();
    welford_save_state(mean_dst);
    tile_regs_commit();

    cb_reserve_back(cb_ex, 1);
    cb_reserve_back(cb_ex2, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_ex);
    pack_tile(mean_dst, cb_ex);
    pack_tile(var_dst, cb_ex2);
    tile_regs_release();
    cb_push_back(cb_ex, 1);
    cb_push_back(cb_ex2, 1);

    for (uint32_t wt = 0; wt < Wt; wt += blk) {
        // Fused pre-add
        reconfig_data_format(cb_in, cb_inb);
        add_tiles_init(cb_in, cb_inb);
        cb_wait_front(cb_in, blk);
        cb_wait_front(cb_inb, blk);
        tile_regs_acquire();
        for (uint32_t j = 0; j < blk; j++) {
            add_tiles(cb_in, cb_inb, j, j, j);
        }
        tile_regs_commit();
        cb_pop_front(cb_inb, blk);
        cb_pop_front(cb_in, blk);

        // Pack to intermediate CB (needed
        // to workaround transpose_wh_dest bug)
        pack_reconfig_data_format(cb_interm_pre_add);
        cb_reserve_back(cb_interm_pre_add, blk);
        tile_regs_wait();
        for (uint32_t j = 0; j < blk; j++) {
            pack_tile(j, cb_interm_pre_add);
        }
        tile_regs_release();
        cb_push_back(cb_interm_pre_add, blk);

        // Now run Welfords in these blk number of tiles
        cb_wait_front(cb_interm_pre_add, blk);
        cb_wait_front(cb_ex, 1);
        cb_wait_front(cb_ex2, 1);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_in, cb_ex);
        copy_tile_init(cb_ex);
        copy_tile(cb_ex, 0, mean_dst);
        reconfig_data_format_srca(cb_ex, cb_ex2);
        copy_tile_to_dst_init_short_with_dt(cb_ex, cb_ex2);
        copy_tile(cb_ex2, 0, var_dst);
        welford_restore_state(mean_dst);

        reconfig_data_format_srca(cb_ex2, cb_interm_pre_add);
        transpose_wh_init_short(cb_interm_pre_add);
        for (uint32_t j = 0; j < blk; j++) {
            // Welford's needs transposed input tile
            transpose_wh_tile(cb_interm_pre_add, j, input_dst);

            if constexpr (is_last_tile_full) {
                // All tiles can go through the faster call which does 32 rows
                welford_update<W>(input_dst, sample_idx, reciprocal_lut);
            } else {
                // If it is the end tile, do it differently
                if ((wt + j) == (Wt - 1)) {
                    welford_update<W>(input_dst, sample_idx, reciprocal_lut);
                } else {
                    welford_update_rows<W>(input_dst, sample_idx, 0, last_tile_rows, reciprocal_lut);
                }
            }
            sample_idx += tile_width;
        }
        welford_save_state(mean_dst);
        tile_regs_commit();
        cb_pop_front(cb_interm_pre_add, blk);
        cb_pop_front(cb_ex, 1);
        cb_pop_front(cb_ex2, 1);

        cb_reserve_back(cb_ex, 1);
        cb_reserve_back(cb_ex2, 1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_interm_pre_add, cb_ex);
        pack_tile(mean_dst, cb_ex);
        pack_tile(var_dst, cb_ex2);
        tile_regs_release();
        cb_push_back(cb_ex, 1);
        cb_push_back(cb_ex2, 1);
    }

    cb_wait_front(cb_ex, 1);
    cb_wait_front(cb_ex2, 1);
    tile_regs_acquire();
    copy_tile_init(cb_ex);
    copy_tile(cb_ex, 0, mean_dst);
    copy_tile_to_dst_init_short_with_dt(cb_ex, cb_ex2);
    copy_tile(cb_ex2, 0, var_dst);
    welford_restore_state(mean_dst);
    // Store the mean and variance to the destination registers
    welford_finalize_to_row<W>(mean_dst, W - 1, reciprocal_lut);
    tile_regs_commit();
    cb_pop_front(cb_ex, 1);
    cb_pop_front(cb_ex2, 1);
}

/* @brief: Welford's algorithm for no fused pre-add
 * @param: cb_in: input CB
 * @param: input_dst: input tile for Welford's algorithm
 * @param: mean_dst: mean tile for Welford's algorithm
 * @param: Wt: width of the input in tiles
 * @param: tile_width: width of each tile
 * @param: W: width of the input
 * @param: p_reciprocals: pointer to the reciprocal LUT
 */
template <tt::CBIndex cb_in, uint32_t input_dst, uint32_t mean_dst, uint32_t Wt, uint32_t tile_width, uint32_t W>
void welford_no_fuse_pre_add(const std::array<uint32_t, W>& reciprocal_lut) {
    // The number of valid rows in the last tile in width dimension
    constexpr uint32_t last_tile_rows = W % tile_width;
    constexpr bool is_last_tile_full = (last_tile_rows == 0);

    uint32_t sample_idx = 0;
    reconfig_data_format_srca(cb_in);
    transpose_wh_init_short(cb_in);
    tile_regs_acquire();
    welford_init();

    // Process all but the last tile
    for (uint32_t wt = 0; wt < (Wt - 1); ++wt) {
        cb_wait_front(cb_in, 1);
        // Welford's needs transposed input tile
        transpose_wh_tile(cb_in, 0, input_dst);
        welford_update<W>(input_dst, sample_idx, reciprocal_lut);

        // Pop the input
        cb_pop_front(cb_in, 1);
        sample_idx += tile_width;
    }

    // Process the last tile
    cb_wait_front(cb_in, 1);
    transpose_wh_tile(cb_in, 0, input_dst);

    if constexpr (is_last_tile_full) {
        welford_update<W>(input_dst, sample_idx, reciprocal_lut);
    } else {
        welford_update_rows<W>(input_dst, sample_idx, 0, last_tile_rows, reciprocal_lut);
    }

    // Store the mean and variance to the destination registers
    welford_finalize_to_row<W>(mean_dst, W - 1, reciprocal_lut);

    tile_regs_commit();
    cb_pop_front(cb_in, 1);
}

void MAIN {
    namespace kutil = norm::kernel_util;

    uint32_t NCHt = get_arg_val<uint32_t>(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);
    constexpr uint32_t do_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t do_beta = get_compile_time_arg_val(3);
    constexpr bool FLOAT32_DTYPE = get_compile_time_arg_val(4) == 1;
    constexpr uint32_t W = get_compile_time_arg_val(5);
    constexpr uint32_t tile_width = get_compile_time_arg_val(6);
    constexpr bool fuse_pre_add = static_cast<bool>(get_compile_time_arg_val(8));

    // Note that the entire W dimension must fit in the intermed0 CB for this kernel to be correct
    constexpr auto cb_eps = tt::CBIndex::c_3;     // single tile generated by the reader
    constexpr auto cb_in = tt::CBIndex::c_0;      // input x or a for fused pre-add (x=a+b)
    constexpr auto cb_inb = tt::CBIndex::c_1;     // input b for fused pre-add
    constexpr auto cb_out = tt::CBIndex::c_16;    // output
    constexpr auto cb_gamma = tt::CBIndex::c_5;
    constexpr auto cb_beta = tt::CBIndex::c_6;
    uint32_t cb_xmm = tt::CBIndex::c_24;                   // x - E[x]
    constexpr auto cb_ex = tt::CBIndex::c_18;              // E[x]
    constexpr auto cb_ex2 = tt::CBIndex::c_19;             // Var[x] = E[(x-E[x])^2]
    constexpr auto cb_ex2pe = tt::CBIndex::c_21;           // Var[x]+ε
    constexpr auto cb_fusion = tt::CBIndex::c_22;          // stream gamma/beta
    constexpr auto cb_interm_pre_add = tt::CBIndex::c_23;  // intermediate for fused pre-add
    constexpr auto cb_reciprocals = tt::CBIndex::c_25;     // Pre-computed reciprocals for Welford's algorithm

    constexpr uint32_t onetile = 1;

    // Initialize the hardware based on the first op
    // that will be done
    if constexpr (fuse_pre_add) {
        // Init for x = in + b
        binary_op_init_common(cb_in, cb_inb, cb_interm_pre_add);
    } else {
        // Init for transpose
        constexpr auto first_out_cb = cb_ex;
        unary_op_init_common(cb_in, first_out_cb);
    }

    cb_wait_front(cb_eps, onetile);     // comes from the reader

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t input_dst = 0;  // Input tile for Welford's algorithm
    constexpr uint32_t mean_dst = 1;   // Mean tile for Welford's
    constexpr uint32_t var_dst = 2;    // Variance tile for Welford's

    // Get pointer to the reciprocal LUT
    using recip_lut_t = std::array<uint32_t, W>;
    auto p_reciprocals = kutil::compute::memory::get_pointer_to_cb_data<recip_lut_t>(cb_reciprocals, 0);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // Depending on whether we need to fuse pre-add, the approach for welford is different.
        // So we move it to a separate function.
        if constexpr (fuse_pre_add) {
            welford_fuse_pre_add<
                cb_in,
                cb_inb,
                cb_interm_pre_add,
                cb_ex,
                cb_ex2,
                input_dst,
                mean_dst,
                var_dst,
                Wt,
                tile_width,
                W,
                blk>(*p_reciprocals);
        } else {
            welford_no_fuse_pre_add<cb_in, input_dst, mean_dst, Wt, tile_width, W>(*p_reciprocals);
        }
        // We should expect that either of the two would have have populated dst regs with mean and
        // variance in mean_dst and var_dst respectively.

        cb_reserve_back(cb_ex, onetile);
        cb_reserve_back(cb_ex2, onetile);
        tile_regs_wait();
        pack_reconfig_data_format(cb_ex);
        pack_tile(mean_dst, cb_ex);
        pack_tile(var_dst, cb_ex2);
        tile_regs_release();
        cb_push_back(cb_ex, onetile);
        cb_push_back(cb_ex2, onetile);

        // Transpose mean and variance back to
        // columns and pack back to CBs
        reconfig_data_format_srca(cb_ex);
        transpose_wh_init_short(cb_ex);

        cb_wait_front(cb_ex, onetile);
        cb_wait_front(cb_ex2, onetile);
        tile_regs_acquire();
        transpose_wh_tile(cb_ex, 0, mean_dst);
        transpose_wh_tile(cb_ex2, 0, var_dst);
        tile_regs_commit();
        cb_pop_front(cb_ex, onetile);
        cb_pop_front(cb_ex2, onetile);

        cb_reserve_back(cb_ex, onetile);
        cb_reserve_back(cb_ex2, onetile);
        tile_regs_wait();
        pack_reconfig_data_format(cb_ex);
        pack_tile(mean_dst, cb_ex);
        pack_reconfig_data_format(cb_ex2);
        pack_tile(var_dst, cb_ex2);
        tile_regs_release();
        cb_push_back(cb_ex, onetile);
        cb_push_back(cb_ex2, onetile);

        // =====================================
        // Calculate 1/(√(Var(X) + ε))
        // =====================================
        reconfig_data_format(cb_ex2, cb_eps);
        add_tiles_init(cb_ex2, cb_eps);

        cb_wait_front(cb_ex2, onetile);
        tile_regs_acquire();
        add_tiles(cb_ex2, cb_eps, 0, 0, dst0);
        rsqrt_tile_init();
        rsqrt_tile(dst0);
        tile_regs_commit();
        cb_pop_front(cb_ex2, onetile);

        cb_reserve_back(cb_ex2pe, onetile);
        tile_regs_wait();
        pack_tile(dst0, cb_ex2pe);
        tile_regs_release();
        cb_push_back(cb_ex2pe, onetile);

        // broadcasts the tile since cb_ex2pe is a column vector that contains the important data
        cb_wait_front(cb_ex2pe, onetile);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_ex2pe);
        unary_bcast_init<BroadcastType::COL>(cb_ex2pe, cb_ex2pe);
        unary_bcast<BroadcastType::COL>(cb_ex2pe, 0, dst0);
        cb_pop_front(cb_ex2pe, onetile);
        tile_regs_commit();

        cb_reserve_back(cb_ex2pe, onetile);
        tile_regs_wait();
        pack_tile(dst0, cb_ex2pe);
        tile_regs_release();
        cb_push_back(cb_ex2pe, onetile);

        // =====================================
        // Second pass over the input.
        // Computes the final value:
        //    x-E[x]
        //(---------------*𝛄)+ß
        //  √(Var(x)+ε)
        // =====================================
        cb_wait_front(cb_ex2pe, onetile);
        cb_wait_front(cb_ex, onetile);

        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            cb_wait_front(cb_in, blk);
            tile_regs_acquire();
            reconfig_data_format(cb_in, cb_ex);
            sub_bcast_cols_init_short(cb_in, cb_ex);
            // x-E[x]
            for (uint32_t j = 0; j < blk; j++) {
                sub_tiles_bcast_cols(cb_in, cb_ex, j, 0, j);
            }
            cb_pop_front(cb_in, blk);

            reconfig_data_format_srca(cb_in, cb_ex2pe);
            if constexpr (fuse_pre_add) {
                // Fuse in = in + b
                reconfig_data_format_srca(cb_ex2pe, cb_inb);
                binary_dest_reuse_tiles_init<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_inb);
                cb_wait_front(cb_inb, blk);
                for (uint32_t j = 0; j < blk; j++) {
                    binary_dest_reuse_tiles<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_inb, j, j);
                }
                cb_pop_front(cb_inb, blk);
                reconfig_data_format_srca(cb_inb, cb_ex2pe);
            }

            // Multiply by 1/(√(Var(X) + ε))
            binary_dest_reuse_tiles_init<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_ex2pe);
            for (uint32_t j = 0; j < blk; j++) {
                binary_dest_reuse_tiles<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_ex2pe, 0, j);
            }
            tile_regs_commit();

            if constexpr (!(do_gamma == 1 or do_beta == 1)) {
                cb_xmm = cb_out;
            }

            pack_reconfig_data_format(cb_xmm);
            cb_reserve_back(cb_xmm, blk);
            tile_regs_wait();
            for (uint32_t j = 0; j < blk; j++) {
                pack_tile(j, cb_xmm);
            }
            cb_push_back(cb_xmm, blk);
            tile_regs_release();

            if constexpr (do_gamma == 1) {
                // Multiply by gamma
                reconfig_data_format(cb_xmm, cb_gamma);
                tile_regs_acquire();
                cb_wait_front(cb_gamma, blk);
                cb_wait_front(cb_xmm, blk);
                mul_bcast_rows_init_short(cb_xmm, cb_gamma);
                for (uint32_t j = 0; j < blk; j++) {
                    mul_tiles_bcast_rows(cb_xmm, cb_gamma, j, j, j);
                }
                tile_regs_commit();
                cb_pop_front(cb_gamma, blk);
                cb_pop_front(cb_xmm, blk);

                if constexpr (!do_beta) {
                    pack_reconfig_data_format(cb_out);
                }
                tile_regs_wait();
                if constexpr (!do_beta) {
                    cb_reserve_back(cb_out, blk);
                    for (uint32_t j = 0; j < blk; j++) {
                        pack_tile(j, cb_out);
                    }
                    cb_push_back(cb_out, blk);
                } else {
                    cb_reserve_back(cb_xmm, blk);
                    for (uint32_t j = 0; j < blk; j++) {
                        pack_tile(j, cb_xmm);
                    }
                    cb_push_back(cb_xmm, blk);
                }
                tile_regs_release();
            }

            if constexpr (do_beta == 1) {
                // Add beta
                tile_regs_acquire();
                reconfig_data_format(cb_xmm, cb_beta);
                add_bcast_rows_init_short(cb_xmm, cb_beta);
                cb_wait_front(cb_xmm, blk);
                cb_wait_front(cb_beta, blk);
                for (uint32_t j = 0; j < blk; j++) {
                    add_tiles_bcast_rows(cb_xmm, cb_beta, j, j, j);
                }
                tile_regs_commit();
                cb_pop_front(cb_beta, blk);
                cb_pop_front(cb_xmm, blk);

                pack_reconfig_data_format(cb_out);
                cb_reserve_back(cb_out, blk);
                tile_regs_wait();
                for (uint32_t j = 0; j < blk; j++) {
                    pack_tile(j, cb_out);
                }
                tile_regs_release();
                cb_push_back(cb_out, blk);
            }
        }

        cb_xmm = tt::CBIndex::c_24;  // x minus mean
        cb_pop_front(cb_ex2pe, onetile);
        cb_pop_front(cb_ex, onetile);
    }  // NCHt loop
}
}  // namespace NAMESPACE
