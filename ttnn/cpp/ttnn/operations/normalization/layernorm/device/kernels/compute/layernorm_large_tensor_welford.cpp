// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/welford.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/transpose_wh.h"
#include "ttnn/operations/normalization/kernel_util/compute/memory.h"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "experimental/circular_buffer.h"

namespace generic = norm::kernel_util::generic;

template <
    uint32_t cb_in,
    uint32_t cb_inb,
    uint32_t cb_interm_pre_add,
    uint32_t cb_ex,
    uint32_t cb_ex2,
    uint32_t input_dst,
    uint32_t mean_dst,
    uint32_t var_dst,
    uint32_t Wt,
    uint32_t tile_width,
    uint32_t W,
    uint32_t blk>
void welford_fuse_pre_add(const std::array<uint32_t, W>& reciprocal_lut) {
    experimental::CircularBuffer cb_in_obj(cb_in);
    experimental::CircularBuffer cb_inb_obj(cb_inb);
    experimental::CircularBuffer cb_interm_pre_add_obj(cb_interm_pre_add);
    experimental::CircularBuffer cb_ex_obj(cb_ex);
    experimental::CircularBuffer cb_ex2_obj(cb_ex2);

    // The number of valid columns in the last tile in width dimension.
    // Because the Welford's llk is given transposed data, skip some rows when
    // we want to skip some columns from getting processed by layer_norm.
    // When last tile is full the value is 0 and is not used because full update is done.
    constexpr uint32_t last_tile_rows = W % tile_width;
    constexpr bool is_last_tile_full = (last_tile_rows == 0);

    uint32_t sample_idx = 0;

    tile_regs_acquire();
    welford_init();
    welford_save_state(mean_dst);
    tile_regs_commit();

    cb_ex_obj.reserve_back(1);
    cb_ex2_obj.reserve_back(1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_ex);
    pack_tile(mean_dst, cb_ex);
    pack_tile(var_dst, cb_ex2);
    tile_regs_release();
    cb_ex_obj.push_back(1);
    cb_ex2_obj.push_back(1);

    for (auto block : generic::blocks(Wt, blk)) {
        // Fused pre-add
        reconfig_data_format(cb_in, cb_inb);
        add_tiles_init(cb_in, cb_inb);
        cb_in_obj.wait_front(block.full_block_size());
        cb_inb_obj.wait_front(block.full_block_size());
        tile_regs_acquire();
        for (auto i : block.local()) {
            add_tiles(cb_in, cb_inb, i, i, i);
        }
        tile_regs_commit();
        cb_in_obj.pop_front(block.full_block_size());
        cb_inb_obj.pop_front(block.full_block_size());

        // Pack to intermediate CB (needed
        // to workaround transpose_wh_dest bug)
        pack_reconfig_data_format(cb_interm_pre_add);
        cb_interm_pre_add_obj.reserve_back(block.full_block_size());
        tile_regs_wait();
        for (auto i : block.local()) {
            pack_tile(i, cb_interm_pre_add);
        }
        tile_regs_release();
        cb_interm_pre_add_obj.push_back(block.full_block_size());

        // Now run Welfords in these blk number of tiles
        cb_interm_pre_add_obj.wait_front(block.full_block_size());
        cb_ex_obj.wait_front(1);
        cb_ex2_obj.wait_front(1);
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
        for (auto i : block.local()) {
            // Welford's needs transposed input tile
            transpose_wh_tile(cb_interm_pre_add, i, input_dst);

            if constexpr (is_last_tile_full) {
                // All tiles can go through the faster call which does 32 rows
                welford_update<W>(input_dst, sample_idx, reciprocal_lut);
            } else {
                // If it is the end tile, do it differently
                if ((block.start() + i) == (Wt - 1)) {
                    welford_update<W>(input_dst, sample_idx, reciprocal_lut);
                } else {
                    welford_update_rows<W>(input_dst, sample_idx, 0, last_tile_rows, reciprocal_lut);
                }
            }
            sample_idx += tile_width;
        }
        welford_save_state(mean_dst);
        tile_regs_commit();
        cb_interm_pre_add_obj.pop_front(block.full_block_size());
        cb_ex_obj.pop_front(1);
        cb_ex2_obj.pop_front(1);

        cb_ex_obj.reserve_back(1);
        cb_ex2_obj.reserve_back(1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_interm_pre_add, cb_ex);
        pack_tile(mean_dst, cb_ex);
        pack_tile(var_dst, cb_ex2);
        tile_regs_release();
        cb_ex_obj.push_back(1);
        cb_ex2_obj.push_back(1);
    }

    cb_ex_obj.wait_front(1);
    cb_ex2_obj.wait_front(1);
    tile_regs_acquire();
    copy_tile_init(cb_ex);
    copy_tile(cb_ex, 0, mean_dst);
    copy_tile_to_dst_init_short_with_dt(cb_ex, cb_ex2);
    copy_tile(cb_ex2, 0, var_dst);
    welford_restore_state(mean_dst);
    // Store the mean and variance to the destination registers
    welford_finalize_to_row<W>(mean_dst, W - 1, reciprocal_lut);
    tile_regs_commit();
    cb_ex_obj.pop_front(1);
    cb_ex2_obj.pop_front(1);
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
template <
    uint32_t cb_in,
    uint32_t input_dst,
    uint32_t mean_dst,
    uint32_t Wt,
    uint32_t tile_width,
    uint32_t W,
    uint32_t blk>
void welford_no_fuse_pre_add(const std::array<uint32_t, W>& reciprocal_lut) {
    experimental::CircularBuffer cb_in_obj(cb_in);

    // The number of valid columns in the last tile in width dimension.
    // Because the Welford's llk is given transposed data, skip some rows when
    // we want to skip some columns from getting processed by layer_norm.
    // When last tile is full the value is 0 and is not used because full update is done.
    constexpr uint32_t last_tile_rows = W % tile_width;
    constexpr bool is_last_tile_full = (last_tile_rows == 0);

    uint32_t sample_idx = 0;
    reconfig_data_format_srca(cb_in);
    transpose_wh_init_short(cb_in);
    tile_regs_acquire();
    welford_init();

    // Process all but the last tile
    for (uint32_t wt = 0; wt < (Wt - 1); ++wt) {
        cb_in_obj.wait_front(1);
        // Welford's needs transposed input tile
        transpose_wh_tile(cb_in, 0, input_dst);
        welford_update<W>(input_dst, sample_idx, reciprocal_lut);

        // Pop the input
        cb_in_obj.pop_front(1);
        sample_idx += tile_width;
    }

    // Process the last tile
    // Reader is sending full blocks, so we need to stay in sync.
    // wait/pop the last tile + any remaining in the last block
    const auto num_to_sync = generic::blocks(Wt, blk).back().remainder() + 1;
    cb_in_obj.wait_front(num_to_sync);
    transpose_wh_tile(cb_in, 0, input_dst);

    if constexpr (is_last_tile_full) {
        welford_update<W>(input_dst, sample_idx, reciprocal_lut);
    } else {
        welford_update_rows<W>(input_dst, sample_idx, 0, last_tile_rows, reciprocal_lut);
    }

    // Store the mean and variance to the destination registers
    welford_finalize_to_row<W>(mean_dst, W - 1, reciprocal_lut);

    tile_regs_commit();

    cb_in_obj.pop_front(num_to_sync);
}

void kernel_main() {
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
    // CB indices - configurable via named compile-time args for kernel chaining support
    constexpr auto cb_eps = get_named_compile_time_arg_val("cb_eps");  // single tile generated by the reader
    constexpr auto cb_in = get_named_compile_time_arg_val("cb_in");    // input x or a for fused pre-add (x=a+b)
    constexpr auto cb_inb = get_named_compile_time_arg_val("cb_inb");  // input b for fused pre-add
    constexpr auto cb_out = get_named_compile_time_arg_val("cb_out");  // output
    constexpr auto cb_gamma = get_named_compile_time_arg_val("cb_gamma");
    constexpr auto cb_beta = get_named_compile_time_arg_val("cb_beta");
    uint32_t cb_xmm = get_named_compile_time_arg_val("cb_xmm");                        // x - E[x]
    constexpr auto cb_ex = get_named_compile_time_arg_val("cb_ex");                    // E[x]
    constexpr auto cb_ex2 = get_named_compile_time_arg_val("cb_ex2");                  // Var[x] = E[(x-E[x])^2]
    constexpr auto cb_ex2pe = get_named_compile_time_arg_val("cb_ex2pe");              // Var[x]+ε
    constexpr auto cb_fusion = get_named_compile_time_arg_val("cb_fusion");            // stream gamma/beta
    constexpr auto cb_interm_pre_add = get_named_compile_time_arg_val("cb_x");         // intermediate for fused pre-add
    constexpr auto cb_reciprocals = get_named_compile_time_arg_val("cb_reciprocals");  // Pre-computed reciprocals

    experimental::CircularBuffer cb_eps_obj(cb_eps);
    experimental::CircularBuffer cb_in_obj(cb_in);
    experimental::CircularBuffer cb_inb_obj(cb_inb);
    experimental::CircularBuffer cb_out_obj(cb_out);
    experimental::CircularBuffer cb_gamma_obj(cb_gamma);
    experimental::CircularBuffer cb_beta_obj(cb_beta);
    experimental::CircularBuffer cb_ex_obj(cb_ex);
    experimental::CircularBuffer cb_ex2_obj(cb_ex2);
    experimental::CircularBuffer cb_ex2pe_obj(cb_ex2pe);

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

    cb_eps_obj.wait_front(onetile);  // comes from the reader

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
            welford_no_fuse_pre_add<cb_in, input_dst, mean_dst, Wt, tile_width, W, blk>(*p_reciprocals);
        }
        // We should expect that either of the two would have have populated dst regs with mean and
        // variance in mean_dst and var_dst respectively.

        cb_ex_obj.reserve_back(onetile);
        cb_ex2_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_reconfig_data_format(cb_ex);
        pack_tile(mean_dst, cb_ex);
        pack_tile(var_dst, cb_ex2);
        tile_regs_release();
        cb_ex_obj.push_back(onetile);
        cb_ex2_obj.push_back(onetile);

        // Transpose mean and variance back to
        // columns and pack back to CBs
        reconfig_data_format_srca(cb_ex);
        transpose_wh_init_short(cb_ex);

        cb_ex_obj.wait_front(onetile);
        cb_ex2_obj.wait_front(onetile);
        tile_regs_acquire();
        transpose_wh_tile(cb_ex, 0, mean_dst);
        transpose_wh_tile(cb_ex2, 0, var_dst);
        tile_regs_commit();
        cb_ex_obj.pop_front(onetile);
        cb_ex2_obj.pop_front(onetile);

        cb_ex_obj.reserve_back(onetile);
        cb_ex2_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_reconfig_data_format(cb_ex);
        pack_tile(mean_dst, cb_ex);
        pack_reconfig_data_format(cb_ex2);
        pack_tile(var_dst, cb_ex2);
        tile_regs_release();
        cb_ex_obj.push_back(onetile);
        cb_ex2_obj.push_back(onetile);

        // =====================================
        // Calculate 1/(√(Var(X) + ε))
        // =====================================
        reconfig_data_format(cb_ex2, cb_eps);
        add_tiles_init(cb_ex2, cb_eps);

        cb_ex2_obj.wait_front(onetile);
        tile_regs_acquire();
        add_tiles(cb_ex2, cb_eps, 0, 0, dst0);
        rsqrt_tile_init();
        rsqrt_tile(dst0);
        tile_regs_commit();
        cb_ex2_obj.pop_front(onetile);

        cb_ex2pe_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_tile(dst0, cb_ex2pe);
        tile_regs_release();
        cb_ex2pe_obj.push_back(onetile);

        // broadcasts the tile since cb_ex2pe is a column vector that contains the important data
        cb_ex2pe_obj.wait_front(onetile);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_ex2pe);
        unary_bcast_init<BroadcastType::COL>(cb_ex2pe, cb_ex2pe);
        unary_bcast<BroadcastType::COL>(cb_ex2pe, 0, dst0);
        cb_ex2pe_obj.pop_front(onetile);
        tile_regs_commit();

        cb_ex2pe_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_tile(dst0, cb_ex2pe);
        tile_regs_release();
        cb_ex2pe_obj.push_back(onetile);

        // =====================================
        // Second pass over the input.
        // Computes the final value:
        //    x-E[x]
        //(---------------*𝛄)+ß
        //  √(Var(x)+ε)
        // =====================================
        cb_ex2pe_obj.wait_front(onetile);
        cb_ex_obj.wait_front(onetile);

        for (auto block : generic::blocks(Wt, blk)) {
            // Last block may only be partially-filled,
            // and only tiles that have data in them are
            // processed, but need to sync with reader on full blocks
            cb_in_obj.wait_front(block.full_block_size());
            tile_regs_acquire();
            reconfig_data_format(cb_in, cb_ex);
            sub_bcast_cols_init_short(cb_in, cb_ex);
            // x-E[x]
            for (auto i : block.local()) {
                sub_tiles_bcast_cols(cb_in, cb_ex, i, 0, i);
            }
            cb_in_obj.pop_front(block.full_block_size());

            if constexpr (fuse_pre_add) {
                // Fuse in = in + b
                reconfig_data_format_srca(cb_in, cb_inb);
                binary_dest_reuse_tiles_init<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_inb);
                cb_inb_obj.wait_front(block.full_block_size());
                for (auto i : block.local()) {
                    binary_dest_reuse_tiles<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_inb, i, i);
                }
                cb_inb_obj.pop_front(block.full_block_size());
                reconfig_data_format_srca(cb_inb, cb_ex2pe);
            }

            // Multiply by 1/(√(Var(X) + ε))
            reconfig_data_format_srca(fuse_pre_add ? cb_inb : cb_in, cb_ex2pe);
            binary_dest_reuse_tiles_init<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_ex2pe);
            for (auto i : block.local()) {
                binary_dest_reuse_tiles<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_ex2pe, 0, i);
            }
            tile_regs_commit();

            if constexpr (!(do_gamma == 1 or do_beta == 1)) {
                cb_xmm = cb_out;
            }

            pack_reconfig_data_format(cb_xmm);
            // Sync with writer on full blocks
            experimental::CircularBuffer(cb_xmm).reserve_back(block.full_block_size());
            tile_regs_wait();
            for (auto i : block.local()) {
                pack_tile(i, cb_xmm);
            }
            experimental::CircularBuffer(cb_xmm).push_back(block.full_block_size());
            tile_regs_release();

            if constexpr (do_gamma == 1) {
                // Multiply by gamma
                reconfig_data_format(cb_xmm, cb_gamma);
                tile_regs_acquire();
                cb_gamma_obj.wait_front(block.full_block_size());
                experimental::CircularBuffer(cb_xmm).wait_front(block.full_block_size());
                mul_bcast_rows_init_short(cb_xmm, cb_gamma);
                for (auto i : block.local()) {
                    mul_tiles_bcast_rows(cb_xmm, cb_gamma, i, i, i);
                }
                tile_regs_commit();
                cb_gamma_obj.pop_front(block.full_block_size());
                experimental::CircularBuffer(cb_xmm).pop_front(block.full_block_size());

                if constexpr (!do_beta) {
                    pack_reconfig_data_format(cb_out);
                }
                tile_regs_wait();
                if constexpr (!do_beta) {
                    cb_out_obj.reserve_back(block.full_block_size());
                    for (auto i : block.local()) {
                        pack_tile(i, cb_out);
                    }
                    cb_out_obj.push_back(block.full_block_size());
                } else {
                    experimental::CircularBuffer(cb_xmm).reserve_back(block.full_block_size());
                    for (auto i : block.local()) {
                        pack_tile(i, cb_xmm);
                    }
                    experimental::CircularBuffer(cb_xmm).push_back(block.full_block_size());
                }
                tile_regs_release();
            }

            if constexpr (do_beta == 1) {
                // Add beta
                tile_regs_acquire();
                reconfig_data_format(cb_xmm, cb_beta);
                add_bcast_rows_init_short(cb_xmm, cb_beta);
                experimental::CircularBuffer(cb_xmm).wait_front(block.full_block_size());
                cb_beta_obj.wait_front(block.full_block_size());
                for (auto i : block.local()) {
                    add_tiles_bcast_rows(cb_xmm, cb_beta, i, i, i);
                }
                tile_regs_commit();
                cb_beta_obj.pop_front(block.full_block_size());
                experimental::CircularBuffer(cb_xmm).pop_front(block.full_block_size());

                pack_reconfig_data_format(cb_out);
                cb_out_obj.reserve_back(block.full_block_size());
                tile_regs_wait();
                for (auto i : block.local()) {
                    pack_tile(i, cb_out);
                }
                tile_regs_release();
                cb_out_obj.push_back(block.full_block_size());
            }
        }

        cb_xmm = get_named_compile_time_arg_val("cb_xmm");  // x minus mean
        cb_ex2pe_obj.pop_front(onetile);
        cb_ex_obj.pop_front(onetile);
    }  // NCHt loop
}
