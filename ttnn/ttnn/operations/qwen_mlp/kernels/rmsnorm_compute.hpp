// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/pack.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "named_args.hpp"

struct RmsnormCbs {
    uint32_t in;
    uint32_t weight;
    uint32_t scaler;
    uint32_t squares;
    uint32_t sum;
    uint32_t rstd;
    uint32_t scaled;
    uint32_t out;
    uint32_t stats_out;
    uint32_t stats_in;
};

#define RMSNORM_CBS                                                                                          \
    RmsnormCbs {                                                                                             \
        CT_ARG("cb_norm_in"), CT_ARG("cb_norm_weight"), CT_ARG("cb_norm_scaler"), CT_ARG("cb_norm_squares"), \
            CT_ARG("cb_norm_sum"), CT_ARG("cb_norm_rstd"), CT_ARG("cb_norm_scaled"), CT_ARG("cb_norm_out"),  \
            CT_ARG("cb_stats_out"), CT_ARG("cb_stats_in")                                                    \
    }

struct RmsnormShape {
    uint32_t rows;
    uint32_t block_tiles;
    uint32_t group_size;
    uint32_t dest_tiles;
    uint32_t inv_width_bits;
    uint32_t eps_bits;
};

// Sum of squares of every row of the own K blocks, accumulated in fp32 and handed over as bf16 partial sums.
ALWI void rmsnorm_partial_sums(const RmsnormCbs& cb, const RmsnormShape& s, uint32_t own_blocks) {
    CircularBuffer in(cb.in);
    CircularBuffer squares(cb.squares);
    CircularBuffer sum(cb.sum);
    CircularBuffer stats_out(cb.stats_out);

    sum.reserve_back(s.rows);
    for (uint32_t b = 0; b < own_blocks; b++) {
        for (uint32_t row = 0; row < s.rows; row++) {
            in.wait_front(s.block_tiles);
            squares.reserve_back(s.block_tiles);
            reconfig_data_format(cb.in, cb.in);
            pack_reconfig_data_format(cb.squares);
            mul_init(cb.in, cb.in);
            for (uint32_t first = 0; first < s.block_tiles; first += s.dest_tiles) {
                tile_regs_acquire();
                for (uint32_t i = 0; i < s.dest_tiles; i++) {
                    mul_tiles(cb.in, cb.in, first + i, first + i, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < s.dest_tiles; i++) {
                    pack_tile(i, cb.squares);
                }
                tile_regs_release();
            }
            squares.push_back(s.block_tiles);
            in.pop_front(s.block_tiles);

            squares.wait_front(s.block_tiles);
            reconfig_data_format(cb.scaler, cb.squares);
            pack_reconfig_data_format(cb.sum);
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb.squares, cb.scaler, cb.sum);
            tile_regs_acquire();
            for (uint32_t i = 0; i < s.block_tiles; i++) {
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb.squares, cb.scaler, i, 0, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_l1_acc(b > 0);
            pack_tile<true>(0, cb.sum, row);
            pack_reconfig_l1_acc(0);
            tile_regs_release();
            reduce_uninit(cb.squares);
            squares.pop_front(s.block_tiles);
        }
    }
    sum.push_back(s.rows);

    sum.wait_front(s.rows);
    stats_out.reserve_back(s.rows);
    reconfig_data_format_srca(cb.sum);
    pack_reconfig_data_format(cb.stats_out);
    copy_init(cb.sum);
    tile_regs_acquire();
    for (uint32_t row = 0; row < s.rows; row++) {
        copy_tile(cb.sum, row, row);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t row = 0; row < s.rows; row++) {
        pack_tile(row, cb.stats_out);
    }
    tile_regs_release();
    stats_out.push_back(s.rows);
    sum.pop_front(s.rows);
}

// rstd of every row from the partial sums of all cores of the group.
ALWI void rmsnorm_row_scales(const RmsnormCbs& cb, const RmsnormShape& s, uint32_t sources) {
    CircularBuffer stats_in(cb.stats_in);
    CircularBuffer rstd(cb.rstd);

    stats_in.wait_front(sources * s.rows);
    rstd.reserve_back(s.rows);
    reconfig_data_format_srca(cb.stats_in);
    pack_reconfig_data_format(cb.rstd);
    for (uint32_t row = 0; row < s.rows; row++) {
        copy_init(cb.stats_in);
        tile_regs_acquire();
        copy_tile(cb.stats_in, row, 0);
        add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb.stats_in);
        for (uint32_t source = 1; source < sources; source++) {
            add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb.stats_in, source * s.rows + row, 0);
        }
        binop_with_scalar_tile_init();
        mul_unary_tile(0, s.inv_width_bits);
        add_unary_tile(0, s.eps_bits);
        rsqrt_tile_init();
        rsqrt_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb.rstd, row);
        tile_regs_release();
    }
    rstd.push_back(s.rows);
    stats_in.pop_front(sources * s.rows);
}

// Normalises the own K blocks into the block-sized slots of the out CB, one K block per push.
ALWI void rmsnorm_apply(const RmsnormCbs& cb, const RmsnormShape& s, uint32_t own_blocks) {
    CircularBuffer in(cb.in);
    CircularBuffer weight(cb.weight);
    CircularBuffer rstd(cb.rstd);
    CircularBuffer scaled(cb.scaled);
    CircularBuffer out(cb.out);

    rstd.wait_front(s.rows);
    for (uint32_t b = 0; b < own_blocks; b++) {
        weight.wait_front(s.block_tiles);
        out.reserve_back(s.rows * s.block_tiles);
        for (uint32_t row = 0; row < s.rows; row++) {
            in.wait_front(s.block_tiles);
            scaled.reserve_back(s.block_tiles);
            reconfig_data_format(cb.in, cb.rstd);
            pack_reconfig_data_format(cb.scaled);
            mul_bcast_cols_init(cb.in, cb.rstd);
            for (uint32_t first = 0; first < s.block_tiles; first += s.dest_tiles) {
                tile_regs_acquire();
                for (uint32_t i = 0; i < s.dest_tiles; i++) {
                    mul_tiles_bcast_cols(cb.in, cb.rstd, first + i, row, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < s.dest_tiles; i++) {
                    pack_tile(i, cb.scaled);
                }
                tile_regs_release();
            }
            scaled.push_back(s.block_tiles);
            in.pop_front(s.block_tiles);

            scaled.wait_front(s.block_tiles);
            reconfig_data_format(cb.scaled, cb.weight);
            pack_reconfig_data_format(cb.out);
            mul_bcast_rows_init(cb.scaled, cb.weight);
            for (uint32_t first = 0; first < s.block_tiles; first += s.dest_tiles) {
                tile_regs_acquire();
                for (uint32_t i = 0; i < s.dest_tiles; i++) {
                    mul_tiles_bcast_rows(cb.scaled, cb.weight, first + i, first + i, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < s.dest_tiles; i++) {
                    pack_tile<true>(i, cb.out, row * s.block_tiles + first + i);
                }
                tile_regs_release();
            }
            scaled.pop_front(s.block_tiles);
        }
        out.push_back(s.rows * s.block_tiles);
        weight.pop_front(s.block_tiles);
    }
    rstd.pop_front(s.rows);
}
