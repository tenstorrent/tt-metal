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
};

#define RMSNORM_CBS                                                                                          \
    RmsnormCbs {                                                                                             \
        CT_ARG("cb_norm_in"), CT_ARG("cb_norm_weight"), CT_ARG("cb_norm_scaler"), CT_ARG("cb_norm_squares"), \
            CT_ARG("cb_norm_sum"), CT_ARG("cb_norm_rstd"), CT_ARG("cb_norm_scaled"), CT_ARG("cb_norm_out")   \
    }

ALWI void rmsnorm_row(
    const RmsnormCbs& cb, uint32_t row_tiles, uint32_t group, uint32_t inv_width_bits, uint32_t eps_bits) {
    CircularBuffer in(cb.in);
    CircularBuffer weight(cb.weight);
    CircularBuffer squares(cb.squares);
    CircularBuffer sum(cb.sum);
    CircularBuffer rstd(cb.rstd);
    CircularBuffer scaled(cb.scaled);
    CircularBuffer out(cb.out);
    const uint32_t groups = row_tiles / group;

    sum.reserve_back(1);
    for (uint32_t g = 0; g < groups; g++) {
        in.wait_front(group);
        squares.reserve_back(group);
        reconfig_data_format(cb.in, cb.in);
        pack_reconfig_data_format(cb.squares);
        mul_init(cb.in, cb.in);
        tile_regs_acquire();
        for (uint32_t i = 0; i < group; i++) {
            mul_tiles(cb.in, cb.in, i, i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < group; i++) {
            pack_tile(i, cb.squares);
        }
        tile_regs_release();
        squares.push_back(group);
        in.pop_front(group);

        squares.wait_front(group);
        reconfig_data_format(cb.scaler, cb.squares);
        pack_reconfig_data_format(cb.sum);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb.squares, cb.scaler, cb.sum);
        tile_regs_acquire();
        for (uint32_t i = 0; i < group; i++) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb.squares, cb.scaler, i, 0, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_l1_acc(g > 0);
        pack_tile<true>(0, cb.sum, 0);
        pack_reconfig_l1_acc(0);
        tile_regs_release();
        reduce_uninit(cb.squares);
        squares.pop_front(group);
    }
    sum.push_back(1);

    sum.wait_front(1);
    rstd.reserve_back(1);
    reconfig_data_format_srca(cb.sum);
    pack_reconfig_data_format(cb.rstd);
    copy_init(cb.sum);
    tile_regs_acquire();
    copy_tile(cb.sum, 0, 0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_width_bits);
    add_unary_tile(0, eps_bits);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb.rstd);
    tile_regs_release();
    rstd.push_back(1);
    sum.pop_front(1);

    rstd.wait_front(1);
    for (uint32_t g = 0; g < groups; g++) {
        in.wait_front(group);
        scaled.reserve_back(group);
        reconfig_data_format(cb.in, cb.rstd);
        pack_reconfig_data_format(cb.scaled);
        mul_bcast_cols_init(cb.in, cb.rstd);
        tile_regs_acquire();
        for (uint32_t i = 0; i < group; i++) {
            mul_tiles_bcast_cols(cb.in, cb.rstd, i, 0, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < group; i++) {
            pack_tile(i, cb.scaled);
        }
        tile_regs_release();
        scaled.push_back(group);
        in.pop_front(group);

        scaled.wait_front(group);
        weight.wait_front(group);
        out.reserve_back(group);
        reconfig_data_format(cb.scaled, cb.weight);
        pack_reconfig_data_format(cb.out);
        mul_bcast_rows_init(cb.scaled, cb.weight);
        tile_regs_acquire();
        for (uint32_t i = 0; i < group; i++) {
            mul_tiles_bcast_rows(cb.scaled, cb.weight, i, i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < group; i++) {
            pack_tile(i, cb.out);
        }
        tile_regs_release();
        out.push_back(group);
        scaled.pop_front(group);
        weight.pop_front(group);
    }
    rstd.pop_front(1);
}
