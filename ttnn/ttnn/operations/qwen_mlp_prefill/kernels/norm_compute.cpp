// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/pack.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "named_args.hpp"

void kernel_main() {
    constexpr uint32_t row_tiles = CT_ARG("h_tiles");
    constexpr uint32_t chunk = CT_ARG("norm_chunk_tiles");
    constexpr uint32_t dest = CT_ARG("norm_dest_tiles");
    constexpr uint32_t cb_in = CT_ARG("cb_norm_in");
    constexpr uint32_t cb_weight = CT_ARG("cb_norm_weight");
    constexpr uint32_t cb_scaler = CT_ARG("cb_norm_scaler");
    constexpr uint32_t cb_squares = CT_ARG("cb_norm_squares");
    constexpr uint32_t cb_sum = CT_ARG("cb_norm_sum");
    constexpr uint32_t cb_rstd = CT_ARG("cb_norm_rstd");
    constexpr uint32_t cb_scaled = CT_ARG("cb_norm_scaled");
    constexpr uint32_t cb_out = CT_ARG("cb_norm_out");
    CircularBuffer in(cb_in);
    CircularBuffer weight(cb_weight);
    CircularBuffer scaler(cb_scaler);
    CircularBuffer squares(cb_squares);
    CircularBuffer sum(cb_sum);
    CircularBuffer rstd(cb_rstd);
    CircularBuffer scaled(cb_scaled);
    CircularBuffer out(cb_out);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in, CT_ARG("cb_norm_startup"), cb_sum);
    scaler.wait_front(1);
    weight.wait_front(row_tiles);

    for (uint32_t row = 0; row < CT_ARG("rows") / CT_ARG("norm_cores"); row++) {
        in.wait_front(row_tiles);

        sum.reserve_back(1);
        for (uint32_t first = 0; first < row_tiles; first += chunk) {
            squares.reserve_back(chunk);
            reconfig_data_format(cb_in, cb_in);
            pack_reconfig_data_format(cb_squares);
            mul_init(cb_in, cb_in, false);
            for (uint32_t part = 0; part < chunk; part += dest) {
                tile_regs_acquire();
                for (uint32_t i = 0; i < dest; i++) {
                    mul_tiles(cb_in, cb_in, first + part + i, first + part + i, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < dest; i++) {
                    pack_tile(i, cb_squares);
                }
                tile_regs_release();
            }
            squares.push_back(chunk);

            squares.wait_front(chunk);
            reconfig_data_format(cb_scaler, cb_squares);
            pack_reconfig_data_format(cb_sum);
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_squares, cb_scaler, cb_sum);
            tile_regs_acquire();
            for (uint32_t i = 0; i < chunk; i++) {
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_squares, cb_scaler, i, 0, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_l1_acc(first > 0);
            pack_tile<true>(0, cb_sum, 0);
            pack_reconfig_l1_acc(0);
            tile_regs_release();
            reduce_uninit(cb_squares);
            squares.pop_front(chunk);
        }
        sum.push_back(1);

        sum.wait_front(1);
        rstd.reserve_back(1);
        reconfig_data_format_srca(cb_sum);
        pack_reconfig_data_format(cb_rstd);
        copy_init(cb_sum);
        tile_regs_acquire();
        copy_tile(cb_sum, 0, 0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, CT_ARG("norm_inv_width_bits"));
        add_unary_tile(0, CT_ARG("norm_eps_bits"));
        rsqrt_tile_init();
        rsqrt_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_rstd);
        tile_regs_release();
        rstd.push_back(1);
        sum.pop_front(1);

        rstd.wait_front(1);
        out.reserve_back(row_tiles);
        for (uint32_t first = 0; first < row_tiles; first += chunk) {
            scaled.reserve_back(chunk);
            reconfig_data_format(cb_in, cb_rstd);
            pack_reconfig_data_format(cb_scaled);
            mul_bcast_cols_init(cb_in, cb_rstd);
            for (uint32_t part = 0; part < chunk; part += dest) {
                tile_regs_acquire();
                for (uint32_t i = 0; i < dest; i++) {
                    mul_tiles_bcast_cols(cb_in, cb_rstd, first + part + i, 0, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < dest; i++) {
                    pack_tile(i, cb_scaled);
                }
                tile_regs_release();
            }
            scaled.push_back(chunk);

            scaled.wait_front(chunk);
            reconfig_data_format(cb_scaled, cb_weight);
            pack_reconfig_data_format(cb_out);
            mul_bcast_rows_init(cb_scaled, cb_weight);
            for (uint32_t part = 0; part < chunk; part += dest) {
                tile_regs_acquire();
                for (uint32_t i = 0; i < dest; i++) {
                    mul_tiles_bcast_rows(cb_scaled, cb_weight, part + i, first + part + i, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < dest; i++) {
                    pack_tile<true>(i, cb_out, first + part + i);
                }
                tile_regs_release();
            }
            scaled.pop_front(chunk);
        }
        out.push_back(row_tiles);
        in.pop_front(row_tiles);
        rstd.pop_front(1);
    }
}
