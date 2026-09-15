// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/compute_kernel_api.h"

void kernel_main() {
    using namespace ckernel;

    constexpr uint32_t cb_dy = 0, cb_x = 1, cb_gamma = 2, cb_inv = 3, cb_d = 4, cb_out = 16, cb_acc = 17, cb_part = 18;
    constexpr uint32_t cb_zero_done = 19, cb_go = 20;
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t neg_one_bits = get_compile_time_arg_val(1);
    constexpr uint32_t with_dgamma = get_compile_time_arg_val(2);
    const uint32_t row_count = get_arg_val<uint32_t>(0);

    if constexpr (with_dgamma) {
        compute_kernel_hw_startup(cb_dy, cb_gamma, cb_out);
    } else {
        compute_kernel_hw_startup(cb_dy, cb_inv, cb_out);
    }

    auto zero_fill_acc = [&]() {
        fill_tile_init();
        pack_reconfig_data_format(cb_acc);
        pack_reconfig_l1_acc(0);
        for (uint32_t c = 0; c < Wt; ++c) {
            tile_regs_acquire();
            fill_tile(0, 0.0f);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, cb_acc, c);
            tile_regs_release();
        }
    };

    if constexpr (with_dgamma) {
        cb_reserve_back(cb_acc, Wt);
        zero_fill_acc();
    }

    mul_binary_tile_init();
    addcmul_tile_init();

    for (uint32_t r = 0; r < row_count; ++r) {
        cb_wait_front(cb_dy, Wt);
        cb_wait_front(cb_x, Wt);
        cb_wait_front(cb_inv, 1);
        cb_wait_front(cb_d, 1);
        cb_reserve_back(cb_out, Wt);

        for (uint32_t c = 0; c < Wt; ++c) {
            tile_regs_acquire();
            copy_tile_init(cb_dy);
            copy_tile(cb_dy, c, 0);
            if constexpr (with_dgamma) {
                unary_bcast_init<BroadcastType::ROW>(cb_gamma);
                unary_bcast<BroadcastType::ROW>(cb_gamma, c, 1);
                mul_binary_tile(0, 1, 0);
            }
            unary_bcast_init<BroadcastType::COL>(cb_inv);
            unary_bcast<BroadcastType::COL>(cb_inv, 0, 1);
            mul_binary_tile(0, 1, 0);
            copy_tile_init(cb_x);
            copy_tile(cb_x, c, 2);
            unary_bcast_init<BroadcastType::COL>(cb_d);
            unary_bcast<BroadcastType::COL>(cb_d, 0, 1);
            addcmul_tile<DataFormat::Float32>(0, 2, 1, 0, neg_one_bits);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_out);
            pack_tile(0, cb_out);
            tile_regs_release();

            if constexpr (with_dgamma) {
                tile_regs_acquire();
                copy_tile_init(cb_dy);
                copy_tile(cb_dy, c, 0);
                copy_tile_init(cb_x);
                copy_tile(cb_x, c, 1);
                mul_binary_tile(0, 1, 0);
                unary_bcast_init<BroadcastType::COL>(cb_inv);
                unary_bcast<BroadcastType::COL>(cb_inv, 0, 1);
                mul_binary_tile(0, 1, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_acc);
                pack_reconfig_l1_acc(1);
                pack_tile<true>(0, cb_acc, c);
                pack_reconfig_l1_acc(0);
                tile_regs_release();
            }
        }

        cb_push_back(cb_out, Wt);
        cb_pop_front(cb_dy, Wt);
        cb_pop_front(cb_x, Wt);
        cb_pop_front(cb_inv, 1);
        cb_pop_front(cb_d, 1);
    }

    if constexpr (with_dgamma) {
        const uint32_t role = get_arg_val<uint32_t>(1);  // 0 member, 1 row leader, 2 root

        auto collapse_acc_to_part = [&]() {
            cb_wait_front(cb_acc, Wt);
            cb_reserve_back(cb_part, Wt);
            reconfig_data_format_srca(cb_acc);
            pack_reconfig_data_format(cb_part);
            copy_tile_to_dst_init_short(cb_acc);
            sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
            for (uint32_t c = 0; c < Wt; ++c) {
                tile_regs_acquire();
                copy_tile(cb_acc, c, 0);
                sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(0, 1, 1);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_part);
                tile_regs_release();
            }
            cb_push_back(cb_part, Wt);
            cb_pop_front(cb_acc, Wt);
        };
        auto open_gather = [&]() {
            cb_reserve_back(cb_acc, Wt);
            zero_fill_acc();
            cb_push_back(cb_acc, Wt);
            cb_reserve_back(cb_zero_done, 1);
            cb_push_back(cb_zero_done, 1);
        };
        auto wait_go = [&]() {
            cb_wait_front(cb_go, 1);
            cb_pop_front(cb_go, 1);
        };

        cb_push_back(cb_acc, Wt);
        collapse_acc_to_part();
        for (uint32_t stage = 0; stage < role; ++stage) {
            open_gather();
            wait_go();
            collapse_acc_to_part();
        }
    }
}
