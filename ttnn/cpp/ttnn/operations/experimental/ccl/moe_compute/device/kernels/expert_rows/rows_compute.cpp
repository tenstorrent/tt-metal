// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// expert rows, compute of one core of one expert group. Per job (one expert, at most 32 routed rows):
//  P1: for each of this core's W0/W1 column groups: [g0 u0 g1 u1] = x_job . [W0 j0, W1 j0, W0 j1, W1 j1] over all
//      hidden tiles, activation on the packer's SFPU, a(j0), a(j1) -> cb_a.
//  P2: a2 (every intermediate column of the job, from the exchange) . W2 for each of this core's 4-tile output groups,
//      the W2 rows in moe_compute's per-core rotated order -> cb_rows (bf16 tiles; the writer sends the real rows).
// Order P1(0), P1(1), P2(0), P1(2), P2(1), ..., P2(J - 1).
#include <cstdint>
#include "../moe_ring_common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/fill.h"
#include "../moe_activation.h"

namespace {

constexpr uint32_t BT = get_named_compile_time_arg_val("block_tiles");
constexpr uint32_t cb_w = get_named_compile_time_arg_val("cb_w");

// Weight tiles of one pass arrive in blocks of BT; a run (one group) ends with a short block.
struct WeightStream {
    uint32_t left = 0, cur = 0;
    bool held = false;
    FORCE_INLINE uint32_t take(uint32_t n) {
        if (left == 0) {
            if (held) {
                cb_pop_front(cb_w, BT);
            }
            cb_wait_front(cb_w, BT);
            held = true;
            left = BT;
            cur = 0;
        }
        const uint32_t i = cur;
        cur += n;
        left -= n;
        return i;
    }
    FORCE_INLINE void end_run() {
        if (held) {
            cb_pop_front(cb_w, BT);
        }
        held = false;
        left = 0;
    }
};

}  // namespace

void kernel_main() {
    constexpr uint32_t Ht = get_named_compile_time_arg_val("hidden_tiles");
    constexpr uint32_t Nt = get_named_compile_time_arg_val("intermediate_tiles");
    constexpr uint32_t ring = get_named_compile_time_arg_val("ring_cores");
    constexpr bool has_bias = get_named_compile_time_arg_val("has_bias") == 1;
    constexpr uint32_t cb_x = get_named_compile_time_arg_val("cb_x");
    constexpr uint32_t cb_a = get_named_compile_time_arg_val("cb_a");
    constexpr uint32_t cb_a2 = get_named_compile_time_arg_val("cb_a2");
    constexpr uint32_t cb_rows = get_named_compile_time_arg_val("cb_rows");
    constexpr uint32_t cb_ctl = get_named_compile_time_arg_val("cb_ctl");
    constexpr uint32_t cb_ones = get_named_compile_time_arg_val("cb_ones");
    constexpr uint32_t a_tiles = get_named_compile_time_arg_val("a_tiles");
    constexpr auto activation =
        ttnn::experimental::prim::detail::MoEActivationFunction(get_named_compile_time_arg_val("activation_function"));
    static_assert(BT % 4 == 0, "a block holds whole 4-tile K rows");
    constexpr auto cols_lut = moe_ring::make_shard_lut<Nt, ring>();

    const uint32_t ng = get_arg_val<uint32_t>(0);  // W0/W1 column groups of this core
    const uint32_t nq = get_arg_val<uint32_t>(1);  // W2 output groups of this core
    const uint32_t r = get_arg_val<uint32_t>(2);   // ring position of this core's bank (W2 row rotation)

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_a);
    if constexpr (has_bias) {
        // ones tile: matmul(ones, bias row) adds the bias to every row
        pack_reconfig_data_format(cb_ones);
        copy_init(cb_ones);
        tile_regs_acquire();
        fill_tile_init();
        fill_tile(0, 1.f);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_ones, 1);
        pack_tile(0, cb_ones);
        tile_regs_release();
        cb_push_back(cb_ones, 1);
        cb_wait_front(cb_ones, 1);
    }
    cb_wait_front(cb_ctl, 1);
    const uint32_t nj = read_tile_value(cb_ctl, 0, 0);  // this group's jobs
    if (nj == 0) {
        return;
    }
    pack_reconfig_data_format(cb_a);
    reconfig_data_format_srcb(cb_x);
    reconfig_data_format_srca(cb_w);
    MATH((ckernel::zeroacc()));
    WeightStream ws;

    auto phase1 = [&]() {
        if constexpr (activation != ttnn::experimental::prim::detail::MoEActivationFunction::GELU) {
            ::moe_activation::pack_init_activation<activation>();
        }
        matmul_block_init(cb_x, cb_w, /*transpose=*/false, /*ct_dim=*/4, /*rt_dim=*/1, /*kt_dim=*/1);
        cb_wait_front(cb_x, Ht);
        cb_reserve_back(cb_a, a_tiles);
        for (uint32_t u = 0; u < ng; ++u) {
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < Ht; ++kt) {
                matmul_block(cb_x, cb_w, kt, ws.take(4), 0, false, 4, 1, 1);
            }
            if constexpr (has_bias) {
                matmul_block(cb_ones, cb_w, 0, ws.take(4), 0, false, 4, 1, 1);
            }
            ws.end_run();
            tile_regs_commit();
            // tile_regs_wait() plus a CFG stall, so the SETC16 below waits for math to leave these DEST registers
            PACK(TTI_SEMWAIT(
                p_stall::STALL_TDMA | p_stall::STALL_CFG,
                semaphore::t6_sem(semaphore::MATH_PACK),
                p_stall::STALL_ON_ZERO));
            PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
            ::moe_activation::PackActivation<activation, /*kPairs=*/2>::compute();
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
            pack_tile<true>(0, cb_a, 2 * u);
            pack_tile<true>(2, cb_a, 2 * u + 1);
            tile_regs_release();
        }
        cb_push_back(cb_a, a_tiles);
        cb_pop_front(cb_x, Ht);
    };
    auto phase2 = [&]() {
        cb_wait_front(cb_a2, Nt);
        matmul_block_init(cb_a2, cb_w, /*transpose=*/false, /*ct_dim=*/4, /*rt_dim=*/1, /*kt_dim=*/1);
        for (uint32_t q = 0; q < nq; ++q) {
            tile_regs_acquire();
            // W2 rows: this ring position's columns first, then the previous ring positions' (prepare_w2 order)
            uint32_t src = r, col = 0, left = cols_lut[r];
            for (uint32_t s = 0; s < r; ++s) {
                col += cols_lut[s];
            }
            for (uint32_t i = 0; i < Nt; ++i) {
                while (left == 0) {
                    src = src == 0 ? ring - 1 : src - 1;
                    left = cols_lut[src];
                    col = 0;
                    for (uint32_t s = 0; s < src; ++s) {
                        col += cols_lut[s];
                    }
                }
                matmul_block(cb_a2, cb_w, col, ws.take(4), 0, false, 4, 1, 1);
                ++col;
                --left;
            }
            if constexpr (has_bias) {
                matmul_block(cb_ones, cb_w, 0, ws.take(4), 0, false, 4, 1, 1);
            }
            ws.end_run();
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_rows, 4);
            for (uint32_t t = 0; t < 4; ++t) {
                pack_tile<true>(t, cb_rows, t);
            }
            cb_push_back(cb_rows, 4);
            tile_regs_release();
        }
        cb_pop_front(cb_a2, Nt);
    };

    phase1();
    for (uint32_t j = 1; j < nj; ++j) {
        phase1();
        phase2();
    }
    phase2();
}
