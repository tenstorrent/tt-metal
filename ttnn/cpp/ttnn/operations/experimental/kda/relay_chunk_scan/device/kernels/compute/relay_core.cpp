// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/compute/matmul_subblock.hpp"

constexpr uint32_t cb_e = get_compile_time_arg_val(0);
constexpr uint32_t cb_b = get_compile_time_arg_val(1);
constexpr uint32_t cb_ein = get_compile_time_arg_val(2);
constexpr uint32_t cb_eout = get_compile_time_arg_val(3);
constexpr uint32_t kt = get_compile_time_arg_val(4);
// The same landed entry state as cb_ein, unpacked straight to DST (FP32, exact).
constexpr uint32_t cb_ein_exact = get_compile_time_arg_val(5);
// One BF16 identity tile: B enters the sum as I @ B.
constexpr uint32_t cb_identity = get_compile_time_arg_val(6);
// The final carry (see the reader): FP32 E and B of the final transition, and the relayed final state P in its matmul
// and exact views.
constexpr uint32_t cb_fe = get_compile_time_arg_val(7);
constexpr uint32_t cb_fb = get_compile_time_arg_val(8);
constexpr uint32_t cb_p = get_compile_time_arg_val(9);
constexpr uint32_t cb_p_exact = get_compile_time_arg_val(10);
// The reader's {holder, split, heads}.
constexpr uint32_t cb_ctl = get_compile_time_arg_val(11);
constexpr uint32_t relay_heads = get_compile_time_arg_val(12);  // cb_eout's head slots
constexpr uint32_t state_tiles = kt * kt;
constexpr uint32_t dst_tiles = compute_kernel_lib::DEST_AUTO_LIMIT;
// A tile column of the next state is formed in one DST pass (the op checks K against the DST).
static_assert(kt <= dst_tiles);
// A DST pass of the carry takes `half_dst` tiles, with as many operand tiles beside them.
constexpr uint32_t half_dst = kda::largest_divisor_at_most(kt, dst_tiles / 2);

// The relay's hop for the front head: the next entry state S_out = A S_in + B for one head, from the complement form
// E = A - I: S_out = S_in + E S_in + B, summed in the FP32 DST. The exact copy of S_in unpacks to DST losslessly; E
// S_in accumulates on top at the affine prefix's fidelity, so the matmul's source-register copy of S_in reaches only
// the small E S_in term; B accumulates as I @ B, exact at HiFi2 and above (a matmul's in1 goes to srcA, whose HiFi2
// phases cover a BF16 mantissa, and 1.0 is exact in srcB). Per output tile column (column-major state): column j needs
// only column j of S_in, so it is computed as soon as that column lands.
void relay_hop() {
    pack_reconfig_data_format(cb_eout);
    cb_wait_front(cb_e, state_tiles);
    cb_wait_front(cb_b, state_tiles);
    cb_wait_front(cb_identity, 1);
    for (uint32_t j = 0; j < kt; ++j) {
        cb_reserve_back(cb_eout, kt);
        cb_wait_front(cb_ein, (j + 1) * kt);
        cb_wait_front(cb_ein_exact, (j + 1) * kt);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_ein_exact);
        copy_init(cb_ein_exact);
        for (uint32_t r = 0; r < kt; ++r) {
            copy_tile(cb_ein_exact, j * kt + r, r);
        }
        reconfig_data_format<SrcOrder::Reverse>(cb_e, cb_ein);
        matmul_block_init(cb_e, cb_ein, false, 1, kt, kt);
        for (uint32_t k = 0; k < kt; ++k) {
            matmul_block(cb_e, cb_ein, k, j * kt + k, 0, false, 1, kt, kt);
        }
        reconfig_data_format<SrcOrder::Reverse>(cb_identity, cb_b);
        matmul_block_init(cb_identity, cb_b, false, 1, 1, 1);
        for (uint32_t r = 0; r < kt; ++r) {
            matmul_block(cb_identity, cb_b, 0, r * kt + j, r, false, 1, 1, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t r = 0; r < kt; ++r) {
            pack_tile(r, cb_eout);
        }
        tile_regs_release();
        cb_push_back(cb_eout, kt);
    }
    cb_pop_front(cb_e, state_tiles);
    cb_pop_front(cb_ein, state_tiles);
    cb_pop_front(cb_ein_exact, state_tiles);
    cb_pop_front(cb_b, state_tiles);
}

// The final carry of the front head on the first rank: P + (E P + B) per column with the final transition when the
// interval ends in a separated tail, else P itself. B and P unpack to DST exactly and are added on the SFPU, E P
// accumulates on top.
void carry_pass(bool split) {
    pack_reconfig_data_format(cb_eout);
    if (split) {
        cb_wait_front(cb_fe, state_tiles);
        cb_wait_front(cb_fb, state_tiles);
    }
    for (uint32_t j = 0; j < kt; ++j) {
        cb_reserve_back(cb_eout, kt);
        cb_wait_front(cb_p, (j + 1) * kt);
        cb_wait_front(cb_p_exact, (j + 1) * kt);
        for (uint32_t r0 = 0; r0 < kt; r0 += half_dst) {
            tile_regs_acquire();
            if (split) {
                reconfig_data_format_srca(cb_fb);
                copy_init(cb_fb);
                for (uint32_t i = 0; i < half_dst; ++i) {
                    copy_tile(cb_fb, (r0 + i) * kt + j, i);
                }
                reconfig_data_format_srca(cb_p_exact);
                copy_init(cb_p_exact);
                for (uint32_t i = 0; i < half_dst; ++i) {
                    copy_tile(cb_p_exact, j * kt + r0 + i, half_dst + i);
                }
                add_binary_tile_init();
                for (uint32_t i = 0; i < half_dst; ++i) {
                    add_binary_tile(i, half_dst + i, i);
                }
                reconfig_data_format<SrcOrder::Reverse>(cb_fe, cb_p);
                matmul_block_init(cb_fe, cb_p, false, 1, half_dst, kt);
                for (uint32_t k = 0; k < kt; ++k) {
                    matmul_block(cb_fe, cb_p, r0 * kt + k, j * kt + k, 0, false, 1, half_dst, kt);
                }
            } else {
                reconfig_data_format_srca(cb_p_exact);
                copy_init(cb_p_exact);
                for (uint32_t i = 0; i < half_dst; ++i) {
                    copy_tile(cb_p_exact, j * kt + r0 + i, i);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < half_dst; ++i) {
                pack_tile(i, cb_eout);
            }
            tile_regs_release();
        }
        cb_push_back(cb_eout, kt);
    }
    cb_pop_front(cb_p, state_tiles);
    cb_pop_front(cb_p_exact, state_tiles);
    if (split) {
        cb_pop_front(cb_fe, state_tiles);
        cb_pop_front(cb_fb, state_tiles);
    }
}

// A relay core's heads, one after the other: each head's hop as its entry state lands, then on the first rank each
// head's final carry as its relayed final state lands.
void kernel_main() {
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_e, cb_ein, cb_eout);
    cb_wait_front(cb_ctl, 1);
    const uint32_t holder = read_tile_value(cb_ctl, 0, 0);
    const uint32_t split = read_tile_value(cb_ctl, 0, 1);
    const uint32_t heads = read_tile_value(cb_ctl, 0, 2);
    const uint32_t final_from_last = read_tile_value(cb_ctl, 0, 3);
    cb_pop_front(cb_ctl, 1);
    // cb_eout holds a state per head slot: a core with fewer heads fills the rest, which the writer releases with the
    // heads' states.
    const auto fill = [&]() {
        cb_reserve_back(cb_eout, (relay_heads - heads) * state_tiles);
        cb_push_back(cb_eout, (relay_heads - heads) * state_tiles);
    };
    for (uint32_t h = 0; h < heads; ++h) {
        relay_hop();
    }
    fill();
    cb_pop_front(cb_identity, 1);
    if (holder && !final_from_last) {
        for (uint32_t h = 0; h < heads; ++h) {
            carry_pass(split);
        }
        fill();
    }
}
