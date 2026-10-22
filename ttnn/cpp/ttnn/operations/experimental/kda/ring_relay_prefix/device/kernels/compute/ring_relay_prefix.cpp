// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

namespace {

// `count` tiles of `src`, in order, into `dst` in its format, four per DST acquire.
void copy_tiles(uint32_t src, uint32_t dst, uint32_t count) {
    constexpr uint32_t group = 4;
    reconfig_data_format_srca(src);
    pack_reconfig_data_format(dst);
    copy_init(src);
    for (uint32_t base = 0; base < count; base += group) {
        cb_reserve_back(dst, group);
        tile_regs_acquire();
        for (uint32_t i = 0; i < group; ++i) {
            copy_tile(src, base + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < group; ++i) {
            pack_tile(i, dst);
        }
        tile_regs_release();
        cb_push_back(dst, group);
    }
}

}  // namespace

// E_out = A E_in + B for one head, accumulated in the FP32 DST at the affine prefix's fidelity. The states cross the
// fabric in BF16: E_out leaves for the successor in BF16 (cb_eout), and the predecessor's E_out lands in BF16
// (cb_ein); the chronologically first rank enters at the FP32 initial state (cb_init). cb_state takes, in FP32 and in
// this order, E_out (the final state on the last rank), the entry state, and the final state the last rank sends the
// first rank.
void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t cb_ein = get_compile_time_arg_val(2);
    constexpr uint32_t cb_eout = get_compile_time_arg_val(3);
    constexpr uint32_t kt = get_compile_time_arg_val(4);
    constexpr uint32_t cb_init = get_compile_time_arg_val(5);
    constexpr uint32_t cb_state = get_compile_time_arg_val(6);
    constexpr uint32_t cb_final = get_compile_time_arg_val(7);
    constexpr uint32_t cb_first = get_compile_time_arg_val(8);
    constexpr uint32_t state_tiles = kt * kt;
    static_assert(state_tiles % 4 == 0);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_ein, cb_eout);
    // The reader tells whether this rank is chronologically first.
    cb_wait_front(cb_first, 1);
    const uint32_t cb_entry = read_tile_value(cb_first, 0, 0) != 0 ? cb_init : cb_ein;
    cb_pop_front(cb_first, 1);
    cb_wait_front(cb_a, state_tiles);
    cb_wait_front(cb_b, state_tiles);
    // Per output tile column (column-major state): B's column (BF16, exact in the FP32 DST) is copied into DST first
    // and the K-sum accumulates on top of it, so the state never leaves DST between the product and the add. Column j
    // needs only column j of E_in, so B's column is in DST before that column lands.
    for (uint32_t j = 0; j < kt; ++j) {
        cb_reserve_back(cb_eout, kt);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_b);
        copy_init(cb_b);
        for (uint32_t r = 0; r < kt; ++r) {
            copy_tile(cb_b, r * kt + j, r);
        }
        cb_wait_front(cb_entry, (j + 1) * kt);
        reconfig_data_format<SrcOrder::Reverse>(cb_a, cb_entry);
        matmul_block_init(cb_a, cb_entry, false, 1, kt, kt);
        for (uint32_t k = 0; k < kt; ++k) {
            matmul_block(cb_a, cb_entry, k, j * kt + k, 0, false, 1, kt, kt);
        }
        tile_regs_commit();
        tile_regs_wait();
        // The column leaves for the successor first; its FP32 copy follows off the relay's path.
        pack_reconfig_data_format(cb_eout);
        for (uint32_t r = 0; r < kt; ++r) {
            pack_tile(r, cb_eout);
        }
        cb_push_back(cb_eout, kt);
        cb_reserve_back(cb_state, kt);
        pack_reconfig_data_format(cb_state);
        for (uint32_t r = 0; r < kt; ++r) {
            pack_tile(r, cb_state);
        }
        tile_regs_release();
        cb_push_back(cb_state, kt);
    }
    copy_tiles(cb_entry, cb_state, state_tiles);
    cb_pop_front(cb_a, state_tiles);
    cb_pop_front(cb_entry, state_tiles);
    cb_pop_front(cb_b, state_tiles);
    // Every rank converts its final landing; only the first rank's holds the last rank's final state.
    cb_wait_front(cb_final, state_tiles);
    copy_tiles(cb_final, cb_state, state_tiles);
    cb_pop_front(cb_final, state_tiles);
}
