// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

constexpr uint32_t cb_a = get_compile_time_arg_val(0);
constexpr uint32_t cb_b = get_compile_time_arg_val(1);
constexpr uint32_t cb_ein = get_compile_time_arg_val(2);
constexpr uint32_t cb_eout = get_compile_time_arg_val(3);
constexpr uint32_t kt = get_compile_time_arg_val(4);
constexpr uint32_t state_tiles = kt * kt;

// E_out = A E_in + B for one head, accumulated in the FP32 DST at the affine prefix's fidelity.
void kernel_main() {
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_ein, cb_eout);
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
        cb_wait_front(cb_ein, (j + 1) * kt);
        reconfig_data_format<SrcOrder::Reverse>(cb_a, cb_ein);
        matmul_block_init(cb_a, cb_ein, false, 1, kt, kt);
        for (uint32_t k = 0; k < kt; ++k) {
            matmul_block(cb_a, cb_ein, k, j * kt + k, 0, false, 1, kt, kt);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t r = 0; r < kt; ++r) {
            pack_tile(r, cb_eout);
        }
        tile_regs_release();
        cb_push_back(cb_eout, kt);
    }
    cb_pop_front(cb_a, state_tiles);
    cb_pop_front(cb_ein, state_tiles);
    cb_pop_front(cb_b, state_tiles);
}
