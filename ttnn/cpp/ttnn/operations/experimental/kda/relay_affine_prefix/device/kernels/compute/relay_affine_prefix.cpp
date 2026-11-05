// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/compute/matmul_subblock.hpp"

constexpr uint32_t cb_e = get_compile_time_arg_val(0);
constexpr uint32_t cb_b = get_compile_time_arg_val(1);
constexpr uint32_t cb_ein = get_compile_time_arg_val(2);
constexpr uint32_t cb_eout = get_compile_time_arg_val(3);
constexpr uint32_t kt = get_compile_time_arg_val(4);
// The same landed entry state as cb_ein, unpacked straight to DST (FP32, exact).
constexpr uint32_t cb_ein_exact = get_compile_time_arg_val(5);
constexpr uint32_t state_tiles = kt * kt;
constexpr uint32_t dst_tiles =
    ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();
// Output tiles per DST pass: slots [0, rows) form them, slots [rows, 2 rows) hold their exact entry tiles.
constexpr uint32_t rows = kda::largest_divisor_at_most(kt, dst_tiles / 2);

// The next entry state S_out = A S_in + B for one head, from the complement form E = A - I: S_out = S_in + B + E S_in,
// summed in the FP32 DST. B (BF16) and the exact copy of S_in unpack to DST losslessly and are added on the SFPU; the
// K-sum E S_in accumulates on top at the affine prefix's fidelity, so the matmul's source-register copy of S_in reaches
// only the small E S_in term.
void kernel_main() {
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_e, cb_ein, cb_eout);
    cb_wait_front(cb_e, state_tiles);
    cb_wait_front(cb_b, state_tiles);
    // Per output tile column (column-major state): column j needs only column j of S_in, so it is computed as soon as
    // that column lands.
    for (uint32_t j = 0; j < kt; ++j) {
        cb_reserve_back(cb_eout, kt);
        cb_wait_front(cb_ein, (j + 1) * kt);
        cb_wait_front(cb_ein_exact, (j + 1) * kt);
        for (uint32_t first_row = 0; first_row < kt; first_row += rows) {
            tile_regs_acquire();
            reconfig_data_format_srca(cb_b);
            copy_init(cb_b);
            for (uint32_t r = 0; r < rows; ++r) {
                copy_tile(cb_b, (first_row + r) * kt + j, r);
            }
            reconfig_data_format_srca(cb_ein_exact);
            copy_init(cb_ein_exact);
            for (uint32_t r = 0; r < rows; ++r) {
                copy_tile(cb_ein_exact, j * kt + first_row + r, rows + r);
            }
            add_binary_tile_init();
            for (uint32_t r = 0; r < rows; ++r) {
                add_binary_tile(r, rows + r, r);
            }
            reconfig_data_format<SrcOrder::Reverse>(cb_e, cb_ein);
            matmul_block_init(cb_e, cb_ein, false, 1, rows, kt);
            for (uint32_t k = 0; k < kt; ++k) {
                matmul_block(cb_e, cb_ein, first_row * kt + k, j * kt + k, 0, false, 1, rows, kt);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t r = 0; r < rows; ++r) {
                pack_tile(r, cb_eout);
            }
            tile_regs_release();
        }
        cb_push_back(cb_eout, kt);
    }
    cb_pop_front(cb_e, state_tiles);
    cb_pop_front(cb_ein, state_tiles);
    cb_pop_front(cb_ein_exact, state_tiles);
    cb_pop_front(cb_b, state_tiles);
}
