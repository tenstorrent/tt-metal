// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

constexpr uint32_t cb_e = get_compile_time_arg_val(0);
constexpr uint32_t cb_b = get_compile_time_arg_val(1);
constexpr uint32_t cb_ein = get_compile_time_arg_val(2);
constexpr uint32_t cb_eout = get_compile_time_arg_val(3);
constexpr uint32_t kt = get_compile_time_arg_val(4);
// The same landed entry state as cb_ein, unpacked straight to DST (FP32, exact).
constexpr uint32_t cb_ein_exact = get_compile_time_arg_val(5);
// One BF16 identity tile: B enters the sum as I @ B.
constexpr uint32_t cb_identity = get_compile_time_arg_val(6);
constexpr uint32_t state_tiles = kt * kt;
constexpr uint32_t dst_tiles =
    ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();
// A tile column of the next state is formed in one DST pass (the op checks K against the DST).
static_assert(kt <= dst_tiles);

// The next entry state S_out = A S_in + B for one head, from the complement form E = A - I: S_out = S_in + E S_in + B,
// summed in the FP32 DST. The exact copy of S_in unpacks to DST losslessly; E S_in accumulates on top at the affine
// prefix's fidelity, so the matmul's source-register copy of S_in reaches only the small E S_in term; B accumulates as
// I @ B, exact at HiFi2 and above (a matmul's in1 goes to srcA, whose HiFi2 phases cover a BF16 mantissa, and 1.0 is
// exact in srcB).
void kernel_main() {
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_e, cb_ein, cb_eout);
    cb_wait_front(cb_e, state_tiles);
    cb_wait_front(cb_b, state_tiles);
    cb_wait_front(cb_identity, 1);
    // Per output tile column (column-major state): column j needs only column j of S_in, so it is computed as soon as
    // that column lands.
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
    cb_pop_front(cb_identity, 1);
}
