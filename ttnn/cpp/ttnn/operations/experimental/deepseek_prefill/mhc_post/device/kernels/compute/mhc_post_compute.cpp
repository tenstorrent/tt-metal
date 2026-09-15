// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused mHC post-mix on one core. Per (token-tile, column-tile) work unit:
//
//     out_j = post_j * y + sum_i comb_(i,j) * residual_i        for j in [0, n)
//
// post_j and comb_(i,j) vary per token (tile row) but not per column, so each is first expanded
// into a full tile: `coeff_tile @ consts[k]` copies column k across the whole tile, because
// consts[k] has row k all ones and is zero elsewhere. That is exact -- the products dropped are
// products with a true zero -- and it needs no sub-tile broadcast LLK, which cannot address an
// arbitrary column. The expansion is done once per token-tile and reused for every column tile.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

namespace {

constexpr uint32_t CB_IN = 0;      // c_0: y then the n residual-stream tiles
constexpr uint32_t CB_PC = 1;      // c_1: raw post tile, raw comb tile
constexpr uint32_t CB_CONSTS = 2;  // c_2: n*n column-broadcast tiles, resident
constexpr uint32_t CB_COEF = 3;    // c_3: n broadcast post_j then n*n broadcast comb_(i,j)
constexpr uint32_t CB_OUT = 16;

}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(CB_COEF, CB_IN, CB_OUT);  // must precede any other compute work
    const uint32_t num_units = get_arg_val<uint32_t>(0);
    const uint32_t start_unit = get_arg_val<uint32_t>(1);
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t col_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t num_coef = n + n * n;

    CircularBuffer cb_consts(CB_CONSTS), cb_pc(CB_PC), cb_coef(CB_COEF), cb_in(CB_IN), cb_out(CB_OUT);
    cb_consts.wait_front(n * n);

    uint32_t cached_t0 = 0xFFFFFFFFu;
    for (uint32_t w = 0; w < num_units; ++w) {
        const uint32_t t0 = (start_unit + w) / col_tiles;

        if (t0 != cached_t0) {
            if (cached_t0 != 0xFFFFFFFFu) {
                cb_coef.pop_front(num_coef);
            }
            cached_t0 = t0;

            cb_pc.wait_front(2);
            cb_coef.reserve_back(num_coef);
            reconfig_data_format(CB_PC, CB_CONSTS);
            matmul_init(CB_PC, CB_CONSTS);
            for (uint32_t k = 0; k < num_coef; ++k) {
                // post supplies the first n coefficients (its column j), comb the remaining n*n.
                const uint32_t src_tile = k < n ? 0 : 1;
                const uint32_t col = k < n ? k : k - n;
                tile_regs_acquire();
                matmul_tiles(CB_PC, CB_CONSTS, src_tile, col, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, CB_COEF);
                tile_regs_release();
            }
            cb_coef.push_back(num_coef);
            cb_pc.pop_front(2);
            cb_coef.wait_front(num_coef);
        }

        cb_in.wait_front(1 + n);
        cb_out.reserve_back(n);
        reconfig_data_format(CB_COEF, CB_IN);
        for (uint32_t j = 0; j < n; ++j) {
            // The two-argument mul_tiles_init leaves acc_to_dest set, so every mul_tiles adds its
            // product into the DEST slot instead of overwriting it. The whole mix therefore lands
            // in slot 0 on the FPU, with no scratch slots and no SFPU pass to fold terms together.
            tile_regs_acquire();
            mul_tiles_init(CB_COEF, CB_IN);
            mul_tiles(CB_COEF, CB_IN, j, 0, 0);  // post_j * y
            for (uint32_t i = 0; i < n; ++i) {
                mul_tiles(CB_COEF, CB_IN, n + i * n + j, 1 + i, 0);  // += comb_(i,j) * residual_i
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_OUT);
            tile_regs_release();
        }
        cb_out.push_back(n);
        cb_in.pop_front(1 + n);
    }
}
