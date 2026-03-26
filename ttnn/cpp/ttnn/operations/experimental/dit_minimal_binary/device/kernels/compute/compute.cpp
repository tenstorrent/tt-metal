// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// On-the-fly tilize + element-wise binary + untilize compute kernel.
//
// For each row-block of TILE_HEIGHT rows the kernel performs four phases:
//
//   Phase 1 — Tilize A (ntiles_per_row tiles, one at a time):
//     tilize_block(CB_A_RM → CB_A_TILED)
//
//   Phase 2 — Tilize B (switch source CB, ntiles_per_row tiles):
//     tilize_init_short_with_dt switch + tilize_block(CB_B_RM → CB_B_TILED)
//
//   Phase 3 — Binary op (tile-by-tile):
//     add_tiles / mul_tiles (CB_A_TILED + CB_B_TILED → CB_OUT_TILED)
//     FP32 path uses SFPU (copy_tile + add_binary_tile / mul_binary_tile)
//
//   Phase 4 — Untilize (one tile at a time, pop-to-advance pattern):
//     pack_untilize_block<1, ntiles_per_row_ct>(CB_OUT_TILED → CB_OUT_RM)
//     Bypasses SrcA (19-bit/TF32) via UnpackToDestEn; correct for both bf16 and fp32.
//
// Compile-time args: [ntiles_per_row_ct]
//   ntiles_per_row_ct = last_dim / TILE_WIDTH (baked into pack_untilize template param)
// Compile-time defines: BINARY_OP_INIT, BINARY_OP, IS_FP32 (optional).
// RT args: [num_sticks, ntiles_per_row]
//   num_sticks     = number of sticks (rows) for this core
//   ntiles_per_row = last_dim / TILE_WIDTH

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tilize.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack_untilize.h"

constexpr auto CB_A_RM = tt::CBIndex::c_0;
constexpr auto CB_B_RM = tt::CBIndex::c_1;
constexpr auto CB_A_TILED = tt::CBIndex::c_2;
constexpr auto CB_B_TILED = tt::CBIndex::c_3;
constexpr auto CB_OUT_TILED = tt::CBIndex::c_4;
constexpr auto CB_OUT_RM = tt::CBIndex::c_16;

void kernel_main() {
    constexpr uint32_t ntiles_per_row_ct = get_compile_time_arg_val(0);

    const uint32_t num_sticks = get_arg_val<uint32_t>(0);
    const uint32_t ntiles_per_row = get_arg_val<uint32_t>(1);

    binary_op_init_common(CB_A_RM, CB_B_RM, CB_OUT_TILED);

    constexpr uint32_t TILE_HEIGHT = 32;
    for (uint32_t blk = 0; blk < num_sticks; blk += TILE_HEIGHT) {
        // ============================================================
        // Phase 1: Tilize A tiles (ntiles_per_row tiles, 1 at a time)
        // ============================================================
        cb_wait_front(CB_A_RM, ntiles_per_row);
        cb_reserve_back(CB_A_TILED, ntiles_per_row);

        reconfig_data_format(CB_A_RM, CB_A_RM);
        pack_reconfig_data_format(CB_A_TILED);
        tilize_init(CB_A_RM, ntiles_per_row, CB_A_TILED);

        tilize_block(CB_A_RM, ntiles_per_row, CB_A_TILED);

        cb_push_back(CB_A_TILED, ntiles_per_row);
        cb_pop_front(CB_A_RM, ntiles_per_row);

        // ============================================================
        // Phase 2: Tilize B tiles — switch source CB to CB_B_RM
        // ============================================================

        tilize_init_short_with_dt(CB_A_RM, CB_B_RM, ntiles_per_row, CB_B_TILED);

        cb_wait_front(CB_B_RM, ntiles_per_row);
        cb_reserve_back(CB_B_TILED, ntiles_per_row);

        tilize_block(CB_B_RM, ntiles_per_row, CB_B_TILED);

        cb_push_back(CB_B_TILED, ntiles_per_row);
        cb_pop_front(CB_B_RM, ntiles_per_row);

        // ============================================================
        // Phase 3: Binary op — switch from tilize to binary-op mode
        // ============================================================
        tilize_uninit(CB_B_RM, CB_B_TILED);

        binary_op_init_common(CB_A_TILED, CB_B_TILED, CB_OUT_TILED);
#ifdef IS_FP32
        BINARY_OP_INIT();
#else
        BINARY_OP_INIT(CB_A_TILED, CB_B_TILED);
#endif

        for (uint32_t j = 0; j < ntiles_per_row; ++j) {
            cb_wait_front(CB_A_TILED, 1);
            cb_wait_front(CB_B_TILED, 1);
            cb_reserve_back(CB_OUT_TILED, 1);

            tile_regs_acquire();
#ifdef IS_FP32
            copy_tile_to_dst_init_short(CB_A_TILED);
            copy_tile(CB_A_TILED, 0, 0);
            copy_tile_to_dst_init_short(CB_B_TILED);
            copy_tile(CB_B_TILED, 0, 1);
            BINARY_OP(0, 1, 0);
            static_assert(DST_ACCUM_MODE > 0, "DST_ACCUM_MODE must be enabled for fp32");
#else
            BINARY_OP(CB_A_TILED, CB_B_TILED, 0, 0, 0);
#endif

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_OUT_TILED);
            tile_regs_release();

            cb_push_back(CB_OUT_TILED, 1);
            cb_pop_front(CB_A_TILED, 1);
            cb_pop_front(CB_B_TILED, 1);
        }

        // ============================================================
        // Phase 4: Untilize — one tile at a time, popping after each to advance
        // fifo_rd_ptr.  pack_untilize_block uses llk_unpack_A with UnpackToDestEn,
        // bypassing SrcA (19-bit / TF32), which is required for fp32 correctness
        // and harmless for bf16.
        // ============================================================
        reconfig_data_format(CB_OUT_TILED, CB_OUT_TILED);
        pack_untilize_init<1, ntiles_per_row_ct>(CB_OUT_TILED, CB_OUT_RM);
        cb_reserve_back(CB_OUT_RM, ntiles_per_row);
        for (uint32_t c = 0; c < ntiles_per_row_ct; ++c) {
            cb_wait_front(CB_OUT_TILED, 1);
            pack_untilize_block<1, ntiles_per_row_ct>(CB_OUT_TILED, 1, CB_OUT_RM, c);
            cb_pop_front(CB_OUT_TILED, 1);
        }
        cb_push_back(CB_OUT_RM, ntiles_per_row);
        pack_untilize_uninit(CB_OUT_RM);
    }
}
