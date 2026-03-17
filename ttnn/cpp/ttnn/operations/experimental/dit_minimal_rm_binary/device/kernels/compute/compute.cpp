// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
//   Phase 4 — Untilize (full row-width at once):
//     untilize_block(CB_OUT_TILED → CB_OUT_RM, full_ct_dim = ntiles_per_row)
//
// Compile-time defines: BINARY_OP_INIT, BINARY_OP, IS_FP32 (optional).
// RT args: [num_blocks, ntiles_per_row]
//   num_blocks    = number of TILE_HEIGHT-row blocks for this core
//   ntiles_per_row = last_dim / TILE_WIDTH

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tilize.h"
#include "api/compute/untilize.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/fill.h"

#include "api/debug/dprint.h"
#include "api/debug/dprint_tensix.h"

constexpr auto CB_A_RM = tt::CBIndex::c_0;
constexpr auto CB_B_RM = tt::CBIndex::c_1;
constexpr auto CB_A_TILED = tt::CBIndex::c_2;
constexpr auto CB_B_TILED = tt::CBIndex::c_3;
constexpr auto CB_OUT_TILED = tt::CBIndex::c_4;
constexpr auto CB_OUT_RM = tt::CBIndex::c_16;

template <typename To>
ALWI auto get_pointer_to_cb_data(uint32_t cb_id, uint32_t tile_index) -> To* {
    return reinterpret_cast<To*>(get_tile_address(cb_id, tile_index));
}

void print_cb_data(uint32_t cb_id, uint32_t tile_index) {
    volatile uint16_t* ptr = get_pointer_to_cb_data<uint16_t>(cb_id, tile_index);
    for (int subtile_i = 0; subtile_i < 2; subtile_i++) {
        // Iterate through 16 rows within each subtile row
        for (int local_row = 0; local_row < 16; local_row++) {
            // Calculate the actual row in original matrix
            int row = subtile_i * 16 + local_row;
            // Iterate through 2x2 subtiles horizontally
            for (int subtile_j = 0; subtile_j < 2; subtile_j++) {
                // Iterate through 16 columns within each subtile
                for (int local_col = 0; local_col < 16; local_col++) {
                    // Calculate the actual column in original matrix
                    int col = subtile_j * 16 + local_col;
                    // Calculate index using only multiplication and addition
                    auto index = local_row * 16 + local_col + subtile_i * 512 + subtile_j * 256;
                    // const uint32_t message = read_tile_value(input_val_cb_index, /*tile_index=*/take,
                    // /*element_offset=*/index);ptr
                    UNPACK(DPRINT << BF16(ptr[index]) << ", ");
                }
            }
            UNPACK(DPRINT << ENDL());
        }
    }  // subtile_i
}

void print_cb_row_data(uint32_t cb_id, uint32_t tile_index, uint32_t row_index) {
    volatile uint16_t* ptr = get_pointer_to_cb_data<uint16_t>(cb_id, tile_index);

    int subtile_i = 0;  // only print top faces
    if (row_index >= 16) {
        subtile_i = 1;
    }

    // Iterate through 16 rows within each subtile row
    int local_row = row_index % 16;
    // Calculate the actual row in original matrix
    int row = subtile_i * 16 + local_row;
    // Iterate through 2x2 subtiles horizontally
    for (int subtile_j = 0; subtile_j < 2; subtile_j++) {
        // Iterate through 16 columns within each subtile
        for (int local_col = 0; local_col < 16; local_col++) {
            // Calculate the actual column in original matrix
            int col = subtile_j * 16 + local_col;
            // Calculate index using only multiplication and addition
            auto index = local_row * 16 + local_col + subtile_i * 512 + subtile_j * 256;
            // const uint32_t message = read_tile_value(input_val_cb_index, /*tile_index=*/take,
            // /*element_offset=*/index);ptr
            UNPACK(DPRINT << BF16(ptr[index]) << ", ");
        }
    }
    UNPACK(DPRINT << ENDL());
}

void kernel_main() {
    const uint32_t num_blocks = get_arg_val<uint32_t>(0);
    const uint32_t ntiles_per_row = get_arg_val<uint32_t>(1);

    DPRINT << "num blocks = " << num_blocks << ENDL();
    DPRINT << "ntiles per row = " << ntiles_per_row << ENDL();

    binary_op_init_common(CB_A_RM, CB_B_RM, CB_OUT_TILED);

    // ---- Initial hardware setup: enter tilize-A mode ----
    tilize_init(CB_A_RM, 1, CB_A_TILED);

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        // ============================================================
        // Phase 1: Tilize A tiles (ntiles_per_row tiles, 1 at a time)
        // ============================================================
        for (uint32_t j = 0; j < ntiles_per_row; ++j) {
            cb_wait_front(CB_A_RM, 1);
            cb_reserve_back(CB_A_TILED, 1);

            auto cb_a_ptr = get_pointer_to_cb_data<uint16_t>(CB_A_RM, 0);
            UNPACK(DPRINT << "j = " << j << ", cb a ptr = " << cb_a_ptr << ENDL(););

            tilize_block(CB_A_RM, 1, CB_A_TILED);

            if (j == 0) {
                UNPACK(DPRINT << "cb a tiled j=0 row0:" << ENDL(););
                print_cb_row_data(CB_A_TILED, 0, 0);
            }

            cb_push_back(CB_A_TILED, 1);
            cb_pop_front(CB_A_RM, 1);
        }

        // ============================================================
        // Phase 2: Tilize B tiles — switch source CB to CB_B_RM
        // ============================================================

        UNPACK(DPRINT << "====Phase 2=====" << ENDL(););
        tilize_init_short_with_dt(CB_A_RM, CB_B_RM, 1, CB_B_TILED);

        for (uint32_t j = 0; j < ntiles_per_row; ++j) {
            cb_wait_front(CB_B_RM, 1);
            cb_reserve_back(CB_B_TILED, 1);

            tilize_block(CB_B_RM, 1, CB_B_TILED);

            if (j == 0) {
                UNPACK(DPRINT << "cb b tiled j=0 row0:" << ENDL(););
                print_cb_row_data(CB_B_TILED, 0, 0);
            }

            // DEBUG: Fill cb_b with 1.f
            // tile_regs_acquire();
            // fill_tile_init();
            // fill_tile(0, 1.f);
            // tile_regs_commit();
            // tile_regs_wait();
            // pack_tile(0, CB_B_TILED);
            // tile_regs_release();

            cb_push_back(CB_B_TILED, 1);
            cb_pop_front(CB_B_RM, 1);
        }

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

            if (j == 0 || j == 1) {
                UNPACK(DPRINT << "cb a tiled j=" << j << " row0:" << ENDL(););
                print_cb_row_data(CB_A_TILED, 0, 0);
                UNPACK(DPRINT << "cb b tiled j=" << j << " row0:" << ENDL(););
                print_cb_row_data(CB_B_TILED, 0, 0);
            }

            tile_regs_acquire();
#ifdef IS_FP32
            copy_tile_to_dst_init_short(CB_A_TILED);
            copy_tile(CB_A_TILED, 0, 0);
            copy_tile_to_dst_init_short(CB_B_TILED);
            copy_tile(CB_B_TILED, 0, 1);
            BINARY_OP(0, 1, 0);
#else
            BINARY_OP(CB_A_TILED, CB_B_TILED, 0, 0, 0);
#endif
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_OUT_TILED);
            tile_regs_release();

            if (j == 0 || j == 1) {
                UNPACK(DPRINT << "cb out tiled j=" << j << " row0:" << ENDL(););
                // tile_index=j is correct here: fifo_rd_ptr(CB_OUT_TILED) stays at base (no pops in Phase 3)
                print_cb_row_data(CB_OUT_TILED, j, 0);
            }

            cb_push_back(CB_OUT_TILED, 1);
            cb_pop_front(CB_A_TILED, 1);
            cb_pop_front(CB_B_TILED, 1);
        }

        // ============================================================
        // Phase 4: Untilize — full row at once (full_ct_dim = ntiles_per_row)
        // ============================================================
        untilize_init(CB_OUT_TILED);

        // untilize_block produces ntiles_per_row tile-sized pages in row-major
        // order.  Row k starts at byte offset k * row_size_bytes from the block
        // base; the writer pops all ntiles_per_row pages at once.
        cb_wait_front(CB_OUT_TILED, ntiles_per_row);
        cb_reserve_back(CB_OUT_RM, ntiles_per_row);

        UNPACK(DPRINT << "cb out tiled tile0 row0:" << ENDL(););
        print_cb_row_data(CB_OUT_TILED, 0, 0);
        UNPACK(DPRINT << "cb out tiled tile1 row0:" << ENDL(););
        print_cb_row_data(CB_OUT_TILED, 1, 0);

        untilize_block(CB_OUT_TILED, ntiles_per_row, CB_OUT_RM);

        {
            // CB_OUT_RM: ntiles_per_row tile-sized (2048-byte) pages, row-major.
            // p0[0] = row0 col0 (always correct).
            // p1[0] = tile-index 1 = byte-offset 2048 = row (2048/row_size_bytes).
            // For ntiles_per_row=32: row 1 (padding). For ntiles_per_row=8: row 4.
            volatile uint16_t* p0 = get_pointer_to_cb_data<uint16_t>(CB_OUT_RM, 0);
            volatile uint16_t* p1 = get_pointer_to_cb_data<uint16_t>(CB_OUT_RM, 1);
            UNPACK(DPRINT << "cb out rm tile0[0]:" << BF16(p0[0]) << " [1]:" << BF16(p0[1]) << ENDL();
                   DPRINT << "cb out rm tile1[0]:" << BF16(p1[0]) << " [1]:" << BF16(p1[1]) << ENDL(););
        }

        cb_push_back(CB_OUT_RM, ntiles_per_row);
        cb_pop_front(CB_OUT_TILED, ntiles_per_row);

        // ============================================================
        // Switch back to tilize-A mode for the next block.
        // For the last block this uninit is still harmless.
        // ============================================================
        untilize_uninit(CB_OUT_TILED);
        tilize_init_short_with_dt(CB_B_RM, CB_A_RM, 1, CB_A_TILED);
    }
}
