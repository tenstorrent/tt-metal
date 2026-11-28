// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Bake-off candidate (perf_experiments/tensix_tilize_bf16): the bf16 ROW_MAJOR -> TILE reorder on the
// Tensix. cb_row_stage (32 rows per tile-row = block_cols_tiles bf16 tile pages, reader -> compute)
// -> unpack-tilize to SrcA -> MOVA2D to a 16-bit DEST -> pack Float16_b -> cb_tile_block (bf16
// tiles, compute -> writer). Every block moves the nominal block_rows_tiles tile-rows, so the core's
// work is one pass of that many tile-rows: init/uninit run once.
//
// Bit identity (measured, test_tensix_tilize.py test_diag / test_exact*):
// - Float16_b CBs are NOT bit-exact, whatever the Src zero flag says: with the flag pinned to preserve
//   (TT_ZERO_FLAG 1, 2 or 3), and even on a plain unpack-A copy with no tilize (IMPL copydiag), every
//   exponent-0 pattern (-0.0, all 255 subnormals) comes back +0 and every NaN payload comes back +-inf.
//   The canonicalization is in the Float16_b unpack/ALU/pack path, not only in MOVA2D's zero flag.
// - UInt16 CBs (candidate.py "-u16": the same bytes, declared Integer 16) are bit-exact for all 65536
//   patterns: the unpacker, MOVA2D and packer move raw 16-bit integers, and compute_kernel_hw_startup
//   already disables the zero substitution for a UInt16 operand (requires_disabled_src_zero_flag), so
//   no flag call is needed (TT_ZERO_FLAG 0). compute_kernel_lib::tilize falls back to the standard
//   tilize for UInt16 (its fast path is gated on Float32 / Float16_b); the BH fast-tilize compute API
//   run on UInt16 CBs (IMPL 2) misplaces faces 2-3 and every later tile of a >= 2-tile row: incorrect.
//
// TT_ZERO_FLAG: 1 = ckernel::math::_configure_preserve_zero_flag_state_() after the hardware startup.
//   2 = copy_init(cb_row_stage) (pure compute API; its side effect pins preserve for a Float16_b source).
//   3 = the raw call after tilize_init (IMPL 1/2). 0 = no call.
// TT_TILIZE_IMPL: 0 = compute_kernel_lib::tilize (NoReconfigure). 1 = compute API tilize_init /
//   tilize_block / tilize_uninit. 2 = compute API fast_tilize_* directly. 3 = copy diagnostic.
//   4 = compute API tilize of the live columns only. 5 = per-tile raw-LLK tilize with a push per tile.
// The op's own g6_to_layout_tilize_compute.cpp (copied verbatim as op_tilize_compute.cpp, IMPL "opk"
// in candidate.py) is exact on UInt16 CBs as it stands.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "g6_to_layout_block_walk.hpp"

#ifndef TT_ZERO_FLAG
#define TT_ZERO_FLAG 1
#endif
#ifndef TT_TILIZE_IMPL
#define TT_TILIZE_IMPL 0
#endif

using namespace g6_to_layout;

void kernel_main() {
    constexpr uint32_t cb_row_stage = get_compile_time_arg_val(Geometry::next_ct_arg + 0);
    constexpr uint32_t cb_tile_block = get_compile_time_arg_val(Geometry::next_ct_arg + 1);
    constexpr uint32_t cols = Geometry::block_cols_tiles;

    uint32_t tile_rows = 0;
    CoreBlocks<false> blocks;
    Block block;
    while (blocks.next(block)) {
        tile_rows += Geometry::block_rows_tiles;
    }
    if (tile_rows == 0) {
        return;
    }

    compute_kernel_hw_startup(cb_row_stage, cb_tile_block);
#if TT_ZERO_FLAG == 1
    MATH((ckernel::math::_configure_preserve_zero_flag_state_()));
#elif TT_ZERO_FLAG == 2
    copy_init(cb_row_stage);
#endif

#if TT_TILIZE_IMPL == 0
    using namespace compute_kernel_lib::tilize_config;
    compute_kernel_lib::tilize<
        cols,
        cb_row_stage,
        cb_tile_block,
        InitUninitMode::InitAndUninit,
        WaitMode::WaitBlock,
        ReconfigureRegisterDatatypeMode::NoReconfigure>(tile_rows);
#elif TT_TILIZE_IMPL == 1
    tilize_init(cb_row_stage, cols, cb_tile_block);
#if TT_ZERO_FLAG == 3
    MATH((ckernel::math::_configure_preserve_zero_flag_state_()));
#endif
    for (uint32_t r = 0; r < tile_rows; ++r) {
        cb_wait_front(cb_row_stage, cols);
        cb_reserve_back(cb_tile_block, cols);
        tilize_block(cb_row_stage, cols, cb_tile_block);
        cb_push_back(cb_tile_block, cols);
        cb_pop_front(cb_row_stage, cols);
    }
    tilize_uninit(cb_row_stage, cb_tile_block);
#elif TT_TILIZE_IMPL == 2
    // The compute API fast tilize directly: the helper gates fast tilize on Float32 / Float16_b CB
    // formats, while the BH fast-tilize LLKs only move 16-bit datums (UInt16 CBs pass the gate here).
    fast_tilize_init(cb_row_stage, cols, cb_tile_block);
#if TT_ZERO_FLAG == 3
    MATH((ckernel::math::_configure_preserve_zero_flag_state_()));
#endif
    for (uint32_t r = 0; r < tile_rows; ++r) {
        cb_wait_front(cb_row_stage, cols);
        cb_reserve_back(cb_tile_block, cols);
        fast_tilize_block(cb_row_stage, cols, cb_tile_block);
        cb_push_back(cb_tile_block, cols);
        cb_pop_front(cb_row_stage, cols);
    }
    fast_tilize_uninit(cb_row_stage, cb_tile_block, cols);
#elif TT_TILIZE_IMPL == 4 || TT_TILIZE_IMPL == 5
    // Live columns only: a ragged block (block.cols < block_cols_tiles) and its padding tile-rows tilize
    // nothing past the live tiles; the nominal pages are still pushed (the writer and the ring stay
    // block-aligned). IMPL 4: compute API tilize_block(live) and one push per tile-row. IMPL 5: one tile
    // per DEST section with a push per tile, so the writer starts on tile 0 while tile 1 is tilized
    // (raw LLK: tilize_block packs a whole tile-row before the caller can push any of it).
    tilize_init(cb_row_stage, cols, cb_tile_block);
    CoreBlocks<false> walk;
    while (walk.next(block)) {
        for (uint32_t i = 0; i < Geometry::block_rows_tiles; ++i) {
            const uint32_t live = i < block.rows ? block.cols : 0;
            cb_wait_front(cb_row_stage, cols);
#if TT_TILIZE_IMPL == 4
            cb_reserve_back(cb_tile_block, cols);
            if (live > 0) {
                tilize_block(cb_row_stage, live, cb_tile_block);
            }
            cb_push_back(cb_tile_block, cols);
#else
            for (uint32_t t = 0; t < cols; ++t) {
                cb_reserve_back(cb_tile_block, 1);
                if (t < live) {
                    UNPACK((llk_unpack_tilize(cb_row_stage, t)));
                    MATH((llk_math_wait_for_dest_available()));
                    PACK((llk_packer_wait_for_math_done()));
                    MATH((llk_math_eltwise_unary_datacopy<
                          DataCopyType::A2D,
                          DST_ACCUM_MODE,
                          BroadcastType::NONE,
                          UnpackToDestEn>(0, cb_row_stage)));
                    PACK((llk_pack<DST_ACCUM_MODE, true, PackMode::Default>(0, cb_tile_block, 0)));
                    MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
                    PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
                }
                cb_push_back(cb_tile_block, 1);
            }
#endif
            cb_pop_front(cb_row_stage, cols);
        }
    }
    tilize_uninit(cb_row_stage, cb_tile_block);
#else
    // Diagnostic only (never timed): no tilize, every stage page copied verbatim as a tile
    // (unpack A -> MOVA2D -> pack), to tell the tilize unpacker apart from the copy path.
    copy_init(cb_row_stage);
#if TT_ZERO_FLAG == 0
    MATH((ckernel::math::_configure_src_zero_flag_(false)));
#endif
    for (uint32_t r = 0; r < tile_rows; ++r) {
        cb_wait_front(cb_row_stage, cols);
        cb_reserve_back(cb_tile_block, cols);
        for (uint32_t t = 0; t < cols; ++t) {
            tile_regs_acquire();
            copy_tile(cb_row_stage, t, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tile_block, t);
            tile_regs_release();
        }
        cb_push_back(cb_tile_block, cols);
        cb_pop_front(cb_row_stage, cols);
    }
#endif
}
