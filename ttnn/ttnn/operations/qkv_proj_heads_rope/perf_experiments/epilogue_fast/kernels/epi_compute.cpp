// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Isolated qkv_proj_heads_rope epilogue: Float32 partials (block_m x n_tiles, resident) + rotary table
// (block_m rows x [cos_lo, -sin_lo, sin_hi, cos_hi], resident) -> bf16 rows into two alternating output
// CBs (depth 2 rows each), repeated `iters` times.
//
// variant 0 (baseline): the op's code - eltwise_chain per head, grid(2 out, 2 terms), PerRow DEST
//   accumulation -> one acquire/commit/wait/release + reserve/push per OUTPUT TILE.
// variant 1/2 (raw window): bypasses eltwise_chain because PerRow DEST accumulation is one accumulated
//   output tile per DEST window and the rotary's A/B index maps (base + 2*(o/2) + t, table + 2*(o%2) + t)
//   are not linear in the output index, so no chain shape covers a row. ELWMUL always accumulates into
//   DEST (the released half is zeroed by the packer), so several output tiles accumulate in one window:
//   window_heads heads (2 * window_heads tiles) per acquire, the row reserved/pushed once.
// variant 3 (v cores, baseline): copy chain Float32 -> bf16 per row.
// variant 4: variant 1 with tile_src_mul (whole-tile src banks, face-major, see tile_src_mul.hpp).
// variant 5: variant 1 with tile_src_mul phase-major (one MOP per tile product).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#if EPI_DIAG_B_NOP
#define TILE_SRC_DIAG_B_NOP EPI_DIAG_B_NOP
#endif
#include "tile_src_mul.hpp"

void kernel_main() {
    constexpr uint32_t cb_matmul_partials = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rotary_table = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out_tiles_noc0 = get_compile_time_arg_val(3);
    constexpr uint32_t block_m = get_compile_time_arg_val(4);
    constexpr uint32_t heads_per_core = get_compile_time_arg_val(5);
    constexpr uint32_t seq_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t iters = get_compile_time_arg_val(7);
    constexpr uint32_t variant = get_compile_time_arg_val(8);
    constexpr uint32_t window_heads = get_compile_time_arg_val(9);
    // Diagnostics only (wrong output): 1 = pack one tile per DEST window, 2 = one product per output tile.
    constexpr uint32_t diag = get_compile_time_arg_val(10);
    constexpr uint32_t tiles_per_head = 2;
    constexpr uint32_t n_tiles = heads_per_core * tiles_per_head;
    constexpr uint32_t partials_tiles = block_m * n_tiles;
    constexpr uint32_t table_tiles_per_row = 2 * tiles_per_head;
    constexpr bool is_rotary = variant != 3;

    using namespace compute_kernel_lib;

    compute_kernel_hw_startup(cb_matmul_partials, cb_rotary_table, cb_out_tiles);

    DataflowBuffer partials_dfb(cb_matmul_partials);
    DataflowBuffer table_dfb(cb_rotary_table);
    DataflowBuffer out_dfb(cb_out_tiles);

    using RotaryMulAcc = BinaryFpu<
        BinaryFpuOp::Mul,
        input(
            cb_matmul_partials,
            WaitPolicy::None,
            PopPolicy::None,
            InputTileMapping::Row,
            DataFormatReconfig::Disabled,
            TileAddressing::Offset),
        input(
            cb_rotary_table,
            WaitPolicy::None,
            PopPolicy::None,
            InputTileMapping::Block,
            DataFormatReconfig::Disabled,
            TileAddressing::Offset),
        Dst::D0,
        DestAccumulation::PerRow>;
    using RotaryPack = PackTile<output(
        cb_out_tiles,
        ReservePolicy::PerOuter,
        PushPolicy::PerOuter,
        DataFormatReconfig::Disabled,
        TileAddressing::Direct,
        DestAccumulation::PerRow)>;
    using RotaryPackNoc0 = PackTile<output(
        cb_out_tiles_noc0,
        ReservePolicy::PerOuter,
        PushPolicy::PerOuter,
        DataFormatReconfig::Disabled,
        TileAddressing::Direct,
        DestAccumulation::PerRow)>;
    using ConvertCopy = CopyTile<input(
        cb_matmul_partials,
        WaitPolicy::None,
        PopPolicy::None,
        InputTileMapping::Block,
        DataFormatReconfig::Disabled,
        TileAddressing::Offset)>;
    using ConvertPack =
        PackTile<output(cb_out_tiles, ReservePolicy::Upfront, PushPolicy::AtEnd, DataFormatReconfig::Disabled)>;
    using ConvertPackNoc0 =
        PackTile<output(cb_out_tiles_noc0, ReservePolicy::Upfront, PushPolicy::AtEnd, DataFormatReconfig::Disabled)>;

    auto baseline_row = [&](uint32_t row, auto rotary_pack, auto convert_pack) {
        const uint32_t row_base = row * n_tiles;
        if constexpr (is_rotary) {
            const uint32_t table_base = (row % seq_tiles) * table_tiles_per_row;
            for (uint32_t head = 0; head < heads_per_core; ++head) {
                eltwise_chain<InitReconfigOwner::Caller>(
                    IterationShape::grid(tiles_per_head, 2),
                    RotaryMulAcc{partials_dfb, table_dfb, row_base + head * tiles_per_head, table_base},
                    rotary_pack);
            }
        } else {
            eltwise_chain<InitReconfigOwner::Caller>(
                IterationShape::tiles(n_tiles), ConvertCopy{partials_dfb, row_base}, convert_pack);
        }
    };

    // out_lo = P0 * cos_lo + P1 * (-sin_lo), out_hi = P0 * sin_hi + P1 * cos_hi, window_heads heads per
    // DEST window, accumulated in place (ELWMUL accumulates; the released half is zero).
    auto mul = [&](uint32_t a, uint32_t b, uint32_t dst) {
        if constexpr (variant == 4 || variant == 5) {
            tile_src::tile_src_mul<variant == 5>(cb_matmul_partials, cb_rotary_table, a, b, dst);
        } else {
            mul_tiles(cb_matmul_partials, cb_rotary_table, a, b, dst);
        }
    };
    auto window_row = [&](uint32_t row, uint32_t cb_out) {
        const uint32_t row_base = row * n_tiles;
        const uint32_t table_base = (row % seq_tiles) * table_tiles_per_row;
        cb_reserve_back(cb_out, n_tiles);
        for (uint32_t head0 = 0; head0 < heads_per_core; head0 += window_heads) {
            const uint32_t heads = heads_per_core - head0 < window_heads ? heads_per_core - head0 : window_heads;
            tile_regs_acquire();
            for (uint32_t h = 0; h < heads; ++h) {
                const uint32_t p0 = row_base + (head0 + h) * tiles_per_head;
                const uint32_t dst = h * tiles_per_head;
                mul(p0, table_base + 0, dst);
                if constexpr (diag != 2) {
                    mul(p0 + 1, table_base + 1, dst);
                }
                mul(p0, table_base + 2, dst + 1);
                if constexpr (diag != 2) {
                    mul(p0 + 1, table_base + 3, dst + 1);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t t = 0; t < (diag == 1 ? 1 : heads * tiles_per_head); ++t) {
                pack_tile(t, cb_out);
            }
            tile_regs_release();
        }
        cb_push_back(cb_out, n_tiles);
    };

    cb_wait_front(cb_matmul_partials, partials_tiles);
    if constexpr (is_rotary) {
        cb_wait_front(cb_rotary_table, (block_m < seq_tiles ? block_m : seq_tiles) * table_tiles_per_row);
    }

    for (uint32_t iter = 0; iter < iters; ++iter) {
        // Per-block setup, as in the op.
        if constexpr (is_rotary) {
            reconfig_data_format(cb_matmul_partials, cb_rotary_table);
            mul_init(cb_matmul_partials, cb_rotary_table, true);
            if constexpr (variant == 4 || variant == 5) {
                tile_src::tile_src_mul_init<variant == 5>();
            }
        } else {
            reconfig_data_format_srca(cb_matmul_partials);
            copy_init(cb_matmul_partials);
        }
        pack_reconfig_data_format(cb_out_tiles);

        for (uint32_t row = 0; row < block_m; ++row) {
            const bool odd = row & 1;
            if constexpr (variant == 0 || variant == 3) {
                if (odd) {
                    DataflowBuffer out_noc0_dfb(cb_out_tiles_noc0);
                    baseline_row(row, RotaryPackNoc0{out_noc0_dfb}, ConvertPackNoc0{out_noc0_dfb});
                } else {
                    baseline_row(row, RotaryPack{out_dfb}, ConvertPack{out_dfb});
                }
            } else {
                window_row(row, odd ? cb_out_tiles_noc0 : cb_out_tiles);
            }
        }
    }
}
