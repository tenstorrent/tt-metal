// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// LayerNorm - Compute Kernel (Stage 1 Stub: data_pipeline)
//
// Stage 1 implements a passthrough:
//   cb_in (RM) -> tilize -> cb_tilize_out -> copy -> cb_out -> untilize -> cb_untilize_out
//
// Full layer_norm compute (stages 2-4) will be added incrementally.
//
// Compile-time args:
//   [0]  Ht         -- tile-rows to process (N / 32)
//   [1]  Wt         -- tiles per row (W / 32)
//   [2]  has_weight -- 1 if gamma provided
//   [3]  has_bias   -- 1 if beta provided

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

// CB indices matching op_design.md
constexpr uint32_t cb_in = 0;             // RM input pages (reader -> compute)
constexpr uint32_t cb_eps = 1;            // epsilon tile
constexpr uint32_t cb_scaler = 2;         // reduce scaler
constexpr uint32_t cb_weight = 3;         // gamma tiles
constexpr uint32_t cb_bias = 4;           // beta tiles
constexpr uint32_t cb_tilize_out = 16;    // tilize output
constexpr uint32_t cb_out = 17;           // final output (pre-untilize)
constexpr uint32_t cb_untilize_out = 18;  // untilized output (RM, for writer)
constexpr uint32_t cb_mean = 24;          // row mean
constexpr uint32_t cb_x_minus_mean = 25;  // x - mean
constexpr uint32_t cb_sq = 26;            // (x - mean)^2
constexpr uint32_t cb_var = 27;           // variance
constexpr uint32_t cb_inv_std = 28;       // 1/sqrt(var + eps)
constexpr uint32_t cb_norm = 29;          // normalized output

void kernel_main() {
    // ---- Compile-time args ----
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t has_weight = get_compile_time_arg_val(2);
    constexpr uint32_t has_bias = get_compile_time_arg_val(3);

    // Hardware startup: initialize unpack/math/pack pipelines.
    // Args: input_a=cb_in, input_b=cb_scaler, output=cb_out
    // The CB IDs must match those used by the first operation after startup.
    // tilize helper will reconfigure internally as needed.
    compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);

    // ---- Stage 1: data_pipeline passthrough ----
    // For each tile-row:
    //   Phase 0: Tilize cb_in (RM) -> cb_tilize_out (tiled)
    //   Phase T: Copy   cb_tilize_out -> cb_out    (tile-by-tile passthrough)
    //   Phase 8: Untilize cb_out -> cb_untilize_out (RM sticks for writer)

    for (uint32_t tile_row = 0; tile_row < Ht; ++tile_row) {
        // Phase 0: Tilize — convert Wt RM pages from cb_in into Wt tiles in cb_tilize_out.
        // tilize<input_cb, output_cb>(num_tiles_per_block, num_blocks)
        compute_kernel_lib::tilize<cb_in, cb_tilize_out>(Wt, 1);

        // Phase T: Copy tiles from cb_tilize_out to cb_out (passthrough for Stage 1).
        // In later stages this will be replaced by the full normalization pipeline.
        cb_wait_front(cb_tilize_out, Wt);
        cb_reserve_back(cb_out, Wt);

        copy_tile_to_dst_init_short(cb_tilize_out);
        for (uint32_t t = 0; t < Wt; ++t) {
            acquire_dst();
            copy_tile(cb_tilize_out, t, 0);
            pack_tile(0, cb_out);
            release_dst();
        }

        cb_push_back(cb_out, Wt);
        cb_pop_front(cb_tilize_out, Wt);

        // Phase 8: Untilize cb_out -> cb_untilize_out.
        // WaitUpfront: all Wt tiles have already been pushed to cb_out above.
        compute_kernel_lib::untilize<
            Wt,
            cb_out,
            cb_untilize_out,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::WaitUpfront>(1);
    }
}
