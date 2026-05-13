// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// atan_mean — Compute kernel (fused SFPU atan + REDUCE_ROW AVG).
//
// Per row-tile:
//   1. ``sfpu_atan<cb_input_tiles>(cb_atan_tiles, Wt)``
//        Consumes Wt input tiles, applies SFPU atan, pushes Wt output tiles.
//        The helper owns DEST acquire/commit/wait/release, CB sync, and the
//        data-format reconfig from cb_input_tiles → cb_atan_tiles (both fp32).
//
//   2. ``reduce<AVG, REDUCE_ROW>(cb_atan_tiles, cb_scaler, cb_output_tiles,
//                                ReduceInputBlockShape::row(Wt))``
//        Consumes Wt post-atan tiles, waits on the persistent ``cb_scaler``
//        tile (never popped), emits 1 output tile holding the row mean
//        (col-0 valid region, matmul-mode REDUCE_ROW). The helper handles
//        DEST management, matmul-mode reduce init, and packer reconfig.
//
// ``cb_atan_tiles`` is sized to Wt pages in the program descriptor — both
// helpers own all 3 TRISCs (sequential), so ``sfpu_atan`` pushes all Wt tiles
// before ``reduce<>`` starts consuming.
//
// ``cb_scaler`` is set up once by the reader and never popped — every
// ``reduce<>`` call re-waits on it via ``cb_wait_front``, which is idempotent.
//
// CT args: [CB_INPUT_TILES, CB_SCALER, CB_ATAN_TILES, CB_OUTPUT_TILES, Wt]
// RT args: [num_row_tiles_this_core]

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    const uint32_t num_row_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(1);
    constexpr uint32_t cb_atan_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t Wt = get_compile_time_arg_val(4);

    // Required prologue for all kernel_lib helpers. icb0 anchors the primary
    // input format (fp32 from cb_input_tiles), icb1 the scaler format (bf16),
    // ocb the pack target (fp32). Subsequent helpers reconfig as needed via
    // their INPUT_AND_OUTPUT default reconfig mode.
    compute_kernel_hw_startup(cb_input_tiles, cb_scaler, cb_output_tiles);

    constexpr auto reduce_block_shape = ckl::ReduceInputBlockShape::row(Wt);

    for (uint32_t r = 0; r < num_row_tiles; ++r) {
        // Phase 1: SFPU atan over the Wt input tiles for this row-tile.
        // Reads cb_input_tiles (fp32, Wt tiles) → writes cb_atan_tiles (fp32).
        ckl::sfpu_atan<cb_input_tiles>(cb_atan_tiles, Wt);

        // Phase 2: REDUCE_ROW AVG over the Wt post-atan tiles. With the scaler
        // = 1/W (matmul col-0 fill), the matmul-based REDUCE_ROW path emits
        // the per-row mean directly into column 0 of the output tile.
        ckl::reduce<ckernel::PoolType::AVG, ckernel::ReduceDim::REDUCE_ROW>(
            cb_atan_tiles, cb_scaler, cb_output_tiles, reduce_block_shape);
    }
}
