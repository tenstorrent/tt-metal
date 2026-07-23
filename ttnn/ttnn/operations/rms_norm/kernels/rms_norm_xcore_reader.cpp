// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Cross-core W-split reader for rms_norm (op_design.md §5).
//
// The input W-slice is already resident in L1 (zero-copy sharded cb_x_in), so the
// reader does NOT read x. It only:
//   * prepares the 1/W reduce scaler (full tile + partial tile when W is not
//     tile-aligned; the compute-side partial-holder routes the partial tile), and
//   * reads this core's gamma W-slice (vwt tiles) ONCE into cb_gamma (held,
//     reused across every tile-row) — the same batched tiled-gamma read as the
//     interleaved TILE-gamma path (Refinement 2), offset to the core's W-slice.
//
// gamma is (1,1,1,W) -> Wt tiles in one tile-row; this core reads tiles
// [w_tile_start, w_tile_start + vwt).
//   TILE gamma : read whole tiles straight into cb_gamma (compute skips tilize).
//   RM gamma   : read the W-slice as row-major sticks into cb_gamma_sticks (one
//                tile-wide page per read) so compute tilizes vwt tiles into
//                cb_gamma — the cross-core mirror of the interleaved RM-gamma
//                knob-turn (Refinement 2/4a). gamma stays interleaved in DRAM.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

namespace {
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_gamma = 3;
constexpr uint32_t cb_gamma_sticks = 4;
constexpr uint32_t TILE_DIM = 32;
}  // namespace

void kernel_main() {
    constexpr bool HAS_GAMMA = get_compile_time_arg_val(0) != 0;
    constexpr uint32_t inv_N_bits = get_compile_time_arg_val(1);
    constexpr bool HAS_PARTIAL_W = get_compile_time_arg_val(2) != 0;
    constexpr uint32_t partial_w = get_compile_time_arg_val(3);
    constexpr uint32_t gamma_page = get_compile_time_arg_val(4);
    constexpr bool GAMMA_IS_RM = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t gamma_elem = get_compile_time_arg_val(6);
    constexpr uint32_t origin_W = get_compile_time_arg_val(7);

    constexpr auto gamma_args = TensorAccessorArgs<8>();

    const uint32_t gamma_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_tile_start = get_arg_val<uint32_t>(1);
    const uint32_t vwt = get_arg_val<uint32_t>(2);

    // scaler = 1/origin_W so the SUM-reduce over the LOCAL slice yields the slice's
    // (1/W)-scaled partial; summing the K partials across cores gives mean(x^2).
    const float scaler_f = __builtin_bit_cast(float, inv_N_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
        scaler_f);  // tile 0: full scaler
    if constexpr (HAS_PARTIAL_W) {
        dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            scaler_f, partial_w);  // tile 1: partial scaler (zeros padded lanes)
    }

    if constexpr (HAS_GAMMA) {
        const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr, gamma_page);
        if constexpr (GAMMA_IS_RM) {
            // gamma (1,1,1,W) is one row -> page 0. Read this core's W-slice one
            // tile-column at a time (row_bytes <= one tile width) so each read
            // pushes exactly ONE tile-wide page; compute tilizes vwt of them.
            // Columns beyond origin_W are gamma tensor-padding (output discarded),
            // so clamp cols but still emit vwt pages. Matches the interleaved
            // RM-gamma read (base page 0, per-tile column byte offset).
            for (uint32_t wt = 0; wt < vwt; ++wt) {
                const uint32_t col0 = (w_tile_start + wt) * TILE_DIM;
                uint32_t cols = (col0 < origin_W) ? (origin_W - col0) : 0;
                if (cols > TILE_DIM) {
                    cols = TILE_DIM;
                }
                dataflow_kernel_lib::read_sticks_for_tilize<cb_gamma_sticks>(
                    gamma_accessor, /*total_num_rows=*/1, cols * gamma_elem, /*start_page=*/0, col0 * gamma_elem);
            }
        } else {
            const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma);
            cb_reserve_back(cb_gamma, vwt);
            uint32_t gl1 = get_write_ptr(cb_gamma);
            for (uint32_t wt = 0; wt < vwt; ++wt) {
                noc_async_read_tile(w_tile_start + wt, gamma_accessor, gl1);
                gl1 += gamma_tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_gamma, vwt);
        }
    }
}
