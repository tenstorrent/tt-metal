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
// [w_tile_start, w_tile_start + vwt). TILE gamma only on this path (RM gamma +
// sharded is refused op-side for now).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

namespace {
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_gamma = 3;
}  // namespace

void kernel_main() {
    constexpr bool HAS_GAMMA = get_compile_time_arg_val(0) != 0;
    constexpr uint32_t inv_N_bits = get_compile_time_arg_val(1);
    constexpr bool HAS_PARTIAL_W = get_compile_time_arg_val(2) != 0;
    constexpr uint32_t partial_w = get_compile_time_arg_val(3);
    constexpr uint32_t gamma_page = get_compile_time_arg_val(4);

    constexpr auto gamma_args = TensorAccessorArgs<5>();

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
