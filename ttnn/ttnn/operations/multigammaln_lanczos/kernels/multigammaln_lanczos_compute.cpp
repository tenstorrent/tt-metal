// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// multigammaln_lanczos — Compute kernel (Refinement 1: DST-resident accumulator).
//
// Per input tile, a SINGLE tile_regs_acquire spans all 4 Lanczos lgamma
// evaluations. The global accumulator lives in D0 across iterations — no
// cb_accumulator round-trip is needed (and the CB no longer exists).
//
// DST layout (4 slots — fp32_dest_acc + half-sync):
//   D0 — global accumulator (persistent across all 4 lgamma iterations)
//   D1 — a = (input + offset[k]) for the current iteration
//   D2 — local lgamma running sum / partial result
//   D3 — scratch (a + i, log(a + 4.5), pole mask, ...)
//
// Slot budget note: the step `D2 += (a - 0.5) * log(a + 4.5)` needs both
// (a − 0.5) and log(a + 4.5) live simultaneously with D0 (global accumulator)
// and D2 (local sum). That is 5 live values — one more than the half-sync
// budget. We resolve this by COMPILING D1 INTO (a - 0.5) for that single
// multiply (corrupting D1) and immediately reloading D1 = a from
// cb_input_tiles before pole zeroing. cb_input_tiles is held across the
// entire 4-iteration block (popped only after phase 4), so the reload is a
// cheap L1 read from the same already-resident tile.
//
// CT args: [cb_input_tiles, cb_output_tiles]
// RT args: [num_tiles_this_core]

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/copy_dest_values.h"

namespace {

// Compile-time bit-cast for packing fp32 constants into the uint32 SFPU param
// the binop_with_scalar / unary_ne APIs accept.
constexpr uint32_t f2u(float f) { return __builtin_bit_cast(uint32_t, f); }

// Lanczos 6-term coefficients (literal values from op_design.md).
constexpr float LANCZOS_COEFFS[6] = {
    76.18009172947146f,
    -86.50532032941677f,
    24.01409824083091f,
    -1.231739572450155f,
    0.1208650973866179e-2f,
    -0.5395239384953e-5f,
};

// Per-lgamma offset for the 4 multivariate sub-evaluations (k = 0..3 →
// lgamma(a), lgamma(a − 0.5), lgamma(a − 1.0), lgamma(a − 1.5)).
constexpr float LGAMMA_OFFSETS[4] = {0.0f, -0.5f, -1.0f, -1.5f};

// 4.5 − 0.918938531357171 — algebraic re-grouping constant.
//   result = (a − 0.5)·log(a + 4.5) + log(temp) − a − LANCZOS_OFFSET
constexpr float LANCZOS_OFFSET = 3.581061468642829f;

// 3·log(π) — the constant added once at the end of the 4-lgamma sum.
constexpr float THREE_LOG_PI = 3.434189657547f;

}  // namespace

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);

    // Unpacker bound to cb_input_tiles, packer bound to cb_output_tiles.
    // Since the global accumulator never leaves DST, the packer never has to
    // switch CBs (no pack_reconfig_data_format calls needed).
    init_sfpu(cb_input_tiles, cb_output_tiles);

    for (uint32_t t = 0; t < num_tiles; ++t) {
        cb_wait_front(cb_input_tiles, 1);
        cb_reserve_back(cb_output_tiles, 1);

        // =============================================================
        // ONE tile_regs_acquire spans all 4 lgamma iterations + final.
        // D0 holds the global accumulator across the entire block.
        // =============================================================
        tile_regs_acquire();

        // D0 = 0 — global accumulator (persistent for all 4 iterations).
        fill_tile_init();
        fill_tile(0, 0.0f);

        for (uint32_t k = 0; k < 4; ++k) {
            // -------- Load D1 = a = (input + offset[k]) --------
            copy_tile_to_dst_init_short(cb_input_tiles);
            copy_tile(cb_input_tiles, 0, 1);
            binop_with_scalar_tile_init();
            add_unary_tile(1, f2u(LGAMMA_OFFSETS[k]));

            // -------- D2 = 0 (local lgamma accumulator) --------
            fill_tile_init();
            fill_tile(2, 0.0f);

            // -------- Lanczos polynomial: D2 += C[i] / (a + i) --------
            for (uint32_t i = 0; i < 6; ++i) {
                copy_dest_values_init();
                copy_dest_values<DataFormat::Float32>(1, 3);  // D3 = a
                binop_with_scalar_tile_init();
                add_unary_tile(3, f2u(static_cast<float>(i)));  // D3 = a + i
                recip_tile_init();
                recip_tile(3);  // D3 = 1 / (a + i)
                binop_with_scalar_tile_init();
                mul_unary_tile(3, f2u(LANCZOS_COEFFS[i]));  // D3 = C[i] / (a + i)
                add_binary_tile_init();
                add_binary_tile(2, 3, 2);  // D2 += D3
            }

            // -------- D2 = log(1 + D2) = log(temp) --------
            binop_with_scalar_tile_init();
            add_unary_tile(2, f2u(1.0f));
            log_tile_init();
            log_tile(2);

            // -------- D2 -= (a + LANCZOS_OFFSET) --------
            copy_dest_values_init();
            copy_dest_values<DataFormat::Float32>(1, 3);  // D3 = a
            binop_with_scalar_tile_init();
            add_unary_tile(3, f2u(LANCZOS_OFFSET));  // D3 = a + 3.581…
            sub_binary_tile_init();
            sub_binary_tile(2, 3, 2);  // D2 -= D3

            // -------- D2 += (a − 0.5) · log(a + 4.5) --------
            // Compute log(a + 4.5) in D3, then CORRUPT D1 := a − 0.5 so we
            // can do the multiply with two DST slots (D1, D3). After the add,
            // we reload D1 = a from cb_input_tiles for pole zeroing.
            copy_dest_values_init();
            copy_dest_values<DataFormat::Float32>(1, 3);  // D3 = a
            binop_with_scalar_tile_init();
            add_unary_tile(3, f2u(4.5f));  // D3 = a + 4.5
            log_tile_init();
            log_tile(3);  // D3 = log(a + 4.5)
            binop_with_scalar_tile_init();
            sub_unary_tile(1, f2u(0.5f));  // D1 = a − 0.5 (CORRUPT)
            mul_binary_tile_init();
            mul_binary_tile(3, 1, 3);  // D3 = (a − 0.5) · log(a + 4.5)
            add_binary_tile_init();
            add_binary_tile(2, 3, 2);  // D2 += D3

            // -------- Reload D1 = a for pole zeroing --------
            copy_tile_to_dst_init_short(cb_input_tiles);
            copy_tile(cb_input_tiles, 0, 1);
            binop_with_scalar_tile_init();
            add_unary_tile(1, f2u(LGAMMA_OFFSETS[k]));  // D1 = a

            // -------- Pole zeroing: D2 *= (a != 1) * (a != 2) --------
            copy_dest_values_init();
            copy_dest_values<DataFormat::Float32>(1, 3);  // D3 = a
            unary_ne_tile_init();
            unary_ne_tile(3, f2u(1.0f));  // D3 = (a != 1.0)
            mul_binary_tile_init();
            mul_binary_tile(2, 3, 2);  // D2 *= D3

            copy_dest_values_init();
            copy_dest_values<DataFormat::Float32>(1, 3);  // D3 = a
            unary_ne_tile_init();
            unary_ne_tile(3, f2u(2.0f));  // D3 = (a != 2.0)
            mul_binary_tile_init();
            mul_binary_tile(2, 3, 2);  // D2 *= D3

            // -------- Accumulate: D0 += D2 --------
            add_binary_tile_init();
            add_binary_tile(0, 2, 0);
        }

        // -------- Final: D0 += 3·log(π) --------
        binop_with_scalar_tile_init();
        add_unary_tile(0, f2u(THREE_LOG_PI));

        // -------- Pack the single output tile --------
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_output_tiles);
        tile_regs_release();

        cb_push_back(cb_output_tiles, 1);
        cb_pop_front(cb_input_tiles, 1);
    }
}
