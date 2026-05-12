// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// multigammaln_lanczos — Compute kernel.
//
// Per input tile:
//   1) Init cb_accumulator to zero.
//   2) For k = 0..3: run one Lanczos lgamma evaluation against (a − 0.5·k),
//      read the running accumulator from cb_accumulator front, add the new
//      lgamma value, pack to cb_accumulator back. The input tile is read four
//      times via copy_tile (no pop between iterations).
//   3) Finalize: D0 = accumulator + 3·log(π); pack to cb_output_tiles.
//   4) Drain: pop accumulator, push output, pop input.
//
// DST layout (4 slots — fp32_dest_acc + half-sync):
//   D0 — local lgamma accumulator / final result before pack
//   D1 — a = (input + offset[k]), unchanged within an lgamma iteration
//   D2 — scratch (a+i, log(a+4.5), pole mask, accumulator reload)
//   D3 — scratch (a − 0.5)
//
// CT args: [cb_input_tiles, cb_output_tiles, cb_accumulator]
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

// Compile-time bit-cast (gcc/clang constexpr intrinsic — also used in tt-metal
// reduction kernels for packing fp32 scalars into uint32 SFPU params).
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

// 4.5 − 0.918938531357171 — the algebraic re-grouping constant baked at
// compile time so the result formula is
//   D0 = (a − 0.5)·log(a + 4.5) + log(temp) − a − LANCZOS_OFFSET
constexpr float LANCZOS_OFFSET = 3.581061468642829f;

// 3·log(π) — the constant added once at the end of the 4-lgamma sum.
constexpr float THREE_LOG_PI = 3.434189657547f;

}  // namespace

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t cb_accumulator = get_compile_time_arg_val(2);

    // Initial hw setup. Unpacker bound to cb_input_tiles, packer bound to
    // cb_accumulator: phases 1 and 2.k both pack into cb_accumulator, so we
    // keep the packer there by default and only reconfig to cb_output_tiles
    // in phase 3 (then reconfig back at the end of the iteration).
    init_sfpu(cb_input_tiles, cb_accumulator);

    for (uint32_t t = 0; t < num_tiles; ++t) {
        // Wait for the input tile. We do NOT pop until after all 4 lgamma
        // iterations have read it.
        cb_wait_front(cb_input_tiles, 1);

        // ================================================================
        // Phase 1: init accumulator tile to zero.
        // ================================================================
        cb_reserve_back(cb_accumulator, 1);
        tile_regs_acquire();
        fill_tile_init();
        fill_tile(0, 0.0f);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_accumulator);
        tile_regs_release();
        cb_push_back(cb_accumulator, 1);

        // ================================================================
        // Phase 2.k: 4 Lanczos lgamma evaluations.
        // ================================================================
        for (uint32_t k = 0; k < 4; ++k) {
            cb_wait_front(cb_accumulator, 1);
            cb_reserve_back(cb_accumulator, 1);

            tile_regs_acquire();

            // 2.k.0: D1 = (input tile) + offsets[k] = a.
            copy_tile_to_dst_init_short(cb_input_tiles);
            copy_tile(cb_input_tiles, 0, 1);
            binop_with_scalar_tile_init();
            add_unary_tile(1, f2u(LGAMMA_OFFSETS[k]));

            // 2.k.1: D0 = 0 — local lgamma accumulator.
            fill_tile_init();
            fill_tile(0, 0.0f);

            // 2.k.2: Lanczos polynomial — for i = 0..5,
            //        D0 += LANCZOS_COEFFS[i] / (a + i).
            for (uint32_t i = 0; i < 6; ++i) {
                copy_dest_values_init();
                copy_dest_values<DataFormat::Float32>(1, 2);  // D2 = a
                binop_with_scalar_tile_init();
                add_unary_tile(2, f2u(static_cast<float>(i)));  // D2 = a + i
                recip_tile_init();
                recip_tile(2);  // D2 = 1 / (a + i)
                binop_with_scalar_tile_init();
                mul_unary_tile(2, f2u(LANCZOS_COEFFS[i]));  // D2 = C[i] / (a + i)
                add_binary_tile_init();
                add_binary_tile(0, 2, 0);  // D0 += D2
            }

            // 2.k.3: D0 = log(1 + D0) = log(temp).
            binop_with_scalar_tile_init();
            add_unary_tile(0, f2u(1.0f));
            log_tile_init();
            log_tile(0);

            // 2.k.4: D0 -= (a + LANCZOS_OFFSET).
            copy_dest_values_init();
            copy_dest_values<DataFormat::Float32>(1, 2);  // D2 = a
            binop_with_scalar_tile_init();
            add_unary_tile(2, f2u(LANCZOS_OFFSET));  // D2 = a + 3.5810...
            sub_binary_tile_init();
            sub_binary_tile(0, 2, 0);  // D0 -= D2

            // 2.k.5: D0 += (a − 0.5) · log(a + 4.5).
            copy_dest_values_init();
            copy_dest_values<DataFormat::Float32>(1, 2);  // D2 = a
            binop_with_scalar_tile_init();
            add_unary_tile(2, f2u(4.5f));  // D2 = a + 4.5
            log_tile_init();
            log_tile(2);  // D2 = log(a + 4.5)
            copy_dest_values_init();
            copy_dest_values<DataFormat::Float32>(1, 3);  // D3 = a
            binop_with_scalar_tile_init();
            sub_unary_tile(3, f2u(0.5f));  // D3 = a − 0.5
            mul_binary_tile_init();
            mul_binary_tile(2, 3, 2);  // D2 = (a−0.5)·log(a+4.5)
            add_binary_tile_init();
            add_binary_tile(0, 2, 0);  // D0 += D2

            // 2.k.6: pole zeroing — D0 *= (a != 1.0); D0 *= (a != 2.0).
            // Mask is applied to a (always finite within the test domain), so
            // NaN/Inf in D0 from a polynomial pole would NOT contaminate the
            // multiply (mask cleanly evaluates to {0.0, 1.0}).
            copy_dest_values_init();
            copy_dest_values<DataFormat::Float32>(1, 2);  // D2 = a
            unary_ne_tile_init();
            unary_ne_tile(2, f2u(1.0f));  // D2 = (a != 1.0)
            mul_binary_tile_init();
            mul_binary_tile(0, 2, 0);  // D0 *= D2

            copy_dest_values_init();
            copy_dest_values<DataFormat::Float32>(1, 2);  // D2 = a
            unary_ne_tile_init();
            unary_ne_tile(2, f2u(2.0f));  // D2 = (a != 2.0)
            mul_binary_tile_init();
            mul_binary_tile(0, 2, 0);  // D0 *= D2

            // 2.k.7: D0 += previous global accumulator (front of cb_accumulator).
            copy_tile_to_dst_init_short(cb_accumulator);
            copy_tile(cb_accumulator, 0, 2);  // D2 = old accumulator
            add_binary_tile_init();
            add_binary_tile(0, 2, 0);  // D0 += D2

            // 2.k.8: pack the new accumulator; cycle the CB by popping the
            // old front and pushing the new back.
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_accumulator);
            tile_regs_release();
            cb_pop_front(cb_accumulator, 1);
            cb_push_back(cb_accumulator, 1);
        }

        // ================================================================
        // Phase 3: finalize — D0 = accumulator + 3·log(π); pack to output.
        // ================================================================
        cb_wait_front(cb_accumulator, 1);
        cb_reserve_back(cb_output_tiles, 1);

        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_accumulator);
        copy_tile(cb_accumulator, 0, 0);
        binop_with_scalar_tile_init();
        add_unary_tile(0, f2u(THREE_LOG_PI));
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_output_tiles);
        pack_tile(0, cb_output_tiles);
        tile_regs_release();

        // ================================================================
        // Phase 4: drain CBs for the next iteration.
        // ================================================================
        cb_pop_front(cb_accumulator, 1);
        cb_push_back(cb_output_tiles, 1);
        cb_pop_front(cb_input_tiles, 1);

        // Re-bind the packer back to cb_accumulator so the next iteration's
        // phase 1 packs to the right CB. (Both are fp32, so this is mostly a
        // CB index reconfig — but the packer's bound CB id changes.)
        pack_reconfig_data_format(cb_accumulator);
    }
}
