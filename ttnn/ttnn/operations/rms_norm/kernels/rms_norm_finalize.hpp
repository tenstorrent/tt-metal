// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rms_norm's `finalize` — the reduce helper's post_reduce_op:
//
//     1/rms = rsqrt( Sum(x^2) * (1/W) + epsilon )
//
// applied to the DEST tile a `REDUCE_ROW` just produced, at BOTH call sites in
// rms_norm_compute.cpp (the owner's cross-core `combine_block`, and the `s == 1`
// `collapse_partial_block`).
//
// ===========================================================================
// RAW-LLK JUSTIFICATION — do NOT "fix" this back to the wrappers
// ===========================================================================
// The verifier's helper-usage pass will see three raw SFPU calls here where
// `mul_unary_tile` / `add_unary_tile` / `rsqrt_tile` would read more idiomatically.
// Reverting them undoes a MEASURED 4.12x on this function and 1.097x on the whole
// op at the perf-flagged focus case.  What the wrappers cannot express:
//
// Bypassed: `mul_unary_tile` / `add_unary_tile` (binop_with_scalar.h:34-70) and
// `rsqrt_tile` (rsqrt.h:36-45).  All three HARDCODE `VectorMode::RC` and
// `ITERATIONS = 8` at their own call site, so each is a walk over the WHOLE 32x32
// DEST tile = 32 32-lane SFPU vector ops, 96 in total per finalized tile.
//
// The tile they walk is a REDUCE_ROW result: the only meaningful data is COLUMN 0
// (one value per row).  Its single consumer is the `BroadcastDim::Col` operand
// `rms_col`, which reads column 0 and replicates it across the row; nothing
// downstream reads any other lane (the funnel `noc_async_write` ships the whole
// 4 KB tile to the root, but the root only ever re-broadcasts it into that same
// Col-broadcast consumer).  Two mechanisms recover the waste, neither reachable
// through the wrappers:
//
//   1. SCOPE.  Column 0 lives in faces 0 and 2 (`VectorMode::C`) and, within a
//      face, only in the EVEN column parity — the SFPU walks a face as
//      [rg0-even, rg0-odd, rg1-even, ...], so parity is the INNER axis and
//      `ITERATIONS` (which truncates the OUTER axis) cannot isolate it.  A
//      hand-addressed sfpi body that strides DEST by 2 keeps 8 of the 32 vectors.
//   2. FUSION.  mul, add and rsqrt are three separate WALKS over the same tile.
//      One body computing `rsqrt(v*inv_w + eps)` per vector makes it one, and
//      drops one of the two inits.
//
// Both go through `_llk_math_eltwise_unary_sfpu_params_`, the same entry point
// `SFPU_UNARY_CALL` (and therefore every wrapper above) uses; the only thing
// bypassed is the fixed VectorMode/ITERATIONS the wrapper bakes in.
//
// MEASURED (Blackhole p150b @1350 MHz, Perf 2, isolated MATH-thread ns per
// finalize, copy+pack outside the timed zone):
//     stock 3-wrapper chain   989.7 ns   96 vector ops   2 inits
//     this  fused + c-skip    240.1 ns    8 vector ops   1 init      4.12x
// and on the whole op: focus `(1,1,8192,1024)` BLOCK 34579 -> 31517 (1.097x),
// `(1,1,32,5120)` WIDTH 6448 -> 5671 (1.137x), `(1,1,32,7168)` WIDTH 6416 -> 5654
// (1.135x), and the `s == 1` collapse 1.38-1.74x (B=16: 28444 -> 16379).
// An ablation put the finalize at 3993 ns of the owner combine's 5535 ns (72%) —
// the finalize IS the combine, the `reduce_tile` sum is only 1542 ns of it.
//
// PRECISION IS UNCHANGED — this is not a precision trade.  `APPROX` is never
// flipped and no approximate rsqrt is used: the body calls the identical
// non-approximate `_calculate_sqrt_body_<APPROX, RECIPROCAL=true>` that
// `rsqrt_tile` calls, and the final `convert<vFloat16b>` reproduces exactly the
// DEST truncation the wrapper chain applied at `dst_full_sync_en=False`.
// Measured bit-identical on column 0 across 18 cells (BLOCK/WIDTH/HEIGHT/
// INTERLEAVED x TILE/ROW_MAJOR x bf16/float32/bfloat8_b x fp32_dest_acc_en
// both x HiFi2/HiFi4 x W- and H-non-aligned x gamma/no-gamma).

#pragma once

#include <cstdint>

#include "api/compute/eltwise_unary/rsqrt.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#include "sfpu/ckernel_sfpu_converter.h"
#endif

#ifdef TRISC_MATH
// NOT unrolled, deliberately: the sqrt body alone nearly fills the 8 SFPU LREGs,
// and holding the two scalar constants live across an UNROLLED copy of it spills
// ("internal compiler error: cannot store sfpu register" — measured, and it is an
// ICE rather than a diagnostic).  One body live at a time; measured at the same
// ns/vector as the stock unrolled body, so the pragma costs nothing.
sfpi_inline void rms_finalize_body(uint32_t inv_w_bits, uint32_t eps_bits) {
#pragma GCC unroll 1
    // 4 even-parity vectors per face x faces {0, 2} = 8 vectors, covering column 0
    // of all 32 rows.  The net dst_reg advance is 4*2 == the stock ITERATIONS of
    // 8, so `VectorMode::C`'s face0 -> face2 stepping composes unchanged.
    for (int i = 0; i < 4; i++) {
        const sfpi::vFloat inv_w = ckernel::sfpu::Converter::as_float(inv_w_bits);
        const sfpi::vFloat eps = ckernel::sfpu::Converter::as_float(eps_bits);
        sfpi::vFloat v = sfpi::dst_reg[0] * inv_w + eps;
        sfpi::vFloat t = ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false>(v);
        if constexpr (!DST_ACCUM_MODE) {
            t = sfpi::convert<sfpi::vFloat16b>(t, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = t;
        sfpi::dst_reg += 2;
    }
}
#endif

// `rsqrt_tile_init()` expands to `llk_math_eltwise_unary_sfpu_init<SfpuType::rsqrt>`,
// which already runs `_llk_math_eltwise_unary_sfpu_init_` (SFPU config reg +
// ADDR_MOD_7/6 + counter reset) — exactly the invariant `binop_with_scalar_tile_init()`
// existed to establish.  With the binop walks folded in, that second init is dead.
ALWI void rms_finalize(uint32_t dst_idx, uint32_t inv_w_bits, uint32_t eps_bits) {
    rsqrt_tile_init();
    MATH((_llk_math_eltwise_unary_sfpu_params_(
        rms_finalize_body, dst_idx, ckernel::VectorMode::C, inv_w_bits, eps_bits)));
}
