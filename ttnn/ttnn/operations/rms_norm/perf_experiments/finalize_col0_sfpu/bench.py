# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: rms_norm's combine_block finalize  rstd = rsqrt(sum * inv_w + eps)  on the SFPU.

The op's `finalize_rstd` post_reduce_op runs on a DEST tile whose ONLY meaningful lanes are column 0
(the collapse phase produced column-0-valid tiles; the downstream consumer is a bcast-col multiply).
The current body is three full-tile SFPU passes with two inits per output tile:
    binop_with_scalar_tile_init(); mul_unary_tile(dst, inv_w); add_unary_tile(dst, eps);
    rsqrt_tile_init(); rsqrt_tile(dst);                          -> 3 x 32 = 96 vector ops

Variants (all compute the SAME function at the SAME precision knobs: APPROX / DST_ACCUM_MODE come from the
user's ComputeConfigDescriptor exactly like the op; the rsqrt body is the stock non-approx 23-bit
`_calculate_sqrt_body_<APPROX, RECIPROCAL=true>` that rsqrt_tile uses):
    baseline         the op's lambda verbatim                                     96 vec, 2 inits
    fused_full       one sfpi pass rsqrt(x*inv_w+eps) over the whole tile         32 vec, 1 init
    col0_c           the 3 stock functors under VectorMode::C (left faces)        48 vec, 2 inits
    fused_c          fused pass under VectorMode::C                               16 vec, 1 init
    col0_skip_3pass  the 3 passes as even-parity-stride bodies under C            24 vec, 2 inits
    col0_skip_fused  fused pass, VectorMode::C + even-parity stride (col 0 only)   8 vec, 1 init
Each variant runs with the init INSIDE the per-call body (`pc`, what the op's lambda does) or hoisted once
before the loop (`hoist`).

Single core. DEST[0] gets the finalize once, outside the timed zone, and is packed for the host to check
column 0 vs a float64 reference. DEST[1] gets the finalize `reps` times inside one DeviceZoneScopedN on the
MATH thread (TRISC_1) -- copy/pack are outside the zone, so zone_ns / reps is the per-call SFPU cost.
"""

import struct

import ttnn

TILE = 32
CB_IN = 0
CB_OUT = 16

VARIANTS = ("baseline", "fused_full", "col0_c", "fused_c", "col0_skip_3pass", "col0_skip_fused")
_VARIANT_ID = {v: i for i, v in enumerate(VARIANTS)}
VECTORS = {"baseline": 96, "fused_full": 32, "col0_c": 48, "fused_c": 16, "col0_skip_3pass": 24, "col0_skip_fused": 8}
INITS = {"baseline": 2, "fused_full": 1, "col0_c": 2, "fused_c": 1, "col0_skip_3pass": 2, "col0_skip_fused": 1}
INIT_MODES = ("pc", "hoist")  # per-call init (the op's lambda) / hoisted once before the loop
DEST_MODES = ("dest16", "dest32")  # fp32_dest_acc_en False / True


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def zone_tag(variant: str, init_mode: str, dest_mode: str, approx: bool = False) -> str:
    return f"{variant}_{init_mode}_{dest_mode}" + ("_apx" if approx else "")


def zone_name(variant: str, init_mode: str, dest_mode: str, approx: bool = False) -> str:
    return "FIN_" + zone_tag(variant, init_mode, dest_mode, approx)


# ======================================================================================================
# Compute kernel.  CT args: [variant_id, init_per_call]   RT args: [reps, inv_w_bits, eps_bits]
# Defines: FIN_ZONE_TAG=<token> (zone name = "FIN_" #FIN_ZONE_TAG)
# ======================================================================================================
_KERNEL = r"""
// Isolated bench of rms_norm's finalize post-op. RAW LLK: the fused / column-scoped variants bypass
// mul_unary_tile / add_unary_tile / rsqrt_tile (which hardcode VectorMode::RC + ITERATIONS=8 and three
// separate DEST round trips) and call _llk_math_eltwise_unary_sfpu_params_ with a hand-written sfpi body:
// one DEST load -> SFPMAD(x, inv_w, eps) -> stock _calculate_sqrt_body_<APPROX, RECIPROCAL> -> one store,
// optionally only over the even-parity vectors of the left faces (the only vectors that hold column 0).
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#endif

using ckernel::VectorMode;

#define FIN_STR2(x) #x
#define FIN_STR(x) FIN_STR2(x)
#define FIN_ZONE_LIT "FIN_" FIN_STR(FIN_ZONE_TAG)

constexpr uint32_t V_BASELINE = 0, V_FUSED_FULL = 1, V_COL0_C = 2, V_FUSED_C = 3, V_COL0_SKIP_3PASS = 4,
                   V_COL0_SKIP_FUSED = 5;

#ifdef TRISC_MATH
// Fused finalize body: y = rsqrt(x * inv_w + eps). STRIDE=1 walks every vector (ITER=8 per face);
// STRIDE=2/ITER=4 visits only the even-parity vectors (columns 0,2,..,14) -- column 0 for all rows of the
// face -- and nets the same +8 advance, so VectorMode::C's face-0 -> face-2 stepping composes unchanged.
// REMAT: re-materialize the two scalars every vector (2 SFPLOADI each, behind an asm barrier so LICM cannot
// hoist them). Keeping them live as vFloats across the rsqrt body exhausts the SFPU register file under
// VectorMode::C and sfpi cannot spill (ICE "cannot store sfpu register"); the RC walk happens to fit.
// UNROLL: inner-loop unroll factor. Unrolled, the scheduler interleaves iteration d+1's constant loads with
// iteration d's rsqrt body and the column-scoped instantiations run out of SFPU registers (ICE at
// vFloat::vFloat(float)); UNROLL=1 keeps one vector's worth of live values.
template <int STRIDE, int ITER, bool REMAT, int UNROLL>
sfpi_inline void fused_finalize_body(uint32_t inv_w_bits, uint32_t eps_bits) {
    sfpi::vFloat inv_w, eps;
    if constexpr (!REMAT) {
        inv_w = ckernel::sfpu::Converter::as_float(inv_w_bits);
        eps = ckernel::sfpu::Converter::as_float(eps_bits);
    }
#pragma GCC unroll UNROLL
    for (int d = 0; d < ITER; d++) {
        if constexpr (REMAT) {
            asm volatile("" : "+r"(inv_w_bits), "+r"(eps_bits));
            inv_w = ckernel::sfpu::Converter::as_float(inv_w_bits);
            eps = ckernel::sfpu::Converter::as_float(eps_bits);
        }
        sfpi::vFloat x = sfpi::dst_reg[0];
        x = x * inv_w + eps;
        sfpi::vFloat y = ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false /*FAST_APPROX*/>(x);
        if constexpr (!DST_ACCUM_MODE) {
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = y;
        sfpi::dst_reg += STRIDE;
    }
}

// Even-parity-stride versions of the three stock passes (same per-vector math as calculate_binop_with_scalar /
// _calculate_sqrt_internal_, only the odd-parity vectors are skipped).
template <int BINOP_MODE>
sfpi_inline void binop_skip_body(uint32_t param) {
    const sfpi::vFloat p = ckernel::sfpu::Converter::as_float(param);
#pragma GCC unroll 4
    for (int d = 0; d < 4; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        if constexpr (BINOP_MODE == ckernel::MUL_UNARY) {
            sfpi::dst_reg[0] = v * p;
        } else {
            sfpi::dst_reg[0] = v + p;
        }
        sfpi::dst_reg += 2;
    }
}
sfpi_inline void rsqrt_skip_body() {
#pragma GCC unroll 4
    for (int d = 0; d < 4; d++) {
        sfpi::vFloat y = ckernel::sfpu::_calculate_sqrt_body_<APPROX, true, false>(sfpi::dst_reg[0]);
        if constexpr (!DST_ACCUM_MODE) {
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = y;
        sfpi::dst_reg += 2;
    }
}
#endif

template <int STRIDE, int ITER, bool REMAT, int UNROLL>
ALWI void fused_finalize(uint32_t dst, VectorMode vm, uint32_t inv_w_bits, uint32_t eps_bits) {
    MATH((_llk_math_eltwise_unary_sfpu_params_(
        fused_finalize_body<STRIDE, ITER, REMAT, UNROLL>, dst, vm, inv_w_bits, eps_bits)));
}

// Left-column face walk (faces 0 and 2 == what VectorMode::C does) with an OPAQUE trip count. The LLK's
// `_llk_math_eltwise_sfpu_apply_vector_mode_` C-branch is a 2-trip loop the compiler fully unrolls (its
// `#pragma GCC unroll 0` notwithstanding), and two adjacent copies of MAD+rsqrt get interleaved by the
// scheduler until the SFPU register file overflows (sfpi cannot spill -> ICE). A runtime trip count keeps one
// body copy per iteration, exactly like the 4-trip RC walk that compiles.
template <int STRIDE, int ITER, bool REMAT, int UNROLL>
ALWI void fused_finalize_col(uint32_t dst, uint32_t inv_w_bits, uint32_t eps_bits) {
    MATH(({
        uint32_t num_faces;
        asm volatile("li %0, 2" : "=r"(num_faces));  // opaque 2: keeps the face loop a loop
        _llk_math_eltwise_sfpu_start_(dst);
        for (uint32_t face = 0; face < num_faces; ++face) {
            fused_finalize_body<STRIDE, ITER, REMAT, UNROLL>(inv_w_bits, eps_bits);
            _llk_math_eltwise_sfpu_inc_dst_face_addr_();
            _llk_math_eltwise_sfpu_inc_dst_face_addr_();
        }
        _llk_math_eltwise_sfpu_done_();
    }));
}

// ---- init (what the op's lambda runs before its SFPU calls) ----
template <uint32_t VARIANT>
ALWI void finalize_init() {
    if constexpr (VARIANT == V_BASELINE || VARIANT == V_COL0_C || VARIANT == V_COL0_SKIP_3PASS) {
        binop_with_scalar_tile_init();
    }
    rsqrt_tile_init();
}

// ---- body (the per-output-tile SFPU work; init handled separately) ----
template <uint32_t VARIANT>
ALWI void finalize_body(uint32_t dst, uint32_t inv_w_bits, uint32_t eps_bits) {
    if constexpr (VARIANT == V_BASELINE) {
        // NOTE: the op's lambda interleaves the two inits with the passes; with per-call init we replicate
        // that order exactly in finalize_call below.
        mul_unary_tile(dst, inv_w_bits);
        add_unary_tile(dst, eps_bits);
        rsqrt_tile(dst);
    } else if constexpr (VARIANT == V_FUSED_FULL) {
        fused_finalize<1, 8, false, 8>(dst, VectorMode::RC, inv_w_bits, eps_bits);
    } else if constexpr (VARIANT == V_COL0_C) {
        MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_binop_with_scalar,
            (APPROX, MUL_UNARY, 8, DST_ACCUM_MODE), dst, VectorMode::C, inv_w_bits));
        MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_binop_with_scalar,
            (APPROX, ADD_UNARY, 8, DST_ACCUM_MODE), dst, VectorMode::C, eps_bits));
        MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_rsqrt,
            (APPROX, 8, DST_ACCUM_MODE, false, false), dst, VectorMode::C));
    } else if constexpr (VARIANT == V_FUSED_C) {
        fused_finalize_col<1, 8, false, 8>(dst, inv_w_bits, eps_bits);
    } else if constexpr (VARIANT == V_COL0_SKIP_3PASS) {
        MATH((_llk_math_eltwise_unary_sfpu_params_(binop_skip_body<MUL_UNARY>, dst, VectorMode::C, inv_w_bits)));
        MATH((_llk_math_eltwise_unary_sfpu_params_(binop_skip_body<ADD_UNARY>, dst, VectorMode::C, eps_bits)));
        MATH((_llk_math_eltwise_unary_sfpu_params_(rsqrt_skip_body, dst, VectorMode::C)));
    } else {
        fused_finalize_col<2, 4, false, 4>(dst, inv_w_bits, eps_bits);
    }
}

// One finalize call as the op would issue it (init per call, in the op's order).
template <uint32_t VARIANT, bool INIT_PER_CALL>
ALWI void finalize_call(uint32_t dst, uint32_t inv_w_bits, uint32_t eps_bits) {
    if constexpr (VARIANT == V_BASELINE && INIT_PER_CALL) {
        binop_with_scalar_tile_init();
        mul_unary_tile(dst, inv_w_bits);
        add_unary_tile(dst, eps_bits);
        rsqrt_tile_init();
        rsqrt_tile(dst);
    } else {
        if constexpr (INIT_PER_CALL) {
            finalize_init<VARIANT>();
        }
        finalize_body<VARIANT>(dst, inv_w_bits, eps_bits);
    }
}

void kernel_main() {
    constexpr uint32_t cb_in = 0, cb_out = 16;
    constexpr uint32_t VARIANT = get_compile_time_arg_val(0);
    constexpr bool INIT_PER_CALL = get_compile_time_arg_val(1) != 0;
    const uint32_t reps = get_arg_val<uint32_t>(0);
    const uint32_t inv_w_bits = get_arg_val<uint32_t>(1);
    const uint32_t eps_bits = get_arg_val<uint32_t>(2);

    compute_kernel_hw_startup(cb_in, cb_in, cb_out);
    copy_tile_init(cb_in);

    cb_reserve_back(cb_in, 1);
    cb_push_back(cb_in, 1);  // sharded input already resident
    cb_wait_front(cb_in, 1);

    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);  // DEST[0]: correctness copy
    copy_tile(cb_in, 0, 1);  // DEST[1]: timing copy
    if constexpr (!INIT_PER_CALL) {
        finalize_init<VARIANT>();
    }
    {
        DeviceZoneScopedN(FIN_ZONE_LIT);  // read TRISC_1 (math): pure SFPU cost of `reps` finalize calls
        for (uint32_t r = 0; r < reps; ++r) {
            finalize_call<VARIANT, INIT_PER_CALL>(1, inv_w_bits, eps_bits);
        }
    }
    finalize_call<VARIANT, INIT_PER_CALL>(0, inv_w_bits, eps_bits);  // once, for the host check
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out, 0);
    cb_push_back(cb_out, 1);
    tile_regs_release();
    cb_pop_front(cb_in, 1);
}
"""


def _single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def sharded_memory_config():
    return ttnn.create_sharded_memory_config(
        shape=(TILE, TILE),
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def compute_config(dest_mode: str, approx: bool = False) -> ttnn.ComputeConfigDescriptor:
    # The op's perf-flagged config (HiFi2, approx off) by default; the DEST width and (for the domain sweep)
    # approx mode vary, and baseline + candidate always share the same values. Fidelity is an FPU knob and
    # does not reach the SFPU, so it is held at the focus value.
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=(dest_mode == "dest32"),
        math_approx_mode=approx,
    )


def run_finalize(input_tensor, *, variant, init_mode, dest_mode, inv_w, eps, reps, approx=False):
    if variant not in VARIANTS:
        raise ValueError(variant)
    if init_mode not in INIT_MODES:
        raise ValueError(init_mode)
    if dest_mode not in DEST_MODES:
        raise ValueError(dest_mode)
    dtype = ttnn.float32 if dest_mode == "dest32" else ttnn.bfloat16
    if input_tensor.dtype != dtype or input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"input must be {dtype} TILE_LAYOUT for {dest_mode}")
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([TILE, TILE]), dtype, ttnn.TILE_LAYOUT, input_tensor.device(), sharded_memory_config()
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[_VARIANT_ID[variant], 1 if init_mode == "pc" else 0],
        defines=[("FIN_ZONE_TAG", zone_tag(variant, init_mode, dest_mode, approx))],
        runtime_args=[(ttnn.CoreCoord(0, 0), [reps, f32_bits(inv_w), f32_bits(eps)])],
        config=compute_config(dest_mode, approx),
    )
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_IN, input_tensor),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, output),
    ]
    descriptor = ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)
    return ttnn.generic_op([input_tensor, output], descriptor)
