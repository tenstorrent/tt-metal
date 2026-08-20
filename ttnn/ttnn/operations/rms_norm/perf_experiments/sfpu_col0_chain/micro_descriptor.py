# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED micro-bench of rms_norm's `cp_rms_chain` — the SFPU payload alone.

WHAT IS ISOLATED
    The op's chain is

        eltwise_chain(tiles(BLOCK_HT),
            CopyTile<input(cb_sumsq)>, MulUnary{1/W}, AddUnary{eps}, Rsqrt{},
            PackTile<output(cb_rms_recip)>)

    `cb_sumsq` is a REDUCE_ROW output: one meaningful value per row, in COLUMN 0.
    The three SFPU passes nonetheless run `VectorMode::RC` + `ITERATIONS=8`, i.e.
    32 vector ops each (4 faces x 4 row-groups x 2 column parities) = 96 vector
    ops per tile, of which only 8 ever touch column 0.

    Here the tile is seeded into DEST ONCE and packed out ONCE, both OUTSIDE a
    `DeviceZoneScopedN`; inside the zone the chain body runs `reps` times on the
    MATH thread only.  The test reads TRISC_1, so the number is pure SFPU cycles
    (unpack/pack come back ~0 = proof of isolation).  This replaces the op's
    3,409 ns `cp_rms_chain` math zone, which is OCCUPANCY (it encloses the
    helper's `cb_wait_front` on the cross-core broadcast of `cb_sumsq`).

PRECISION CONTRACT
    Every variant runs under the caller's ComputeConfigDescriptor verbatim
    (focus = HiFi2 / fp32_dest_acc_en=False / math_approx_mode=False).  Nothing
    here touches a precision knob; `APPROX` and `DST_ACCUM_MODE` are whatever the
    config defines them to be, identically for baseline and candidates.

THE LADDER (all compute rsqrt(x * 1/W + eps) at the same precision knobs)
    name           passes  scope                              vector ops
    none           -       empty loop (math-loop overhead)     0
    chain_rc       3       VectorMode::RC  (the op TODAY)      96   <- BASELINE
    chain_c        3       VectorMode::C                       48
    chain_cskip    3       VectorMode::C + even-parity stride   24
    fused_rc       1       VectorMode::RC                       32
    fused_c        1       VectorMode::C                        16
    fused_cskip    1       VectorMode::C + even-parity stride    8

    `chain_*` keeps three separate SFPU passes (the helper's shape); `fused_*`
    computes rsqrt(x*a+b) in ONE sfpi body, so the two scalar steps never
    round-trip through DEST.
"""

from __future__ import annotations

import struct

import ttnn

TILE = 32
CB_IN = 0
CB_OUT = 16

# name -> (method, do_sfpu, VectorMode int, ITERATIONS)
#   method 0 = 3 stock scoped passes; 1 = 3 even-parity-stride passes;
#          2 = 1 fused scoped pass;   3 = 1 fused even-parity-stride pass.
# VectorMode: None=0, R=1, C=2, RC=4.
_SCOPE = {
    "none": (0, 0, 4, 8),
    "chain_rc": (0, 1, 4, 8),
    "chain_c": (0, 1, 2, 8),
    "chain_cskip": (1, 1, 2, 4),
    "fused_rc": (2, 1, 4, 8),
    "fused_c": (2, 1, 2, 8),
    "fused_cskip": (3, 1, 2, 4),
}
VARIANTS = tuple(_SCOPE)

# vector ops actually issued per tile, for the ns/vector column
VEC_OPS = {
    "none": 0,
    "chain_rc": 96,
    "chain_c": 48,
    "chain_cskip": 24,
    "fused_rc": 32,
    "fused_c": 16,
    "fused_cskip": 8,
}

DESC = {
    "none": "empty math loop (overhead floor)",
    "chain_rc": "3 passes, VectorMode::RC        <- the op TODAY",
    "chain_c": "3 passes, VectorMode::C",
    "chain_cskip": "3 passes, C + even-parity stride",
    "fused_rc": "1 fused pass, VectorMode::RC",
    "fused_c": "1 fused pass, VectorMode::C",
    "fused_cskip": "1 fused pass, C + even-parity stride",
}


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


_KERNEL = r"""
// ---------------------------------------------------------------------------
// rms_norm perf_experiments/sfpu_col0_chain — ISOLATED micro-bench kernel.
//
// RAW LLK IS DELIBERATE HERE (perf-lab rules): the helper being bypassed is
// `compute_kernel_lib::eltwise_chain` with MulUnary/AddUnary/Rsqrt.  Those op
// structs call `mul_unary_tile` / `add_unary_tile` / `rsqrt_tile`, each of which
// HARDCODES `VectorMode::RC` and `ITERATIONS=8` at its `SFPU_UNARY_CALL` site --
// neither the vector mode nor the iteration count nor an address stride is
// reachable from the chain's public API.  The mechanism exploited is that a
// REDUCE_ROW result lives only in column 0, which is covered by the EVEN-parity
// vectors of faces 0 and 2 (`VectorMode::C` + `dst_reg += 2`), so 8 of the 96
// vector ops carry all the data.
// ---------------------------------------------------------------------------
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#include "ckernel_sfpu_binop_with_unary.h"
#include "sfpu/ckernel_sfpu_converter.h"
#endif

using ckernel::VectorMode;

constexpr uint32_t method    = get_compile_time_arg_val(0);
constexpr uint32_t do_sfpu   = get_compile_time_arg_val(1);
constexpr uint32_t vmode_val = get_compile_time_arg_val(2);
constexpr uint32_t iters     = get_compile_time_arg_val(3);

// ---- (a) the three stock passes, with BOTH knobs threaded through -----------
template <int IT>
ALWI void mul_unary_scoped(uint32_t idst, uint32_t p, VectorMode vm) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_binop_with_scalar,
        (APPROX, ckernel::MUL_UNARY, IT), idst, vm, p));
}
template <int IT>
ALWI void add_unary_scoped(uint32_t idst, uint32_t p, VectorMode vm) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_binop_with_scalar,
        (APPROX, ckernel::ADD_UNARY, IT), idst, vm, p));
}
template <int IT>
ALWI void rsqrt_scoped(uint32_t idst, VectorMode vm) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_rsqrt,
        (APPROX, IT, DST_ACCUM_MODE, false /*FAST_APPROX*/, false /*legacy_compat*/), idst, vm));
}

#ifdef TRISC_MATH
// ---- (b) even-parity stride bodies: visit offsets 0,2,4,6 (dst_reg += 2), so
// the net advance is +8 == the stock ITERATIONS=8 and VectorMode::C's
// face0 -> face2 stepping composes unchanged.  Column 0 is EVEN, so the skipped
// odd-parity vectors only ever touch columns 1,3,..,15. ---------------------
template <int BINOP>
sfpi_inline void skip_binop_body(uint32_t param) {
    const sfpi::vFloat p = ckernel::sfpu::Converter::as_float(param);
    for (int d = 0; d < 4; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = (BINOP == ckernel::sfpu::MUL) ? (v * p) : (v + p);
        sfpi::dst_reg += 2;
    }
}
sfpi_inline void skip_mul_body(uint32_t p) { skip_binop_body<ckernel::sfpu::MUL>(p); }
sfpi_inline void skip_add_body(uint32_t p) { skip_binop_body<ckernel::sfpu::ADD>(p); }

sfpi_inline void skip_rsqrt_body() {
    for (int d = 0; d < 4; d++) {
        sfpi::vFloat t = ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false>(sfpi::dst_reg[0]);
        if constexpr (!DST_ACCUM_MODE) { t = sfpi::convert<sfpi::vFloat16b>(t, sfpi::RoundMode::Nearest); }
        sfpi::dst_reg[0] = t;
        sfpi::dst_reg += 2;
    }
}

// ---- (c) FUSED body: rsqrt(x*a + b) in one pass.  Same three arithmetic steps
// at the same precision knobs; the two scalar steps simply never round-trip
// through DEST (which at fp32_dest_acc_en=False is a 16-bit datum), so this is
// at least as accurate as the 3-pass form. -----------------------------------
// The two scalars are RE-MATERIALISED inside the loop (one SFPLOADI each)
// instead of being hoisted into two vFloats live across the whole unrolled
// body: holding them alongside the sqrt body's temporaries overflows the SFPU
// register file and the sfpi backend ICEs with "cannot store sfpu register
// (register spill)".
template <int N, int STRIDE>
sfpi_inline void fused_body(uint32_t a_bits, uint32_t b_bits) {
    for (int d = 0; d < N; d++) {
        sfpi::vFloat v =
            sfpi::dst_reg[0] * ckernel::sfpu::Converter::as_float(a_bits) +
            ckernel::sfpu::Converter::as_float(b_bits);
        sfpi::vFloat t = ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false>(v);
        if constexpr (!DST_ACCUM_MODE) { t = sfpi::convert<sfpi::vFloat16b>(t, sfpi::RoundMode::Nearest); }
        sfpi::dst_reg[0] = t;
        sfpi::dst_reg += STRIDE;
    }
}
sfpi_inline void fused_body_8(uint32_t a, uint32_t b) { fused_body<8, 1>(a, b); }
sfpi_inline void fused_body_skip(uint32_t a, uint32_t b) { fused_body<4, 2>(a, b); }
#endif

ALWI void chain_once(uint32_t idst, uint32_t a_bits, uint32_t b_bits) {
    const VectorMode vm = static_cast<VectorMode>(vmode_val);
    if constexpr (method == 0) {
        if constexpr (iters == 8) {
            mul_unary_scoped<8>(idst, a_bits, vm);
            add_unary_scoped<8>(idst, b_bits, vm);
            rsqrt_scoped<8>(idst, vm);
        }
    } else if constexpr (method == 1) {
        MATH((_llk_math_eltwise_unary_sfpu_params_(skip_mul_body, idst, VectorMode::C, a_bits)));
        MATH((_llk_math_eltwise_unary_sfpu_params_(skip_add_body, idst, VectorMode::C, b_bits)));
        MATH((_llk_math_eltwise_unary_sfpu_params_(skip_rsqrt_body, idst, VectorMode::C)));
    } else if constexpr (method == 2) {
        MATH((_llk_math_eltwise_unary_sfpu_params_(fused_body_8, idst, vm, a_bits, b_bits)));
    } else {
        MATH((_llk_math_eltwise_unary_sfpu_params_(fused_body_skip, idst, VectorMode::C, a_bits, b_bits)));
    }
}

void kernel_main() {
    constexpr uint32_t cb_in = 0, cb_out = 16;
    const uint32_t reps   = get_arg_val<uint32_t>(0);
    const uint32_t a_bits = get_arg_val<uint32_t>(1);   // 1/W  (fp32 bits)
    const uint32_t b_bits = get_arg_val<uint32_t>(2);   // eps  (fp32 bits)

    compute_kernel_hw_startup(cb_in, cb_in, cb_out);
    copy_tile_init(cb_in);
    // Both SFPU families used by the chain need their init; the fused/skip
    // rsqrt bodies reuse the stock rsqrt constant programming.
    binop_with_scalar_tile_init();
    rsqrt_tile_init();

    cb_reserve_back(cb_in, 1);
    cb_push_back(cb_in, 1);      // sharded input already resident
    cb_wait_front(cb_in, 1);

    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);      // seed DEST[0] once — OUTSIDE the timed zone
    {
        DeviceZoneScopedN("RMS_CHAIN");
        for (uint32_t r = 0; r < reps; ++r) {
            if constexpr (do_sfpu) {
                chain_once(0, a_bits, b_bits);
            }
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out, 0);     // pack once (post-zone) so the host can verify
    cb_push_back(cb_out, 1);
    tile_regs_release();
    cb_pop_front(cb_in, 1);
}
"""


def _single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def create_sharded_memory_config():
    return ttnn.create_sharded_memory_config(
        shape=(TILE, TILE),
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def create_program_descriptor(input_tensor, output_tensor, *, variant, inv_w, eps, reps, compute_config):
    if variant not in _SCOPE:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    method, do_sfpu, vmode_val, iters = _SCOPE[variant]
    compute = ttnn.KernelDescriptor(
        kernel_source=_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[method, do_sfpu, vmode_val, iters],
        runtime_args=[(ttnn.CoreCoord(0, 0), [reps, f32_bits(inv_w), f32_bits(eps)])],
        config=compute_config,
    )
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_IN, input_tensor),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, output_tensor),
    ]
    return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)


def run_micro(input_tensor, *, variant, inv_w, eps, reps, compute_config, out_dtype=ttnn.float32):
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([TILE, TILE]),
        out_dtype,
        ttnn.TILE_LAYOUT,
        input_tensor.device(),
        create_sharded_memory_config(),
    )
    desc = create_program_descriptor(
        input_tensor, output, variant=variant, inv_w=inv_w, eps=eps, reps=reps, compute_config=compute_config
    )
    return ttnn.generic_op([input_tensor, output], desc)
