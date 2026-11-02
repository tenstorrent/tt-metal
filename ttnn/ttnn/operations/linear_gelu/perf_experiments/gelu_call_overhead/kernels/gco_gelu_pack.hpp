// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// gelu_call_overhead bench: packer-thread (TRISC2) GELU + pack of one matmul output subblock.
//
// Raw LLK: bypasses compute_kernel_lib::apply_activation_from_pack + gelu_tanh_tile_pack (SFPU_UNARY_CALL)
// + pack_block in matmul_block's pack stage. The helper exposes no hook between the SFPU pass and the pack,
// so per-subblock address setup, per-tile STALLWAIT(SFPU, MATH), the per-subblock packer reconfig and the
// GELU/pack interleave can only be changed here. The GELU arithmetic is calculate_gelu_tanh_fast verbatim.
//
// Compile-time knobs (each 0/1 unless noted), set by the bench harness:
//   GCO_BATCH       one sfpu start/done per subblock (faces walk contiguous DEST tiles) instead of per tile
//   GCO_NOSTALL     drop TTI_STALLWAIT(STALL_SFPU, MATH) (the FPU-idle wait meant for math-thread SFPU)
//   GCO_NORECONFIG  drop the per-subblock pack_reconfig_data_format (single pack target, set at startup)
//   GCO_INTERLEAVE  pack tile t right after GELU(t), so the packer overlaps GELU(t+1)
//   GCO_UNROLL      unroll factor of the per-face 8-row loop (default 8 = shared header)
//   GCO_HOIST       keep k loop-invariant immediates in LREGs across the face loop instead of an SFPLOADI per use:
//                   1 = {2.0}, 2 = {2.0, 127.0}, 3 = {2.0, 127.0, 254.0} (same instructions otherwise, bit-identical)
//   Perf-only diagnostics (wrong results / extra work): GCO_PACK_REPS (0 = no pack, 2 = pack each tile twice),
//   GCO_BODY_REPS (face body k times), GCO_CALL_REPS (whole per-tile call k times), GCO_EMPTY (no SFPU body, keep the
//   call scaffolding).
#pragma once

#include "api/compute/eltwise_unary/gelu.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"

#ifndef GCO_BATCH
#define GCO_BATCH 0
#endif
#ifndef GCO_NOSTALL
#define GCO_NOSTALL 0
#endif
#ifndef GCO_NORECONFIG
#define GCO_NORECONFIG 0
#endif
#ifndef GCO_INTERLEAVE
#define GCO_INTERLEAVE 0
#endif
#ifndef GCO_UNROLL
#define GCO_UNROLL 8
#endif
#ifndef GCO_BODY_REPS
#define GCO_BODY_REPS 1
#endif
#ifndef GCO_CALL_REPS
#define GCO_CALL_REPS 1
#endif
#ifndef GCO_HOIST
#define GCO_HOIST 0
#endif
#ifndef GCO_PACK_REPS
#define GCO_PACK_REPS 1
#endif
#ifndef GCO_EMPTY
#define GCO_EMPTY 0
#endif

#define GCO_STR_(x) #x
#define GCO_PRAGMA_(x) _Pragma(GCO_STR_(x))
#define GCO_UNROLL_PRAGMA(n) GCO_PRAGMA_(GCC unroll n)

#ifdef TRISC_PACK
namespace ckernel::sfpu {

// calculate_gelu_tanh_fast<is_fp32_dest_acc_en, 8> with a configurable unroll; arithmetic unchanged.
template <bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void gco_gelu_tanh_fast_face() {
#if GCO_HOIST >= 1
    const sfpi::vFloat k_two = 2.0f;
#else
    constexpr float k_two = 2.0f;
#endif
#if GCO_HOIST >= 2
    const sfpi::vFloat k_bias = 127.0f;
#else
    constexpr float k_bias = 127.0f;
#endif
#if GCO_HOIST >= 3
    const sfpi::vFloat k_hi = 254.0f;
    const sfpi::vFloat k_lo = 0.0f;
#endif
    GCO_UNROLL_PRAGMA(GCO_UNROLL)
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat x2 = x * x;
        sfpi::vFloat t = x2 * sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0;
        sfpi::vFloat xlog2 = x * t + k_bias;
#if GCO_HOIST >= 3
        xlog2 = sfpi::clamp(xlog2, k_lo, k_hi);
#else
        xlog2 = sfpi::clamp(xlog2, 0.0f, 254.0f);
#endif

        sfpi::vFloat z = sfpi::as<sfpi::vFloat>(_float_to_int32_for_exp_21f_(xlog2));
        sfpi::vInt exponential_part = sfpi::exexp(z, sfpi::ExponentMode::Biased);
        sfpi::vMag fractional_part = sfpi::exman(z);
        sfpi::vFloat frac = sfpi::convert<sfpi::vFloat>(fractional_part, sfpi::RoundMode::Nearest);
        frac = PolynomialEvaluator::eval(
            frac, 1.0f, sfpi::vConstFloatPrgm2, GELU_TANH_FAST_EXP2_C2, GELU_TANH_FAST_EXP2_C3, GELU_TANH_FAST_EXP2_C4);
        sfpi::vFloat e = sfpi::setexp(frac, exponential_part);

        sfpi::vFloat den = e + 1.0f;
        sfpi::vFloat r = sfpi::approx_recip(den);
        r = r * (k_two - den * r);
        r = r * (k_two - den * r);
        sfpi::vFloat result = x * r;

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
#endif

namespace gco {

#ifdef TRISC_PACK
FORCE_INLINE void face_body() {
#if !GCO_EMPTY
    for (uint32_t rep = 0; rep < GCO_BODY_REPS; ++rep) {
#if GCO_UNROLL == 8 && GCO_HOIST == 0
        ckernel::sfpu::calculate_gelu_tanh_fast<DST_ACCUM_MODE, 8>();
#else
        ckernel::sfpu::gco_gelu_tanh_fast_face<DST_ACCUM_MODE, 8>();
#endif
    }
#endif
}

FORCE_INLINE void sfpu_start(uint32_t dst_index) {
    ckernel::math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
#if !GCO_NOSTALL
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
#endif
}

// One tile (4 faces x 8 x 4-row groups) at the current DEST RWC position.
FORCE_INLINE void gelu_tile_faces() {
#pragma GCC unroll 0
    for (uint32_t face = 0; face < 4; ++face) {
        face_body();
        ckernel::math::inc_dst_addr<8>();
        ckernel::math::inc_dst_addr<8>();
    }
}

FORCE_INLINE void gelu_tile_percall(uint32_t tile) {
    for (uint32_t rep = 0; rep < GCO_CALL_REPS; ++rep) {
        sfpu_start(tile);
        gelu_tile_faces();
        ckernel::math::clear_dst_reg_addr();
    }
}
#endif

FORCE_INLINE void gelu_pack_subblock(uint32_t out_cb, uint32_t num_tiles) {
#ifdef TRISC_PACK
    TTI_SEMWAIT(
        p_stall::STALL_TDMA | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO);
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset());

#if GCO_INTERLEAVE
#if !GCO_NORECONFIG
    pack_reconfig_data_format(out_cb);
#endif
#if GCO_BATCH
    sfpu_start(0);
#endif
    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
#if GCO_BATCH
        gelu_tile_faces();
#else
        gelu_tile_percall(tile);
#endif
        TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU);
        llk_matmul_pack<DST_ACCUM_MODE, false, PackMode::Default>(tile, out_cb, 1);
    }
#if GCO_BATCH
    ckernel::math::clear_dst_reg_addr();
#endif
#else
#if GCO_BATCH
    sfpu_start(0);
    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        gelu_tile_faces();
    }
    ckernel::math::clear_dst_reg_addr();
#else
    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        gelu_tile_percall(tile);
    }
#endif
    TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU);
#if !GCO_NORECONFIG
    pack_reconfig_data_format(out_cb);
#endif
#if GCO_PACK_REPS >= 1
    llk_matmul_pack<DST_ACCUM_MODE, false, PackMode::Default>(0, out_cb, num_tiles);
#endif
#if GCO_PACK_REPS >= 2
    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        llk_pack<DST_ACCUM_MODE, true, PackMode::Default>(tile, out_cb, tile);
    }
#endif
#endif
#endif
}

}  // namespace gco
