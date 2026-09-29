// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Fused sigmoid-gated RMSNorm compute kernel (kernel_variant >= 1; the legacy kernel is sigmoid_gated_rms_norm.cpp).
//
// Work unit = one head x one 32-row slice = Vt tiles of x and Vt tiles of gate. Per unit, three kinds of DEST pass:
//   A. P = sum_j x_j * x_j                  FPU, ELWMUL accumulates in DEST (one tile)      -> pack "tmp"
//   B. inv = rsqrt(mean_row(P) + eps)       FPU row reduce, then SFPU (add eps + rsqrt)      -> pack "inv"
//   C. out_j = ((x_j * inv) * w_j) * act(g_j), G columns per DEST half (G = 2, or 1 for odd Vt):
//        DEST[k]     = x_j * inv            FPU, column broadcast
//        DEST[k]    *= w_j                  FPU, dest-reuse multiply (wfull = weight row 0 copied to all rows,
//                                           once per core, FPU row-broadcast copy)
//        DEST[G + k] = g_j                  FPU datacopy
//        DEST[k]    *= act(DEST[G + k])     SFPU, one pass (act = sigmoid or silu)          -> pack "out"
// The legacy kernel runs seven passes per unit, each with its own pack and unpack round trip.
// Software pipeline: passes A and B of unit u+1 run between the pass-C groups of unit u (see unit()).
//
// Compile-time options:
//   gate_impl    0 = library silu_tile/sigmoid_tile, then mul_binary_tile,
//                1 = one fused SFPU pass, same arithmetic as the library fp32 sigmoid/silu (bit-exact activation),
//                    constants kept in LRegs,
//                2 = one fused SFPU pass with the exp_21f exponential (about 3 fp32 ulp; not bit-exact).
//   pack_sfpu    1 = all SFPU work runs on the PACK thread, so the MATH thread can run the FPU work of the next DEST
//                half at the same time. 0 = all SFPU work on the MATH thread.
//
// All FPU math uses the kernel's math fidelity (HiFi4) and fp32 DEST, as the legacy kernel does.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_sqrt.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#endif

namespace sgrn {

// Output columns per pass-C DEST group: DEST holds 4 fp32 tiles per half (SyncHalf), G for x*inv*w and G for the
// gate, and G must divide Vt.
template <uint32_t Vt>
constexpr uint32_t group_size() {
    return Vt % 2 == 0 ? 2u : 1u;
}

#if defined(TRISC_MATH) || defined(TRISC_PACK)
// sfpi row offset of tile t relative to the current DEST tile (32 rows of 32 values per tile).
constexpr uint32_t kSfpiTileRows = 32;

// ---- pass B: inv = rsqrt(x + eps) ------------------------------------------------------------------------------
// The 23-bit reciprocal square root of ckernel_sfpu_sqrt.h (_calculate_sqrt_body_<false, true, false>, i.e. what
// rsqrt_tile() runs) with its three constants (sqrt_init<false>: Prgm0 = 0x5f1110a0, Prgm1 = 2.2825186,
// Prgm2 = 2.2533049) as an LReg (the magic number) and literals instead of the programmable constant registers, so
// that it leaves the gate constants (gate_init) in place. Same arithmetic and constants, so the same result.
sfpi_inline sfpi::vFloat rsqrt_body_lreg(const sfpi::vFloat x, const sfpi::vInt magic) {
    constexpr float k1 = 2.2825186f;
    constexpr float k2 = 2.2533049f;
    sfpi::vInt i = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(x) >> 1);
    sfpi::vFloat y = sfpi::as<sfpi::vFloat>(magic - i);
    sfpi::vFloat xy = x * y;
    sfpi::vFloat negative_y = -y;
    sfpi::vFloat c = negative_y * xy;
    sfpi::vFloat infinity = sfpi::sFloat16b(std::numeric_limits<float>::infinity());
    sfpi::vInt infinity_bits = sfpi::as<sfpi::vInt>(infinity);
    y = y * (k1 + c * (k2 + c));
    xy = x * y;
    negative_y = -y;
    sfpi::vFloat one_minus_xyy = 1.0f + (negative_y * xy);
    sfpi::vFloat half_y = sfpi::addexp(y, -1);
    sfpi::vInt x_bits = sfpi::as<sfpi::vInt>(x);
    sfpi::vInt infinity_minus_x_bits = infinity_bits - x_bits;
    v_if(infinity_minus_x_bits != 0 && ckernel::sfpu::_bits_without_sign_(x) != 0) {
        y = one_minus_xyy * half_y + y;
        v_if(x < 0.0f) { y = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;
    }
    v_else { y = sfpi::as<sfpi::vFloat>(infinity_minus_x_bits); }
    v_endif;
    return y;
}

template <uint32_t epsilon_bits>
inline void inv_rms_face() {
    constexpr float eps = __builtin_bit_cast(float, epsilon_bits);
    const sfpi::vInt magic = 0x5f1110a0;
#pragma GCC unroll 2
    for (int d = 0; d < 8; d++) {
        const sfpi::vFloat v = sfpi::vFloat(sfpi::dst_reg[0]) + eps;
        sfpi::dst_reg[0] = rsqrt_body_lreg(v, magic);
        sfpi::dst_reg++;
    }
}

// ---- pass C: DEST[k] *= act(DEST[k + G]) ------------------------------------------------------------------------
// fp32 exp of ckernel_sfpu_exp.h (_sfpu_exp_fp32_accurate_, guarded form) with log2(e) read from Prgm1, one
// polynomial coefficient from Prgm2 and two constants held in LRegs by the caller. Same arithmetic and constants,
// so the same result.
sfpi_inline sfpi::vFloat exp_fp32_hoisted(sfpi::vFloat a, sfpi::vFloat neg_ln2_hi, sfpi::vFloat p0) {
    sfpi::vInt i, e;
    sfpi::vFloat f, r, j, y;
    sfpi::vSMag16 sm;

    j = sfpi::vConstFloatPrgm1 * a;
    r = p0;
    sm = sfpi::convert<sfpi::vSMag16>(j, sfpi::RoundMode::Nearest);
    j = sfpi::convert<sfpi::vFloat>(sm, sfpi::RoundMode::Nearest);

    f = j * neg_ln2_hi + a;
    f = j * -1.42860677e-6f + f;

    r = r * f + 8.37312452e-3f;
    r = r * f + 4.16695364e-2f;
    r = r * f + 1.66664720e-1f;
    r = r * f + sfpi::vConstFloatPrgm2;  // kExpP4
    i = sfpi::abs(sfpi::as<sfpi::vInt>(sm));
    y = r * f + 1.0f;
    i = sfpi::as<sfpi::vInt>(sfpi::copysgn(sfpi::as<sfpi::vFloat>(i), j));
    r = y * f + 1.0f;

    y *= std::numeric_limits<float>::infinity();
    e = sfpi::exexp(r, sfpi::ExponentMode::Biased) + i;
    v_block {
        sfpi::vInt e_lt_255 = __builtin_rvtt_sfpiadd_i(e.get(), -255, sfpi::SFPIADD_MOD1_CC_LT0);
        y = sfpi::setexp(r, e);
        v_if(e_lt_255 < -254) { y = 0.0f; }
        v_endif;
    }
    v_endblock;
    return y;
}

// exp_21f of ckernel_sfpu_exp.h (_sfpu_exp_21f_bf16_) with its constants held in LRegs, fp32 result (no bf16
// rounding). About 3 fp32 ulp, against about 1 ulp for the fp32 exp above.
sfpi_inline sfpi::vFloat exp_21f_hoisted(sfpi::vFloat val, sfpi::vFloat c0, sfpi::vFloat c1, sfpi::vFloat c2) {
    sfpi::vFloat xlog2 = val * sfpi::vConstFloatPrgm1 + 127.f;
    xlog2 = sfpi::clamp(xlog2, 0.0f, 255.0f);
    sfpi::vFloat z = sfpi::as<sfpi::vFloat>(ckernel::sfpu::_float_to_int32_for_exp_21f_(xlog2));
    sfpi::vInt exponential_part = sfpi::exexp(z, sfpi::ExponentMode::Biased);
    sfpi::vMag fractional_part = sfpi::exman(z);
    sfpi::vFloat frac = sfpi::convert<sfpi::vFloat>(fractional_part, sfpi::RoundMode::Nearest);
    frac = ckernel::sfpu::PolynomialEvaluator::eval(frac, c0, c1, c2);
    return sfpi::setexp(frac, exponential_part);
}

// Prgm0 = 2.0 (Newton step of sfpu_reciprocal_iter), Prgm1 = log2(e) = 1/ln2 (both exps), Prgm2 = the last
// coefficient of the fp32 exp polynomial.
constexpr float kExpP4 = 4.99999851e-1f;
inline void gate_init() {
    ckernel::sfpu::sfpu_reciprocal_init<false>();
    sfpi::vConstFloatPrgm1 = 1.4426950216293334961f;
    sfpi::vConstFloatPrgm2 = kExpP4;
}

// One face (8 sfpi rows) of DEST[k] *= act(DEST[k + G]).
template <uint32_t G, uint32_t gate_silu, uint32_t fast_exp>
inline void gate_mul_face() {
    if constexpr (fast_exp) {
        sfpi::vFloat c0 = 1.0017248f, c1 = 7.839635491371155e-08f, c2 = 4.791750143340323e-15f;
#pragma GCC unroll 4
        for (int d = 0; d < 8; d++) {
            const sfpi::vFloat g = sfpi::dst_reg[G * kSfpiTileRows];
            const sfpi::vFloat den = 1.0f + exp_21f_hoisted(-g, c0, c1, c2);
            sfpi::vFloat act = ckernel::sfpu::sfpu_reciprocal_iter<2>(den);
            if constexpr (gate_silu) {
                act = g * act;
            }
            sfpi::dst_reg[0] = sfpi::dst_reg[0] * act;
            sfpi::dst_reg++;
        }
    } else {
        sfpi::vFloat neg_ln2_hi = -6.93145752e-1f, p0 = 1.37805939e-3f;
#pragma GCC unroll 4
        for (int d = 0; d < 8; d++) {
            const sfpi::vFloat g = sfpi::dst_reg[G * kSfpiTileRows];
            const sfpi::vFloat den = 1.0f + exp_fp32_hoisted(-g, neg_ln2_hi, p0);
            sfpi::vFloat act = ckernel::sfpu::sfpu_reciprocal_iter<2>(den);
            if constexpr (gate_silu) {
                act = g * act;
            }
            sfpi::dst_reg[0] = sfpi::dst_reg[0] * act;
            sfpi::dst_reg++;
        }
    }
}
#endif

// Packer side of "tile_regs_wait" for SFPU work done on the PACK thread: block CFG/TDMA until MATH has committed
// the half (the SFPU call starts with a SETC16, so it is held too), as the MoE fused SwiGLU kernel does.
ALWI void pack_wait_for_math() {
    PACK(TTI_SEMWAIT(
        p_stall::STALL_TDMA | p_stall::STALL_CFG,
        ckernel::semaphore::t6_sem(ckernel::semaphore::MATH_PACK),
        p_stall::STALL_ON_ZERO));
}
ALWI void pack_wait_for_sfpu() { PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU)); }

#ifdef TRISC_PACK
// One SFPU function on one DEST tile from the PACK thread, in the DEST half the packer owns (the packer's
// dest_offset_id). Unlike _llk_math_eltwise_unary_sfpu_params_ there is no "wait until the FPU is idle" stall:
// the MATH_PACK semaphore already orders this after MATH's writes to the half, and the FPU may keep working on
// the other half meanwhile.
template <typename F>
ALWI void pack_sfpu_tile(F f, uint32_t dst_index, VectorMode mode) {
    ckernel::math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    _llk_math_eltwise_sfpu_apply_vector_mode_(f, mode);
    ckernel::math::clear_dst_reg_addr();
}
#endif

// ---- pass A: P = sum_j x_j^2 in DEST[0] (ELWMUL accumulates into DEST; the half starts zeroed) -> tmp -----------
template <uint32_t Vt, uint32_t XCB>
ALWI void pass_a(DataflowBuffer& tmp) {
    tmp.reserve_back(1);
    reconfig_data_format(XCB, XCB);
    pack_reconfig_data_format(dfb::tmp);
    mul_init(XCB, XCB);
    tile_regs_acquire();
    for (uint32_t j = 0; j < Vt; j++) {
        mul_tiles(XCB, XCB, j, j, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, dfb::tmp);
    tile_regs_release();
    tmp.push_back(1);
}

// ---- pass B: inv = rsqrt(mean_row(P) + eps), column 0 of DEST[0] -> inv ------------------------------------------
template <uint32_t pack_sfpu, uint32_t epsilon_bits>
ALWI void pass_b(DataflowBuffer& tmp, DataflowBuffer& inv) {
    tmp.wait_front(1);
    inv.reserve_back(1);
    reconfig_data_format(dfb::scaler, dfb::tmp);  // row reduce runs MVMUL with scaler -> SrcA, data -> SrcB
    pack_reconfig_data_format(dfb::inv);
    reduce_init<PoolType::AVG, ReduceDim::REDUCE_ROW>(dfb::tmp, dfb::scaler, dfb::inv);
    tile_regs_acquire();
    reduce_tile<PoolType::AVG, ReduceDim::REDUCE_ROW>(dfb::tmp, dfb::scaler, 0, 0, 0);
    if constexpr (!pack_sfpu) {
        // Common SFPU state (the rsqrt keeps its constants in LRegs; gate_init also readies the gate constants).
        MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::rsqrt>(gate_init)));
        MATH((_llk_math_eltwise_unary_sfpu_params_(inv_rms_face<epsilon_bits>, 0, VectorMode::C)));
    }
    tile_regs_commit();
    if constexpr (pack_sfpu) {
        pack_wait_for_math();
        PACK((pack_sfpu_tile(inv_rms_face<epsilon_bits>, 0, VectorMode::C)));
        pack_wait_for_sfpu();
    } else {
        tile_regs_wait();
    }
    pack_tile(0, dfb::inv);
    tile_regs_release();
    reduce_uninit();
    tmp.pop_front(1);
    inv.push_back(1);
}

// ---- pass C, one group of G columns: out_j = ((x_j * inv) * w_j) * act(g_j) -> out ------------------------------
template <uint32_t G, uint32_t gate_silu, uint32_t gate_impl, uint32_t pack_sfpu, uint32_t XCB>
ALWI void pass_c(uint32_t j0, DataflowBuffer& out) {
    constexpr uint32_t fast_exp = gate_impl == 2 ? 1u : 0u;
    out.reserve_back(G);
    pack_reconfig_data_format(dfb::out);
    tile_regs_acquire();
    reconfig_data_format(XCB, dfb::inv);
    mul_bcast_cols_init(XCB, dfb::inv);
    for (uint32_t k = 0; k < G; k++) {
        mul_tiles_bcast_cols(XCB, dfb::inv, j0 + k, 0, k);
    }
    // SrcA takes the DEST tile (x * inv): give it a 32-bit format (inv's), also when x is bf16.
    reconfig_data_format(dfb::inv, dfb::wfull);
    mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(dfb::wfull);
    for (uint32_t k = 0; k < G; k++) {
        mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(dfb::wfull, j0 + k, k);
    }
    reconfig_data_format_srca(dfb::gate);
    copy_init(dfb::gate);
    for (uint32_t k = 0; k < G; k++) {
        copy_tile(dfb::gate, j0 + k, G + k);
    }
    if constexpr (!pack_sfpu) {
        if constexpr (gate_impl == 0) {
            if constexpr (gate_silu) {
                silu_tile_init();
                for (uint32_t k = 0; k < G; k++) {
                    silu_tile(G + k);
                }
            } else {
                sigmoid_tile_init();
                for (uint32_t k = 0; k < G; k++) {
                    sigmoid_tile(G + k);
                }
            }
            mul_binary_tile_init();
            for (uint32_t k = 0; k < G; k++) {
                mul_binary_tile(k, G + k, k);
            }
        } else {
            MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::silu>(gate_init)));
            for (uint32_t k = 0; k < G; k++) {
                MATH((_llk_math_eltwise_unary_sfpu_params_(gate_mul_face<G, gate_silu, fast_exp>, k, VectorMode::RC)));
            }
        }
    }
    tile_regs_commit();
    if constexpr (pack_sfpu) {
        pack_wait_for_math();
        for (uint32_t k = 0; k < G; k++) {
            PACK((pack_sfpu_tile(gate_mul_face<G, gate_silu, fast_exp>, k, VectorMode::RC)));
        }
        pack_wait_for_sfpu();
    } else {
        tile_regs_wait();
    }
    for (uint32_t k = 0; k < G; k++) {
        pack_tile(k, dfb::out);
    }
    tile_regs_release();
    out.push_back(G);
}

// One unit u of the software pipeline: passes A and B of unit u+1 run between the pass-C groups of unit u, so the
// tmp and inv round trips (pack -> L1 -> unpack) of unit u+1 overlap the pass-C work of unit u.
// Thread order: A(u+1), C(u, first group), B(u+1), C(u, other groups). x of unit u is in XC, of unit u+1 in XN
// (the reader alternates x0/x1), so each unit's tiles start at the front of its DFB. inv holds inv(u) and inv(u+1).
template <
    uint32_t Vt,
    uint32_t gate_silu,
    uint32_t gate_impl,
    uint32_t pack_sfpu,
    uint32_t epsilon_bits,
    uint32_t XC,
    uint32_t XN>
ALWI void unit(
    bool has_next,
    DataflowBuffer& xc,
    DataflowBuffer& xn,
    DataflowBuffer& gate,
    DataflowBuffer& tmp,
    DataflowBuffer& inv,
    DataflowBuffer& out) {
    constexpr uint32_t G = group_size<Vt>();
    gate.wait_front(Vt);
    inv.wait_front(1);
    if (has_next) {
        xn.wait_front(Vt);
        pass_a<Vt, XN>(tmp);
    }
    pass_c<G, gate_silu, gate_impl, pack_sfpu, XC>(0, out);
    if (has_next) {
        pass_b<pack_sfpu, epsilon_bits>(tmp, inv);
    }
    for (uint32_t j0 = G; j0 < Vt; j0 += G) {
        pass_c<G, gate_silu, gate_impl, pack_sfpu, XC>(j0, out);
    }
    xc.pop_front(Vt);
    gate.pop_front(Vt);
    inv.pop_front(1);
}

}  // namespace sgrn

// The SFPU runs on exactly one thread (PACK if pack_sfpu, else MATH); the other thread issues no SFPU work, because
// both threads would share the SFPU's registers.
template <uint32_t Vt, uint32_t gate_silu, uint32_t gate_impl, uint32_t pack_sfpu, uint32_t epsilon_bits>
TT_KERNEL void compute(uint32_t wi_count) {
    compute_kernel_hw_startup(dfb::x0, dfb::x0, dfb::tmp);
    DataflowBuffer x0(dfb::x0);
    DataflowBuffer x1(dfb::x1);
    DataflowBuffer gate(dfb::gate);
    DataflowBuffer weight(dfb::weight);
    DataflowBuffer tmp(dfb::tmp);
    DataflowBuffer inv(dfb::inv);
    DataflowBuffer out(dfb::out);
    DataflowBuffer scaler(dfb::scaler);
    DataflowBuffer wfull(dfb::wfull);
    if (wi_count == 0) {
        return;
    }

    if constexpr (pack_sfpu) {
        // The PACK thread's SFPU state (config, ADDR_MOD_7, Prgm0..2 = gate constants) is set once: nothing else
        // on this thread reprograms it (pass B's rsqrt keeps its constants in LRegs).
        PACK((llk_math_eltwise_unary_sfpu_init<SfpuType::silu>(sgrn::gate_init)));
    }
    x0.wait_front(Vt);
    sgrn::pass_a<Vt, dfb::x0>(tmp);
    scaler.wait_front(1);
    sgrn::pass_b<pack_sfpu, epsilon_bits>(tmp, inv);

    // wfull = weight row 0 broadcast to all 32 rows (FPU row-broadcast copy), so pass C multiplies by the weight
    // with a plain dest-reuse multiply. Once per core; exact (bf16 in, bf16 out).
    weight.wait_front(Vt);
    wfull.reserve_back(Vt);
    reconfig_data_format(dfb::weight, dfb::weight);
    pack_reconfig_data_format(dfb::wfull);
    unary_bcast_init<BroadcastType::ROW>(dfb::weight);
    for (uint32_t j = 0; j < Vt; j++) {
        tile_regs_acquire();
        unary_bcast<BroadcastType::ROW>(dfb::weight, j, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::wfull);
        tile_regs_release();
    }
    unary_bcast_uninit<BroadcastType::ROW>(dfb::weight);
    wfull.push_back(Vt);
    wfull.wait_front(Vt);

    for (uint32_t u = 0; u < wi_count; u += 2) {
        sgrn::unit<Vt, gate_silu, gate_impl, pack_sfpu, epsilon_bits, dfb::x0, dfb::x1>(
            u + 1 < wi_count, x0, x1, gate, tmp, inv, out);
        if (u + 1 < wi_count) {
            sgrn::unit<Vt, gate_silu, gate_impl, pack_sfpu, epsilon_bits, dfb::x1, dfb::x0>(
                u + 2 < wi_count, x1, x0, gate, tmp, inv, out);
        }
    }
}
