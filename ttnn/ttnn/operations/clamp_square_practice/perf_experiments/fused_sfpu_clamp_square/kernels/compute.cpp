// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// perf experiment `fused_sfpu_clamp_square` — compute kernel bake-off for clamp_square_practice.
//
// CS_VARIANT selects the SFPU stage; everything else (CopyTile unpack/datacopy, PackTile, block_tiles,
// FullBlock CB sync, bf16 DEST, fp32_dest_acc_en=False) is the op's eltwise_chain, unchanged.
//
//   0  baseline     Clamp<D0>{min,max} -> Square<D0>   (the op's current kernel, verbatim)
//   1  fused_sfpi   ONE SFPU pass: load, max(lo), min(hi), x*x, software RNE->bf16 (square's own
//                   float32_to_bf16_rne_prgm incl. mask), store.  Bounds = vFloat locals (SFPLOADI per face).
//   2  fused_creg   same, bounds pinned in programmable const regs (LREG12/13), RNE without the mask
//                   (SFPSTORE to 16-bit DEST keeps the high half), const 0x7fff in LREG14, lsb = (b<<15)>>31.
//   3  fused_stoch  bounds in cregs, rounding by one SFPSTOCHRND fp32->fp16b (BH "Nearest" = nearest-AWAY,
//                   NOT RNE -> differs from torch/baseline on exact ties; precision option only).
//   4  fused_trunc  bounds in cregs, NO rounding (SFPSTORE truncation; precision probe only).
//   5  fused_creg_mask  as 2, plus the 0xffff0000 mask (vUInt local) — isolates the mask's cost.
//
// Raw-LLK justification (variants 1..5): the eltwise helper library has Clamp and Square as two
// separate SFPU elements; each is its own pass over DEST (load/store per row each), Clamp reprograms
// LREG12 twice per row, and with two distinct SFPU element types the chain re-runs both inits per block.
// There is no helper element computing clamp(x)^2 in one DEST pass, so this is a custom UnaryOp element
// (a user-supplied chain element; CopyTile/PackTile/the chain walk are still the helper's).
//
// CT args: [cb_input_tiles, cb_output_tiles, block_tiles]
// RT args: [core_tiles, min_bits (fp32), max_bits (fp32)]

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

#ifndef CS_VARIANT
#define CS_VARIANT 0
#endif
#ifndef CS_UNROLL
#define CS_UNROLL 0
#endif

#ifdef TRISC_MATH
#include "ckernel_sfpu_square.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace cs_fused {

using namespace sfpi;

#define CS_PRAGMA(x) _Pragma(#x)
#define CS_UNROLL_PRAGMA_(n) CS_PRAGMA(GCC unroll n)
#define CS_UNROLL_PRAGMA(n) CS_UNROLL_PRAGMA_(n)

#ifndef CS_PAIR
#define CS_PAIR 0
#endif

// clamp(v)^2 rounded per MODE (fp32 in, value to store out).
template <int MODE>
sfpi_inline vFloat clamp_sq_row(vFloat v, vFloat lo, vFloat hi) {
    v = sfpi::min(sfpi::max(v, lo), hi);
    vFloat r = v * v;
    if constexpr (MODE == 1) {
        r = ckernel::sfpu::float32_to_bf16_rne_prgm(r);
    } else if constexpr (MODE == 2 || MODE == 5) {
        vUInt b = as<vUInt>(r);
        vUInt l = (b << 15) >> 31;  // bit 16 (bf16 lsb) via two immediate shifts: no constant register
        b = b + as<vUInt>(vConstIntPrgm2) + l;
        if constexpr (MODE == 5) {
            b = b & vUInt(0xffff0000u);
        }
        r = as<vFloat>(b);
    }
    return r;
}

// One face (8 sfpi rows). CS_PAIR=0: one row per iteration (dst_reg++). CS_PAIR=1: rows in pairs like
// calculate_square (second store advances dest by 4 via ADDR_MOD_6, programmed by square_init).
template <int MODE>
inline void fused_face(uint32_t lo_bits, uint32_t hi_bits) {
    vFloat lo, hi;
    if constexpr (MODE == 1) {
        lo = ckernel::sfpu::Converter::as_float(lo_bits);
        hi = ckernel::sfpu::Converter::as_float(hi_bits);
    } else {
        lo = vConstFloatPrgm0;
        hi = vConstFloatPrgm1;
    }
#if CS_PAIR
#pragma GCC unroll 4
    for (int d = 0; d < 8; d += 2) {
        vFloat r0 = clamp_sq_row<MODE>(dst_reg[0], lo, hi);
        vFloat r1 = clamp_sq_row<MODE>(dst_reg[1], lo, hi);
        if constexpr (MODE == 3) {
            dst_reg[0] = convert<vFloat16b>(r0, RoundMode::Nearest);
            dst_reg[1].mode(ADDR_MOD_6) = convert<vFloat16b>(r1, RoundMode::Nearest);
        } else {
            dst_reg[0] = r0;
            dst_reg[1].mode(ADDR_MOD_6) = r1;
        }
    }
#else
    CS_UNROLL_PRAGMA(CS_UNROLL)
    for (int d = 0; d < 8; d++) {
        vFloat r = clamp_sq_row<MODE>(dst_reg[0], lo, hi);
        if constexpr (MODE == 3) {
            dst_reg[0] = convert<vFloat16b>(r, RoundMode::Nearest);
        } else {
            dst_reg[0] = r;
        }
        dst_reg++;
    }
#endif
}

// ---------------------------------------------------------------------------------------------------
// Raw-TTI fused face (MODE 6 = exact RNE, MODE 7 = SFPSTOCHRND nearest-away, MODE 9 = ablation: no SFPU).
// Two rows interleaved per step (L0/L1 data, L2/L3 lsb temps), 4 steps per face, fully unrolled.
// Bounds are swapped directly against the read-only const regs LREG12/LREG13 (a const VC is never
// written, so no SFPMOV copy -- exactly the trick ckernel_sfpu_unary_max_min.h uses); 0x7fff in LREG14.
// Second store of each pair uses ADDR_MOD_6 (dest += 4, from square_init) -> no TTINCRWC, no loop.
// RNE to bf16 (bit-identical to calculate_square's float32_to_bf16_rne_prgm, mask dropped because
// SFPSTORE into 16-bit DEST keeps the high half): b += 0x7fff + ((b << 15) >> 31).
template <int MODE>
inline void raw_face(uint32_t, uint32_t) {
    using namespace ckernel;
#pragma GCC unroll 4
    for (int d = 0; d < 4; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 2);
        // max(x, lo): mod1=9 -> VD=max (VC=LREG12 is a constant, never written)
        TTI_SFPSWAP(0, p_sfpu::LREG12, p_sfpu::LREG0, 9);
        TTI_SFPSWAP(0, p_sfpu::LREG12, p_sfpu::LREG1, 9);
        // min(x, hi): VEC_MIN_MAX -> VD=min
        TTI_SFPSWAP(0, p_sfpu::LREG13, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG13, p_sfpu::LREG1, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
        if constexpr (MODE == 6) {
            TTI_SFPSHFT(15, p_sfpu::LREG0, p_sfpu::LREG2, 5);             // L2 = L0 << 15
            TTI_SFPSHFT(15, p_sfpu::LREG1, p_sfpu::LREG3, 5);             // L3 = L1 << 15
            TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, 5);  // L2 = lsb (bit 16)
            TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG3, p_sfpu::LREG3, 5);
            TTI_SFPIADD(0, p_sfpu::LREG14, p_sfpu::LREG0, 4);  // L0 += 0x7fff (no CC)
            TTI_SFPIADD(0, p_sfpu::LREG14, p_sfpu::LREG1, 4);
            TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG0, 4);  // L0 += lsb
            TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG1, 4);
            TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
            TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 2);
        } else {
            // fp32 -> fp16b, BH "Nearest" = round-half-away (as sfpi convert<vFloat16b> emits), store FP16B
            TTI_SFP_STOCH_RND(0, 0, p_sfpu::LCONST_0, p_sfpu::LREG0, p_sfpu::LREG0, 1);
            TTI_SFP_STOCH_RND(0, 0, p_sfpu::LCONST_0, p_sfpu::LREG1, p_sfpu::LREG1, 1);
            TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_7, 0);
            TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_6, 2);
        }
    }
}

// ---------------------------------------------------------------------------------------------------
// MODE 8: SFPLOADMACRO variant of raw_rne. Each SFPLOADMACRO loads a row AND schedules that row's
// x*x (SFPMUL template, VA fixed = the macro's own register, VB := VD via sequence bit 0x80) on the
// MAD sub-unit 5 issued instructions later -- i.e. right after the two clamp swaps -- so the two
// SFPMULs per pair leave the issue stream. Stores cannot be macro-scheduled (Delay is 3 bits, <= 6
// instructions, and the RNE result lands ~14 instructions after the load), so they stay explicit and
// are software-pipelined one pair behind, alternating register sets A={L0,L1|L2,L3} / B={L4,L5|L6,L7}
// so a pair's loads never overwrite the previous pair's pending results. Timing (BH ISA):
//   SFPSWAP: 2 cycles (MAD busy 1st, Simple 2nd); macro-scheduled SFPMAD is NOT scoreboarded -> its
//   consumer is >= 2 issued instructions later (the 2 stores of the other set / 2 SFPNOPs in prologue).
template <int SET>
struct LmRegs {
    static constexpr uint32_t R0 = SET == 0 ? ckernel::p_sfpu::LREG0 : ckernel::p_sfpu::LREG4;
    static constexpr uint32_t R1 = SET == 0 ? ckernel::p_sfpu::LREG1 : ckernel::p_sfpu::LREG5;
    static constexpr uint32_t T0 = SET == 0 ? ckernel::p_sfpu::LREG2 : ckernel::p_sfpu::LREG6;
    static constexpr uint32_t T1 = SET == 0 ? ckernel::p_sfpu::LREG3 : ckernel::p_sfpu::LREG7;
    static constexpr uint32_t M0 = SET == 0 ? 0 : 2;
    static constexpr uint32_t M1 = SET == 0 ? 1 : 3;
};

template <int SET, uint32_t OFF>
inline void lm_load_swap() {
    using namespace ckernel;
    using R = LmRegs<SET>;
    TTI_SFPLOADMACRO((R::M0 << 2) | (R::R0 & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_7, OFF | (R::R0 >> 2));
    TTI_SFPLOADMACRO((R::M1 << 2) | (R::R1 & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_7, (OFF + 2) | (R::R1 >> 2));
    TTI_SFPSWAP(0, p_sfpu::LREG12, R::R0, 9);
    TTI_SFPSWAP(0, p_sfpu::LREG12, R::R1, 9);
    TTI_SFPSWAP(0, p_sfpu::LREG13, R::R0, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG13, R::R1, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);  // MUL(R0) fires after this
}

template <int SET>
inline void lm_store() {  // stores the pair held in SET at counter offsets 0/2, then counter += 4
    using namespace ckernel;
    using R = LmRegs<SET>;
    TTI_SFPSTORE(R::R0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
    TTI_SFPSTORE(R::R1, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 2);  // MUL(R1) of the other set fires after this
}

template <int SET>
inline void lm_rne() {
    using namespace ckernel;
    using R = LmRegs<SET>;
    TTI_SFPSHFT(15, R::R0, R::T0, 5);
    TTI_SFPSHFT(15, R::R1, R::T1, 5);
    TTI_SFPSHFT((-31) & 0xfff, R::T0, R::T0, 5);
    TTI_SFPSHFT((-31) & 0xfff, R::T1, R::T1, 5);
    TTI_SFPIADD(0, p_sfpu::LREG14, R::R0, 4);
    TTI_SFPIADD(0, p_sfpu::LREG14, R::R1, 4);
    TTI_SFPIADD(0, R::T0, R::R0, 4);
    TTI_SFPIADD(0, R::T1, R::R1, 4);
}

inline void lm_face(uint32_t, uint32_t) {
    // pair0 (A) @0/2 ; pair1 (B) @4/6 ; store A ; pair2 (A) @4/6 rel ; store B ; pair3 (B) ; store A ; store B
    lm_load_swap<0, 0>();
    TTI_SFPNOP;  // prologue: stand-ins for the previous pair's stores (MAD-result gap + MUL(R1) trigger)
    TTI_SFPNOP;
    lm_rne<0>();
    lm_load_swap<1, 4>();
    lm_store<0>();
    lm_rne<1>();
    lm_load_swap<0, 4>();
    lm_store<1>();
    lm_rne<0>();
    lm_load_swap<1, 4>();
    lm_store<0>();
    lm_rne<1>();
    lm_store<1>();
}

inline void lm_init() {
    using namespace ckernel;
    // InstructionTemplate[m] = SFPMUL(VA = R_m, VB := VD, VC = LCONST_0) (VD 12..15 = backdoor template write)
    TTI_SFPMUL(p_sfpu::LREG0, 0, p_sfpu::LCONST_0, 12, 0);
    TTI_SFPMUL(p_sfpu::LREG1, 0, p_sfpu::LCONST_0, 13, 0);
    TTI_SFPMUL(p_sfpu::LREG4, 0, p_sfpu::LCONST_0, 14, 0);
    TTI_SFPMUL(p_sfpu::LREG5, 0, p_sfpu::LCONST_0, 15, 0);
    constexpr uint32_t disabled = 7 << 3;  // delay 7: never cancels another macro's pending op
#pragma GCC unroll 4
    for (uint32_t m = 0; m < 4; m++) {
        const uint32_t simple_bits = disabled;
        const uint32_t mad_bits = 0x80 | (5 << 3) | (4 + m);  // template m, VB := VD, 5 instructions later
        const uint32_t round_bits = disabled;
        const uint32_t store_bits = disabled;
        TT_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TT_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TT_SFPCONFIG(0, 4 + m, 0);
    }
    // Misc: StoreMod0 DEFAULT, UsesLoadMod0ForStore none, UnitDelayKind = WaitForElapsedInstructions (all)
    TTI_SFPCONFIG(0xf00, 8, 1);
}

template <int MODE>
inline void fused_init(uint32_t lo_bits, uint32_t hi_bits) {
    // square_init: ADDR_MOD_6 (+4), counters reset, Prgm0..2 = {1, 0x7fff, 0xffff0000}.
    SFPU_UNARY_INIT(square);
    if constexpr (MODE == 8) {
        lm_init();  // templates first: the backdoor template writes use VD=12..15
    }
    if constexpr (MODE != 1) {
        vConstIntPrgm0 = lo_bits;  // LREG12
        vConstIntPrgm1 = hi_bits;  // LREG13
        vConstIntPrgm2 = 0x7fff;   // LREG14 (BH has only 3 programmable cregs)
    }
}

template <int MODE>
inline void fused_tile(uint32_t idst, uint32_t lo_bits, uint32_t hi_bits) {
    if constexpr (MODE == 9) {
        return;  // ablation: no SFPU work at all (output = copy of x; timing floor only)
    } else if constexpr (MODE == 8) {
        _llk_math_eltwise_unary_sfpu_params_(lm_face, idst, VectorMode::RC, lo_bits, hi_bits);
    } else if constexpr (MODE == 6 || MODE == 7) {
        _llk_math_eltwise_unary_sfpu_params_(raw_face<MODE>, idst, VectorMode::RC, lo_bits, hi_bits);
    } else {
        _llk_math_eltwise_unary_sfpu_params_(fused_face<MODE>, idst, VectorMode::RC, lo_bits, hi_bits);
    }
}

}  // namespace cs_fused
#endif  // TRISC_MATH

template <int MODE>
struct FusedClampSquare : compute_kernel_lib::UnaryOp<FusedClampSquare<MODE>, compute_kernel_lib::Dst::D0> {
    uint32_t lo_bits;
    uint32_t hi_bits;
    constexpr FusedClampSquare(uint32_t lo, uint32_t hi) noexcept : lo_bits(lo), hi_bits(hi) {}
    ALWI void init() const { MATH((cs_fused::fused_init<MODE>(lo_bits, hi_bits))); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        MATH((cs_fused::fused_tile<MODE>(slot_offset, lo_bits, hi_bits)));
    }
};

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(2);
    static_assert(
        block_tiles >= 1 && block_tiles <= compute_kernel_lib::DEST_AUTO_LIMIT, "FullBlock block must fit DEST");

    const uint32_t core_tiles = get_arg_val<uint32_t>(0);
    const uint32_t min_bits = get_arg_val<uint32_t>(1);
    const uint32_t max_bits = get_arg_val<uint32_t>(2);

    DataflowBuffer input_tiles(cb_input_tiles);
    DataflowBuffer output_tiles(cb_output_tiles);

    compute_kernel_hw_startup(cb_input_tiles, cb_output_tiles);

    using namespace compute_kernel_lib;
    MaybeDeviceZoneScope("compute_chain");
#if CS_VARIANT == 0
    eltwise_chain(
        IterationShape::tiles(core_tiles).block_size(block_tiles, BlockTailSync::FullBlock),
        CopyTile<
            input(cb_input_tiles, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block),
            Dst::D0>{input_tiles},
        Clamp<Dst::D0>{min_bits, max_bits},
        Square<Dst::D0>{},
        PackTile<output(cb_output_tiles, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>{output_tiles});
#else
    eltwise_chain(
        IterationShape::tiles(core_tiles).block_size(block_tiles, BlockTailSync::FullBlock),
        CopyTile<
            input(cb_input_tiles, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block),
            Dst::D0>{input_tiles},
        FusedClampSquare<CS_VARIANT>{min_bits, max_bits},
        PackTile<output(cb_output_tiles, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>{output_tiles});
#endif
}
