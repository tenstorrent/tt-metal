// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// clamp_square_practice compute — ONE fused SFPU pass: y = RNE_bf16(clamp(x, min, max)^2).
// Graduation candidate from perf experiment `fused_sfpu_clamp_square` (drop-in for the op's
// compute kernel: same CT/RT args, same eltwise_chain walk, CopyTile/PackTile, block_tiles, FullBlock).
//
// CT args: [cb_input_tiles, cb_output_tiles, block_tiles]
// RT args: [core_tiles, min_bits (fp32), max_bits (fp32)]
// Define CS_LOADMACRO=1 to select the SFPLOADMACRO schedule (default 0 = plain raw-TTI schedule).
//
// WHY RAW LLK (helper bypassed: compute_kernel_lib::Clamp + compute_kernel_lib::Square chain elements).
// Measured, Blackhole p150b, bf16, fp32_dest_acc_en=False, compute stage isolated (reader/writer NoC
// payload stubbed, (1,1,4096,4096) = 137 tiles/core, DEVICE KERNEL DURATION):
//     helper Clamp->Square chain 78.6 us | raw fused (this file) 40.5 us (1.94x) | + SFPLOADMACRO 37.6 us (2.09x)
//   whole op (real DM): 184-186 us -> 164-165 us (1.11-1.14x); DM-only floor (no SFPU) 154 us.
// Mechanism: the helper runs two SFPU passes over DEST (load/store per row each); Clamp reprograms LREG12
// via SFPLOADI+SFPCONFIG twice per row; Square does a 5-op software RNE; with two distinct SFPU
// element types the chain re-runs both inits per block. Fused: per row load, 2x SFPSWAP against the
// read-only const regs (no SFPMOV copies), SFPMUL, 4-op RNE, store; fully unrolled, no TTINCRWC.
// SFPSWAP is 2 cycles on BH, so the pass is ~11 cycles/row (raw) / ~10 (load-macro) vs ~22 for the helper.
// Helper gap = CAPABILITY: there is no chain element computing clamp(x)^2 in one DEST pass (nor a way to
// fuse two SFPU element bodies into one row loop); the chain walk itself is still the helper's.
//
// Precision: bit-identical to the helper chain for bf16-exact bounds (exhaustive 65536 bf16 inputs).
// For non-bf16-exact bounds (e.g. -0.3/0.7) the helper squares the bf16-TRUNCATED bound (Clamp stores to
// 16-bit DEST first) and is 1 ULP off the op's fp32 reference on every clamped element; this kernel
// squares the fp32 bound and matches torch.clamp(x.float(), lo, hi)**2 -> bf16 exactly.
//
// Hazards (BH ISA, re-checked): SFPSWAP's next-cycle stall and SFPMAD/SFPMUL results are HW-scoreboarded
// for explicit instructions; SFPSHFT/SFPIADD here read their source via VC (mod 4) or a register written
// >= 2 instructions earlier, so the scoreboard's VD-read blind spot is not hit. The load-macro schedule's
// macro-issued SFPMUL is NOT scoreboarded: its consumer is always >= 2 issued instructions later.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

#ifndef CS_LOADMACRO
#define CS_LOADMACRO 0
#endif

#ifdef TRISC_MATH
#include "ckernel_sfpu_square.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace clamp_square_fused {

using namespace ckernel;

// The RNE below drops the 0xffff0000 mask because SFPSTORE into 16-bit DEST keeps the high half.
// With fp32 DEST the rounding must be skipped instead (as calculate_square does) -> not expressible here.
static_assert(!DST_ACCUM_MODE, "fused clamp-square kernel assumes 16-bit (bf16) DEST");

// RNE fp32 -> bf16 on two registers: d += 0x7fff + ((d << 15) >> 31)   (LREG14 = 0x7fff)
template <uint32_t R0, uint32_t R1, uint32_t T0, uint32_t T1>
ALWI void rne_pair() {
    TTI_SFPSHFT(15, R0, T0, 5);  // T = R << 15 (source VC=R, mod 4 | imm)
    TTI_SFPSHFT(15, R1, T1, 5);
    TTI_SFPSHFT((-31) & 0xfff, T0, T0, 5);  // T = bit 16 of R (the bf16 lsb)
    TTI_SFPSHFT((-31) & 0xfff, T1, T1, 5);
    TTI_SFPIADD(0, p_sfpu::LREG14, R0, 4);  // R += 0x7fff, no CC update
    TTI_SFPIADD(0, p_sfpu::LREG14, R1, 4);
    TTI_SFPIADD(0, T0, R0, 4);  // R += lsb
    TTI_SFPIADD(0, T1, R1, 4);
}

// clamp: max against LREG12 (mod1=9: VD=max), min against LREG13 (VEC_MIN_MAX: VD=min). The const VC is
// never written, so no copy is needed.
template <uint32_t R0, uint32_t R1>
ALWI void clamp_pair() {
    TTI_SFPSWAP(0, p_sfpu::LREG12, R0, 9);
    TTI_SFPSWAP(0, p_sfpu::LREG12, R1, 9);
    TTI_SFPSWAP(0, p_sfpu::LREG13, R0, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG13, R1, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
}

// ---- plain raw schedule: one face = 4 pairs of rows; 2nd store of a pair advances dest by 4 (ADDR_MOD_6)
inline void raw_face() {
#pragma GCC unroll 4
    for (int d = 0; d < 4; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 2);
        clamp_pair<p_sfpu::LREG0, p_sfpu::LREG1>();
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
        rne_pair<p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG3>();
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 2);
    }
}

// ---- SFPLOADMACRO schedule: each SFPLOADMACRO loads a row and schedules its x*x (SFPMUL template with
// VA = the macro's own register, VB := VD) on the MAD sub-unit 5 issued instructions later (right after the
// clamp swaps). Stores stay explicit (macro delay <= 6 instructions, the RNE result lands ~14 later) and run
// one pair behind on the other register set: A = {L0,L1 | L2,L3, macros 0,1}, B = {L4,L5 | L6,L7, macros 2,3}.
template <int SET>
struct Set {
    static constexpr uint32_t R0 = SET == 0 ? p_sfpu::LREG0 : p_sfpu::LREG4;
    static constexpr uint32_t R1 = SET == 0 ? p_sfpu::LREG1 : p_sfpu::LREG5;
    static constexpr uint32_t T0 = SET == 0 ? p_sfpu::LREG2 : p_sfpu::LREG6;
    static constexpr uint32_t T1 = SET == 0 ? p_sfpu::LREG3 : p_sfpu::LREG7;
    static constexpr uint32_t M0 = SET == 0 ? 0 : 2;
    static constexpr uint32_t M1 = SET == 0 ? 1 : 3;
};

template <int SET, uint32_t OFF>
ALWI void lm_load_clamp() {
    using S = Set<SET>;
    TTI_SFPLOADMACRO((S::M0 << 2) | (S::R0 & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_7, OFF | (S::R0 >> 2));
    TTI_SFPLOADMACRO((S::M1 << 2) | (S::R1 & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_7, (OFF + 2) | (S::R1 >> 2));
    clamp_pair<S::R0, S::R1>();  // macro SFPMUL(R0) fires after the last swap, SFPMUL(R1) one instruction later
}

template <int SET>
ALWI void lm_store() {  // the pair held in SET -> dest offsets 0/2, then dest += 4
    using S = Set<SET>;
    TTI_SFPSTORE(S::R0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
    TTI_SFPSTORE(S::R1, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 2);
}

template <int SET>
ALWI void lm_rne() {
    using S = Set<SET>;
    rne_pair<S::R0, S::R1, S::T0, S::T1>();
}

inline void lm_face() {
    lm_load_clamp<0, 0>();
    TTI_SFPNOP;  // prologue stand-ins for the previous pair's two stores (macro-MUL result gap)
    TTI_SFPNOP;
    lm_rne<0>();
    lm_load_clamp<1, 4>();
    lm_store<0>();
    lm_rne<1>();
    lm_load_clamp<0, 4>();
    lm_store<1>();
    lm_rne<0>();
    lm_load_clamp<1, 4>();
    lm_store<0>();
    lm_rne<1>();
    lm_store<1>();
}

inline void lm_init() {
    // InstructionTemplate[m] = SFPMUL(VA = R_m, VB := VD, VC = LCONST_0); VD 12..15 = backdoor template write.
    TTI_SFPMUL(p_sfpu::LREG0, 0, p_sfpu::LCONST_0, 12, 0);
    TTI_SFPMUL(p_sfpu::LREG1, 0, p_sfpu::LCONST_0, 13, 0);
    TTI_SFPMUL(p_sfpu::LREG4, 0, p_sfpu::LCONST_0, 14, 0);
    TTI_SFPMUL(p_sfpu::LREG5, 0, p_sfpu::LCONST_0, 15, 0);
    constexpr uint32_t disabled = 7 << 3;  // delay 7: a disabled unit never cancels another macro's op
#pragma GCC unroll 4
    for (uint32_t m = 0; m < 4; m++) {
        const uint32_t mad_bits = 0x80 | (5 << 3) | (4 + m);  // template m, VB := VD, 5 instructions later
        TT_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | disabled);
        TT_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (disabled << 8) | disabled);
        TT_SFPCONFIG(0, 4 + m, 0);
    }
    TTI_SFPCONFIG(0xf00, 8, 1);  // Misc: all sub-units WaitForElapsedInstructions
}

inline void init(uint32_t lo_bits, uint32_t hi_bits) {
    SFPU_UNARY_INIT(square);  // ADDR_MOD_7/6 (+4), counters, SFPU config
    if constexpr (CS_LOADMACRO) {
        lm_init();  // before the const regs: template writes alias VD 12..15
    }
    sfpi::vConstIntPrgm0 = lo_bits;  // LREG12
    sfpi::vConstIntPrgm1 = hi_bits;  // LREG13
    sfpi::vConstIntPrgm2 = 0x7fff;   // LREG14
}

inline void face(uint32_t, uint32_t) {
    if constexpr (CS_LOADMACRO) {
        lm_face();
    } else {
        raw_face();
    }
}

}  // namespace clamp_square_fused
#endif  // TRISC_MATH

// Custom DEST-only chain element (one SFPU pass per tile).
struct ClampSquareFused : compute_kernel_lib::UnaryOp<ClampSquareFused, compute_kernel_lib::Dst::D0> {
    uint32_t lo_bits;
    uint32_t hi_bits;
    constexpr ClampSquareFused(uint32_t lo, uint32_t hi) noexcept : lo_bits(lo), hi_bits(hi) {}
    ALWI void init() const { MATH((clamp_square_fused::init(lo_bits, hi_bits))); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            clamp_square_fused::face, slot_offset, VectorMode::RC, lo_bits, hi_bits)));
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
    eltwise_chain(
        IterationShape::tiles(core_tiles).block_size(block_tiles, BlockTailSync::FullBlock),
        CopyTile<
            input(cb_input_tiles, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block),
            Dst::D0>{input_tiles},
        ClampSquareFused{min_bits, max_bits},
        PackTile<output(cb_output_tiles, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>{output_tiles});
}
