// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post (Perf 1, Blackhole): the mix / y passes as SFPLOADMACRO streams.
// (Textually included inside namespace ckernel::sfpu by mhc_post_mix_sfpu.hpp.)
// Measured (perf_experiments/mix_sfpu, v2): per tile mix+ss / mix / y 721 / 625 / 429 -> 515 / 445 /
// 334 cycles; focus collapse 96.4 -> 87.9 us end to end; bit-identical.
//
// RAW LLK, bypassing sfpi (the vFloat / dst_reg programs of the non-Blackhole passes) and the
// eltwise_chain SFPU helpers: sfpi issues every SFPLOAD / SFPMAD / SFPSTORE through the single
// SFPU issue slot (one instruction per cycle), so a pass costs loads + MADs + stores cycles.
// SFPLOADMACRO issues the load AND schedules that vector's MAD and STORE on the MAD / store
// sub-units in later cycles, so they run in parallel with the following loads: a pass costs ~its
// load count. The arithmetic is unchanged instruction for instruction (same SFPMUL / SFPMAD
// operands VA = coefficient, VB = data, VC = accumulator, mod 0, fp32 DEST round trip through
// SFPLOAD / SFPSTORE mod0 0 exactly as sfpi does between passes), so results are bit-identical.
//
// Each pass is decomposed into "sub-passes" over the 32 SFPU vectors of a tile, each one term of
// the original chain (the accumulator round-trips DEST between terms, which is exact in fp32 DEST):
//   Mul   : acc = c * acc                    (c = packed coefficient, quadrant QC of D3)
//   Mac   : acc = c * X + acc                (c packed)
//   Lane  : acc = D3 * X + acc               (D3 lane-aligned, MixPass3)
//   Sq    : acc = X * X + acc                (the ss term)
// Schedules (slot = SFPU instruction issued; "delay kind" = WaitForElapsedInstructions, so a
// macro op scheduled from slot s with delay d executes alongside slot s+1+d):
//   Mul : s: LM acc->VD  | MUL VD = L0*VD @s+1 | STORE VD @s+3                     1 slot / vector
//   Mac : s: LM acc->VD, s+1: LD X->L5 | MAD VD = L0*L5+VD @s+2 | STORE @s+4          2 slots / vector
//   Lane: s: LM acc->VD, s+1: LD D3->L0, s+2: LD X->L5 | MAD @s+3 | STORE @s+5        3 slots / vector
//   Sq  : s: LM acc->VD, s+1: LD X->L5 | MAD VD = L5*L5+VD @s+2 | STORE @s+4          2 slots / vector
// VD rotates over L1..L4 (one per quadrant of a row group); L0 = coefficient, L5 = data operand.
// The packed-coefficient sub-passes reload L0 once per row group, with one SFPNOP so the previous
// row group's last MAD has read it. Each sub-pass ends with a short SFPNOP drain (the delay
// counters only advance on issued SFPU instructions).

namespace mhc_lm {

// The macro config is written once per kernel (configured): no other code in the mhc_post compute
// kernel uses SFPLOADMACRO (sfpi / SFPU inits, sfpu_reduce and the hc program do not touch the
// LoadMacroConfig or the instruction templates). If that ever changes, configure per pass call
// (~17 cycles each). The row-group advance is the last load's ADDR_MOD_6 post-increment (re-set at
// every pass entry), and a pass drains once at its end (its sub-pass schedules do not collide).
constexpr bool V2 = true;
inline bool configured = false;

constexpr uint32_t LC = 0;  // coefficient
constexpr uint32_t LX = 5;  // data operand
constexpr uint32_t VD0 = 1, VD1 = 2, VD2 = 3, VD3 = 4;

// Macro indices
constexpr uint32_t M_MUL = 0, M_MAC = 1, M_LANE = 2, M_SQ = 3;

constexpr uint32_t imm(uint32_t slot, uint32_t vec) { return 2 * (slot * mhc_post_mix::VECS + vec); }

#define MHC_LM(MACRO, VD, IMM) TTI_SFPLOADMACRO((((MACRO) << 2) | ((VD) & 3)), 0, ADDR_MOD_7, ((IMM) | ((VD) >> 2)))
#define MHC_LD(LREG, IMM) TTI_SFPLOAD((LREG), 0, ADDR_MOD_7, (IMM))
// last load of a row group: v2 advances the row group with its ADDR_MOD_6 (+4 dest rows)
#define MHC_AM (V2 ? ADDR_MOD_6 : ADDR_MOD_7)
#define MHC_LM_LAST(MACRO, VD, IMM) TTI_SFPLOADMACRO((((MACRO) << 2) | ((VD) & 3)), 0, MHC_AM, ((IMM) | ((VD) >> 2)))
#define MHC_LD_LAST(LREG, IMM) TTI_SFPLOAD((LREG), 0, MHC_AM, (IMM))

// Templates + the 4 macros (all passes share them; written at every pass entry: the config is
// global SFPU state other SFPU code may also use).
inline void configure() {
    if constexpr (V2) {
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_6);
        if (configured) {
            return;
        }
        configured = true;
    }
    // Instruction templates (backdoor load: VD = 12 + i)
    TTI_SFPMUL(LC, 0, p_sfpu::LCONST_0, 12, 0);  // T0: VD = L0 * VB(=VD)      [+ LCONST_0]
    TTI_SFPMAD(LC, LX, 0, 13, 0);                // T1: VD = L0 * L5 + VC(=VD)
    TTI_SFPMAD(LX, LX, 0, 14, 0);                // T2: VD = L5 * L5 + VC(=VD)
    // Sequence bits per sub-unit: bit7 VB=VD (else VC=VD), bits5:3 delay, bits2:0 template (4+i) /
    // 3 = SFPSTORE. Word = store << 24 | round << 16 | mad << 8 | simple.
    constexpr uint32_t mul_mad = 0x80 | (0 << 3) | (4 + 0), mul_st = (2 << 3) | 3;
    constexpr uint32_t mac_mad = (1 << 3) | (4 + 1), mac_st = (3 << 3) | 3;
    constexpr uint32_t lane_mad = (2 << 3) | (4 + 1), lane_st = (4 << 3) | 3;
    constexpr uint32_t sq_mad = (1 << 3) | (4 + 2), sq_st = (3 << 3) | 3;
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mul_mad << 8));
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (mul_st << 8));
    TTI_SFPCONFIG(0, 4 + M_MUL, 0);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mac_mad << 8));
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (mac_st << 8));
    TTI_SFPCONFIG(0, 4 + M_MAC, 0);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (lane_mad << 8));
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (lane_st << 8));
    TTI_SFPCONFIG(0, 4 + M_LANE, 0);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (sq_mad << 8));
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (sq_st << 8));
    TTI_SFPCONFIG(0, 4 + M_SQ, 0);
    // Misc: StoreMod0 = 0, UsesLoadMod0ForStore = 0xf, UnitDelayKind = 0xf (WaitForElapsedInstructions)
    TTI_SFPCONFIG(0xff0, 8, 1);
    TTI_SFPNOP;
}

// Walk the 8 row groups: the row-group body is recorded once (and executed), then replayed.
template <uint32_t LEN, class Body>
inline void walk(Body body) {
    static_assert(LEN <= 16, "SFPU half of the replay buffer");
    load_replay_buf<Exec>(0, LEN, body);
    if constexpr (!V2) {
        TTI_INCRWC(0, 4, 0, 0);
    }
#pragma GCC unroll 3
    for (uint32_t rg = 1; rg < 4; ++rg) {
        lltt::replay(0, LEN);
        if constexpr (!V2) {
            TTI_INCRWC(0, 4, 0, 0);
        }
    }
    // += 8 vectors to the second half (faces 2, 3)
    TTI_INCRWC(0, 4, 0, 0);
    TTI_INCRWC(0, 4, 0, 0);
    TTI_INCRWC(0, 4, 0, 0);
    TTI_INCRWC(0, 4, 0, 0);
#pragma GCC unroll 4
    for (uint32_t rg = 0; rg < 4; ++rg) {
        lltt::replay(0, LEN);
        if constexpr (!V2) {
            TTI_INCRWC(0, 4, 0, 0);
        }
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
}

inline void drain_end() {
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

inline void drain() {
    if constexpr (V2) {
        return;
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

using mhc_mix_detail::Q0;
using mhc_mix_detail::Q1;
using mhc_mix_detail::Q2;
using mhc_mix_detail::Q3;

// acc = c * acc   (c = packed coefficient quadrant QC of D3)
template <uint32_t ACC, uint32_t QC>
inline void sub_mul() {
    walk<6>([] {
        MHC_LD(LC, imm(3, QC));
        MHC_LM(M_MUL, VD0, imm(ACC, Q0));
        MHC_LM(M_MUL, VD1, imm(ACC, Q1));
        MHC_LM(M_MUL, VD2, imm(ACC, Q2));
        MHC_LM_LAST(M_MUL, VD3, imm(ACC, Q3));
        TTI_SFPNOP;
    });
    drain();
}

// acc = c * X + acc   (c packed)
template <uint32_t ACC, uint32_t X, uint32_t QC>
inline void sub_mac() {
    walk<10>([] {
        MHC_LD(LC, imm(3, QC));
        MHC_LM(M_MAC, VD0, imm(ACC, Q0));
        MHC_LD(LX, imm(X, Q0));
        MHC_LM(M_MAC, VD1, imm(ACC, Q1));
        MHC_LD(LX, imm(X, Q1));
        MHC_LM(M_MAC, VD2, imm(ACC, Q2));
        MHC_LD(LX, imm(X, Q2));
        MHC_LM(M_MAC, VD3, imm(ACC, Q3));
        MHC_LD_LAST(LX, imm(X, Q3));
        TTI_SFPNOP;
    });
    drain();
}

// acc = D3 * X + acc   (D3 lane-aligned)
template <uint32_t ACC, uint32_t X>
inline void sub_lane() {
    walk<12>([] {
        MHC_LM(M_LANE, VD0, imm(ACC, Q0));
        MHC_LD(LC, imm(3, Q0));
        MHC_LD(LX, imm(X, Q0));
        MHC_LM(M_LANE, VD1, imm(ACC, Q1));
        MHC_LD(LC, imm(3, Q1));
        MHC_LD(LX, imm(X, Q1));
        MHC_LM(M_LANE, VD2, imm(ACC, Q2));
        MHC_LD(LC, imm(3, Q2));
        MHC_LD(LX, imm(X, Q2));
        MHC_LM(M_LANE, VD3, imm(ACC, Q3));
        MHC_LD(LC, imm(3, Q3));
        MHC_LD_LAST(LX, imm(X, Q3));
    });
    drain();
}

// acc = X * X + acc
template <uint32_t ACC, uint32_t X>
inline void sub_sq() {
    walk<8>([] {
        MHC_LM(M_SQ, VD0, imm(ACC, Q0));
        MHC_LD(LX, imm(X, Q0));
        MHC_LM(M_SQ, VD1, imm(ACC, Q1));
        MHC_LD(LX, imm(X, Q1));
        MHC_LM(M_SQ, VD2, imm(ACC, Q2));
        MHC_LD(LX, imm(X, Q2));
        MHC_LM(M_SQ, VD3, imm(ACC, Q3));
        MHC_LD_LAST(LX, imm(X, Q3));
    });
    drain();
}

}  // namespace mhc_lm

// D0 = A0*D0 ; D0 = A1*D1 + D0 ; D0 = A2*D2 + D0
inline void mhc_mix_pass1() {
    mhc_lm::configure();
    mhc_lm::sub_mul<0, mhc_lm::Q0>();
    mhc_lm::sub_mac<0, 1, mhc_lm::Q1>();
    mhc_lm::sub_mac<0, 2, mhc_lm::Q2>();
    if constexpr (mhc_lm::V2) {
        mhc_lm::drain_end();
    }
}

// D0 = A3*D1 + D0
inline void mhc_mix_pass2() {
    mhc_lm::configure();
    mhc_lm::sub_mac<0, 1, mhc_lm::Q3>();
    if constexpr (mhc_lm::V2) {
        mhc_lm::drain_end();
    }
}

// D0 = D3*D1 + D0 [; D2 = D0*D0 + D2]
template <bool SS>
inline void mhc_mix_pass3() {
    mhc_lm::configure();
    mhc_lm::sub_lane<0, 1>();
    if constexpr (SS) {
        mhc_lm::sub_sq<2, 0>();
    }
    if constexpr (mhc_lm::V2) {
        mhc_lm::drain_end();
    }
}

// D0 = P0*D0 ; D0 = P1*D1 + D0
inline void mhc_y_pass1() {
    mhc_lm::configure();
    mhc_lm::sub_mul<0, mhc_lm::Q0>();
    mhc_lm::sub_mac<0, 1, mhc_lm::Q1>();
    if constexpr (mhc_lm::V2) {
        mhc_lm::drain_end();
    }
}

// D0 = P2*D1 + D0 ; D0 = P3*D2 + D0
inline void mhc_y_pass2() {
    mhc_lm::configure();
    mhc_lm::sub_mac<0, 1, mhc_lm::Q2>();
    mhc_lm::sub_mac<0, 2, mhc_lm::Q3>();
    if constexpr (mhc_lm::V2) {
        mhc_lm::drain_end();
    }
}

// ---- Perf 2 (mix_reuse): c-pair window. D0 / D1 = accumulators of X'_{j,c} /
// X'_{j,c+1}, D2 = the data operand (reloaded per term per c), D3 = coefficient tile (A_j, then
// B_j), then ss. Same per-element SFPMUL / SFPMAD sequence as mhc_mix_pass1..3.
// D0 = A0*D0 [; D1 = A0*D1]
template <bool TWO>
inline void mhc_r_mul() {
    mhc_lm::configure();
    mhc_lm::sub_mul<0, mhc_lm::Q0>();
    if constexpr (TWO) {
        mhc_lm::sub_mul<1, mhc_lm::Q0>();
    }
    mhc_lm::drain_end();
}
// D[ACC] = A[QC]*D2 + D[ACC]
template <uint32_t ACC, uint32_t QC>
inline void mhc_r_mac() {
    mhc_lm::configure();
    mhc_lm::sub_mac<ACC, 2, QC>();
    mhc_lm::drain_end();
}
// D[ACC] = D3*D2 + D[ACC]   (D3 = B_j lane-aligned)
template <uint32_t ACC>
inline void mhc_r_lane() {
    mhc_lm::configure();
    mhc_lm::sub_lane<ACC, 2>();
    mhc_lm::drain_end();
}
// D3 = D0*D0 + D3 [; D3 = D1*D1 + D3]   (D3 = ss)
template <bool TWO>
inline void mhc_r_sq() {
    mhc_lm::configure();
    mhc_lm::sub_sq<3, 0>();
    if constexpr (TWO) {
        mhc_lm::sub_sq<3, 1>();
    }
    mhc_lm::drain_end();
}

// Perf 2 (tail_hc): the combine's running add D0 <- 1.0 * D1 + D0 as a load-macro stream (sub_mac
// with the coefficient register pinned to 1.0): fma(1, x, acc) == round(acc + x), the SFPU add of
// add_binary_tile, so bit-identical; ~2 issue slots per vector instead of load, load, add, store.
namespace mhc_lm {
inline void add_d1_impl() {
    mhc_lm::configure();
    TTI_SFPLOADI(mhc_lm::LC, sfpi::SFPLOADI_MOD0_FLOATB, 0x3F80);  // 1.0
    mhc_lm::walk<10>([] {
        TTI_SFPNOP;
        MHC_LM(mhc_lm::M_MAC, mhc_lm::VD0, mhc_lm::imm(0, mhc_lm::Q0));
        MHC_LD(mhc_lm::LX, mhc_lm::imm(1, mhc_lm::Q0));
        MHC_LM(mhc_lm::M_MAC, mhc_lm::VD1, mhc_lm::imm(0, mhc_lm::Q1));
        MHC_LD(mhc_lm::LX, mhc_lm::imm(1, mhc_lm::Q1));
        MHC_LM(mhc_lm::M_MAC, mhc_lm::VD2, mhc_lm::imm(0, mhc_lm::Q2));
        MHC_LD(mhc_lm::LX, mhc_lm::imm(1, mhc_lm::Q2));
        MHC_LM(mhc_lm::M_MAC, mhc_lm::VD3, mhc_lm::imm(0, mhc_lm::Q3));
        MHC_LD_LAST(mhc_lm::LX, mhc_lm::imm(1, mhc_lm::Q3));
        TTI_SFPNOP;
    });
    mhc_lm::drain_end();
}
}  // namespace mhc_lm
inline void mhc_lm_add_d1() { mhc_lm::add_d1_impl(); }

#undef MHC_LM
#undef MHC_LD
#undef MHC_AM
#undef MHC_LM_LAST
#undef MHC_LD_LAST
