// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"

// Replay-buffer usage on Quasar follows a record-then-replay pattern, distinct
// from the BH reference which records and executes in one pass via
// `load_replay_buf<X, Y, true>` (BH `Exec=1`, i.e. `execute_while_loading=true`).
// Quasar errata TEN-4690 forbids `TTI_REPLAY` with `execute_while_loading=true`,
// so every `load_replay_buf<X, Y, ...>` call in this file passes
// `exec_while_loading=false` (record only) and is followed by an explicit
// `TTI_REPLAY(X, Y, ...)` to execute the recorded body.

// NOTE: The STABLE_SORT=true template specializations below (the doubled-SFPSWAP
// `bitonic_topk_ph*_st*_to_1<true>` overloads, etc.) have NOT been validated on
// Quasar end-to-end. The Python test currently `pytest.skip`s all stable_sort=True
// variants pending tt-metal#33492 (LLK API regression). Treat these code paths
// as unverified-on-Quasar until that issue is resolved and the skip is removed.

namespace ckernel
{
namespace sfpu
{

// SFPSWAP modes (mode-to-int mapping is the same as the Blackhole reference).
// Defined locally because Quasar's ckernel_instr_params.h does not provide p_sfpswap.
struct p_sfpswap
{
    constexpr static std::uint32_t UNCONDITIONALLY = 0;
    constexpr static std::uint32_t ALL_ROWS_MAX    = 1;
    constexpr static std::uint32_t ROWS_01_MAX     = 2;
    constexpr static std::uint32_t ROWS_02_MAX     = 3;
    constexpr static std::uint32_t ROWS_03_MAX     = 4;
    constexpr static std::uint32_t ROW_0_MAX       = 5;
    constexpr static std::uint32_t ROW_1_MAX       = 6;
    constexpr static std::uint32_t ROW_2_MAX       = 7;
    constexpr static std::uint32_t ROW_3_MAX       = 8;
};

// Sort direction for topk. Defined locally — Quasar's llk_defs.h does not define SortDir.
enum SortDir : bool
{
    ArgMax = false,
    ArgMin = true,
};

// Set the per-TRISC dest section base register for the math TRISC.
// Quasar has separate SEC0..SEC3 registers (one per TRISC); SFPLOAD/SFPSTORE on the
// math TRISC compute their effective dest address as SEC1 + dest_counter + dest_reg_addr.
inline void set_dst_write_addr(std::uint32_t addr)
{
    std::uint32_t dst_index = addr + ckernel::trisc::_get_dest_buffer_base_();
    ckernel::trisc::_set_dest_section_base_<ckernel::math::TRISC_ID>(dst_index);
}

// Advance the dest RWC counter by `inc` rows in groups of 8.
// `cr=true` issues an additional carriage-return bit to clear the column counter.
// Quasar TTI_INCRWC arg order is (rwc_cr, rwc_a, rwc_b, rwc_d) — dest is the 4th argument.
inline void bitonic_topk_inc_x8_dest(std::uint32_t inc, bool cr)
{
    std::uint32_t inc_grp8 = inc >> 3;
    if (cr)
    {
        for (std::uint32_t i = 0; i < inc_grp8; i++)
        {
            TTI_INCRWC(0b100, 0, 0, 8);
        }
    }
    else
    {
        for (std::uint32_t i = 0; i < inc_grp8; i++)
        {
            TTI_INCRWC(0, 0, 0, 8);
        }
    }
}

inline void _init_topk()
{
    // Write 0x4 to LaneConfig (config_dest=0xF) to set bit [2] = ENABLE_DEST_INDEX.
    // With this bit set, SFPSWAP performs argmin/argmax: when it conditionally
    // swaps LREG[VC] <-> LREG[VD], it also swaps LREG[4 + (VC&3)] <-> LREG[4 + (VD&3)]
    // in lockstep — letting topk track input indices alongside the values being sorted.
    ckernel::math::_sfpu_load_config32_(0xF, 0x0, 0x4);
}

// Load 8 lanes (one value LREG pair + one index LREG pair) from Dest at runtime offsets.
// Values land in LREG0,1; indices (offset by dst_indices_offset = 128) land in LREG4,5.
// Index format mode is INT32 when dest_acc=fp32, FP16B otherwise (workaround for
// UINT16/LO16 dest-cell layout mismatch on Quasar — indices 0..127 are exactly
// representable in Float16_b so we use the upper-half FP16B path).
template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_load8(std::uint32_t offset, std::uint32_t dist)
{
    constexpr std::uint32_t dst_indices_offset = 128;
    // Workaround: Use FP16B (bfloat16) mode for indices instead of UINT16 (LO16, mode 0b0110).
    // Indices 0..127 are exactly representable in Float16_b. The UINT16/LO16 path has a
    // dest-cell layout mismatch on Quasar (unpacker writes upper half via Float16_b A2D
    // datacopy, but SFPU LO16 reads the lower half). FP16B reads/writes the upper half,
    // which matches what the unpacker deposits.
    constexpr std::uint32_t instr_mod_index = is_fp32_dest_acc_en ? p_sfpu::sfpmem::INT32 : p_sfpu::sfpmem::FP16B;

    std::uint32_t face_offset = offset >> 4;
    std::uint32_t ld_offset   = (offset & 0xF) + face_offset * 32;

    // Values
    TT_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 0, ld_offset);
    TT_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, 0, ld_offset + dist);

    // Indices (paired with LREG0,1; shifted by dst_indices_offset).
    TT_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + ld_offset);
    TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + ld_offset + dist);
}

// Store 8 lanes (the same LREGs that bitonic_topk_load8 fills) back into Dest. Mirrors load8.
template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_store8(std::uint32_t offset, std::uint32_t dist)
{
    constexpr std::uint32_t dst_indices_offset = 128;
    // FP16B index mode (see bitonic_topk_load8 for rationale).
    constexpr std::uint32_t instr_mod_index = is_fp32_dest_acc_en ? p_sfpu::sfpmem::INT32 : p_sfpu::sfpmem::FP16B;

    std::uint32_t face_offset = offset >> 4;
    std::uint32_t ld_offset   = (offset & 0xF) + face_offset * 32;

    // Values
    TT_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, 0, ld_offset);
    TT_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, 0, ld_offset + dist);

    // Indices
    TT_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + ld_offset);
    TT_SFPSTORE(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + ld_offset + dist);
}

// Load 16 lanes (LREG0..3 values + LREG4..7 indices) from Dest at strided offsets
// (0, dist0, dist1, dist1+dist0). The (dist0,dist1)==(4,8) call site is the hot path
// used by phases 5/6 — its addresses are constexpr so we hand them to TTI_SFPLOAD.
template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_load16(std::uint32_t dist0, std::uint32_t dist1)
{
    constexpr std::uint32_t dst_indices_offset = 128;
    // FP16B index mode (see bitonic_topk_load8 for rationale).
    constexpr std::uint32_t instr_mod_index = is_fp32_dest_acc_en ? p_sfpu::sfpmem::INT32 : p_sfpu::sfpmem::FP16B;

    // Values
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 0, 0);
    if ((dist0 == 4) && (dist1 == 8))
    {
        TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, 0, 4);
        TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_7, 0, 8);
        TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_7, 0, 12);
    }
    else
    {
        TT_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, 0, dist0);
        TT_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_7, 0, dist1);
        TT_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_7, 0, dist1 + dist0);
    }

    // Indices (paired with LREG0..3; shifted by dst_indices_offset).
    TTI_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + 0);
    if ((dist0 == 4) && (dist1 == 8))
    {
        TTI_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + 4);
        TTI_SFPLOAD(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + 8);
        TTI_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + 12);
    }
    else
    {
        TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + dist0);
        TT_SFPLOAD(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + dist1);
        TT_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + dist1 + dist0);
    }
}

// Store 16 lanes (the same LREGs that bitonic_topk_load16 fills) back into Dest.
// When alt_addr_mod=true, the FINAL index store (LREG7) uses ADDR_MOD_6 instead of
// ADDR_MOD_7 — phase 6 configures ADDR_MOD_6 with dest.incr=32 so this auto-advances
// Dest by 32 rows after the last store of a 16-element block.
template <bool is_fp32_dest_acc_en, bool alt_addr_mod = false>
inline void bitonic_topk_store16(std::uint32_t dist0, std::uint32_t dist1)
{
    constexpr std::uint32_t dst_indices_offset = 128;
    // FP16B index mode (see bitonic_topk_load8 for rationale).
    constexpr std::uint32_t instr_mod_index = is_fp32_dest_acc_en ? p_sfpu::sfpmem::INT32 : p_sfpu::sfpmem::FP16B;

    // Values
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, 0, 0);
    if ((dist0 == 4) && (dist1 == 8))
    {
        TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, 0, 4);
        TTI_SFPSTORE(p_sfpu::LREG2, 0, ADDR_MOD_7, 0, 8);
        TTI_SFPSTORE(p_sfpu::LREG3, 0, ADDR_MOD_7, 0, 12);
    }
    else
    {
        TT_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, 0, dist0);
        TT_SFPSTORE(p_sfpu::LREG2, 0, ADDR_MOD_7, 0, dist1);
        TT_SFPSTORE(p_sfpu::LREG3, 0, ADDR_MOD_7, 0, dist1 + dist0);
    }

    // Indices — last store optionally swaps to ADDR_MOD_6 for the auto-advance.
    TTI_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + 0);
    if ((dist0 == 4) && (dist1 == 8))
    {
        TTI_SFPSTORE(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + 4);
        TTI_SFPSTORE(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + 8);
        TTI_SFPSTORE(p_sfpu::LREG7, instr_mod_index, alt_addr_mod ? ADDR_MOD_6 : ADDR_MOD_7, 0, dst_indices_offset + 12);
    }
    else
    {
        TT_SFPSTORE(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + dist0);
        TT_SFPSTORE(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + dist1);
        TT_SFPSTORE(p_sfpu::LREG7, instr_mod_index, alt_addr_mod ? ADDR_MOD_6 : ADDR_MOD_7, 0, dst_indices_offset + dist1 + dist0);
    }
}

// Phase 0, step 1 sort building block. Wrapped between two SFPTRANSPs so the swap layer
// operates across the post-transpose lane layout. STABLE_SORT=true variant duplicates each
// SFPSWAP pair (4 swaps interleaved on disjoint LREGs) for stable index tie-breaking.
template <bool STABLE_SORT>
inline void bitonic_topk_ph0_st1_to_1();

template <>
inline void bitonic_topk_ph0_st1_to_1<true>()
{
    TTI_SFPTRANSP;

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX); // Hides LREG0/1 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX); // Hides LREG2/3 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX); // Hides LREG0/1 NOP

    TTI_SFPTRANSP;
}

template <>
inline void bitonic_topk_ph0_st1_to_1<false>()
{
    TTI_SFPTRANSP;

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);

    TTI_SFPTRANSP;
}

// Phase 1, steps 2 then 1. Wrapped between two SFPTRANSPs.
// STABLE_SORT=true: 4 swaps per step (interleaved on disjoint LREGs); 1-cycle stall
// between step 2 and step 1 because they share LREG1.
template <bool STABLE_SORT>
inline void bitonic_topk_ph1_st2_to_1();

template <>
inline void bitonic_topk_ph1_st2_to_1<true>()
{
    TTI_SFPTRANSP;

    // Step 2
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX); // Hides LREG0/2 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_02_MAX); // Hides LREG1/3 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX); // Hides LREG0/2 NOP

    // Step 1 (1-cycle stall: shares LREG1 with Step 2 above)
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX); // Hides LREG0/1 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_02_MAX); // Hides LREG2/3 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX); // Hides LREG0/1 NOP

    TTI_SFPTRANSP;
}

template <>
inline void bitonic_topk_ph1_st2_to_1<false>()
{
    TTI_SFPTRANSP;

    // Step 2
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX);

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX);

    TTI_SFPTRANSP;
}

// Phase 2, steps 3, 2, then 1. Step 3 runs on the natural lane layout (no leading TRANSP),
// then a single TRANSP separates it from steps 2/1. STABLE_SORT=false adds an unconditional
// SFPSWAP(LREG2, LREG3) after step 3 — a deliberate post-step-3 reorder copied from the
// Blackhole reference (matches the BH algorithm exactly; do not remove).
template <bool STABLE_SORT>
inline void bitonic_topk_ph2_st3_to_1();

template <>
inline void bitonic_topk_ph2_st3_to_1<true>()
{
    // Step 3
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX); // Hides LREG0/1 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX); // Hides LREG2/3 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX); // Hides LREG0/1 NOP

    TTI_SFPTRANSP;

    // Step 2
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_01_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX); // Hides LREG0/2 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_01_MAX); // Hides LREG1/3 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX); // Hides LREG0/2 NOP

    // Step 1 (1-cycle stall: shares LREG1 with Step 2 above)
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_01_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX); // Hides LREG0/1 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_01_MAX); // Hides LREG2/3 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX); // Hides LREG0/1 NOP

    TTI_SFPTRANSP;
}

template <>
inline void bitonic_topk_ph2_st3_to_1<false>()
{
    // Step 3
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::UNCONDITIONALLY);

    TTI_SFPTRANSP;

    // Step 2
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_01_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX);

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_01_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX);

    TTI_SFPTRANSP;
}

// Step-N inner-most layer used by the outer-loop in _bitonic_topk_phases_steps for the
// largest-stride compares. No SFPTRANSP — operates directly on LREG0..LREG3.
// dir==ArgMax: SFPSWAP(LREG0,LREG2) + SFPSWAP(LREG1,LREG3).
// dir==ArgMin: SFPSWAP arg order flipped (LREG2,LREG0) + (LREG3,LREG1) — comparison-direction
// inversion that's distinct from the SFPCONFIG bit-8 EXCHANGE_SRCB_SRCC mechanism phase 4 uses.
// CALLER RESPONSIBILITY: there is no trailing SFPTRANSP, so the caller must follow this with
// either a different-LREG SFPSWAP, an SFPTRANSP, or an explicit TTI_SFPNOP(0,0,0) before any
// SFPSTORE that consumes LREG0..LREG3 (avoids the SFPSWAP→SFPSTORE auto-stall hardware bug).
template <bool STABLE_SORT>
inline void bitonic_topk_step_N(bool dir);

template <>
inline void bitonic_topk_step_N<true>(bool dir)
{
    // Step N
    if (dir == static_cast<bool>(SortDir::ArgMax))
    {
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX); // Hides LREG0/2 NOP
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX); // Hides LREG1/3 NOP
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX); // Hides LREG0/2 NOP
    }
    else
    {
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX); // Hides LREG2/0 NOP
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX); // Hides LREG3/1 NOP
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX); // Hides LREG2/0 NOP
    }
}

template <>
inline void bitonic_topk_step_N<false>(bool dir)
{
    // Step N
    if (dir == static_cast<bool>(SortDir::ArgMax))
    {
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    }
    else
    {
        // Min — operand order swapped relative to ArgMax.
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    }
}

// Record the phase-3 step-4-to-1 body into the replay buffer at slot
// [replay_start, replay_start+replay_count). Call this ONCE before the loop
// that invokes `bitonic_topk_ph3_st4_to_1<STABLE_SORT, replay_start>(dir)`.
// `replay_count` = STABLE_SORT ? 9 : 5.
template <bool STABLE_SORT, int replay_start>
inline void load_bitonic_topk_ph3_st4_to_1_replay()
{
    constexpr int replay_count = STABLE_SORT ? 9 : 5;
    if constexpr (STABLE_SORT)
    {
        load_replay_buf<replay_start, replay_count, false>(
            []
            {
                // Step 4 — 4 interleaved SFPSWAPs on disjoint pairs (0,2)/(1,3).
                TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
                TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
                TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
                TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

                // Step 3 — 4 interleaved SFPSWAPs on disjoint pairs (2,3)/(0,1).
                // 1-cycle stall vs Step 4's tail because they share LREG3.
                TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
                TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
                TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
                TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);

                TTI_SFPTRANSP;
            });
    }
    else
    {
        load_replay_buf<replay_start, replay_count, false>(
            []
            {
                // Step 4
                TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
                TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

                // Step 3
                TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
                TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

                TTI_SFPTRANSP;
            });
    }
}

// Phase 3 step-4-to-1: the building block for the largest-stride bitonic compare layer.
// Performs two passes of "step 4 then step 3" SFPSWAPs followed by an SFPTRANSP. The
// double execution implements the full step-4-to-1 sweep.
//
// dir==ArgMin temporarily flips LaneConfig bit [8] (EXCHANGE_SRCB_SRCC) via SFPCONFIG so
// SFPSWAP compares with reversed min/max polarity; bit [8] is cleared again before return.
// Both 0x104 (set) and 0x004 (clear) keep bit [2] (ENABLE_DEST_INDEX) so the index pairing
// invariant established by _init_topk() is preserved across calls.
//
// `replay_start` is a template int because TTI_REPLAY's start operand has a "i" inline-asm
// constraint (compile-time immediate). Phase 5 callers pass 16; phase 6 callers pass 8.
// `replay_count` = STABLE_SORT ? 9 : 5.
//
// Caller MUST call `load_bitonic_topk_ph3_st4_to_1_replay<STABLE_SORT, replay_start>()`
// once before invoking this in a loop.
template <bool STABLE_SORT, int replay_start>
inline void bitonic_topk_ph3_st4_to_1(bool dir)
{
    constexpr int replay_count = STABLE_SORT ? 9 : 5;
    if (dir == static_cast<bool>(SortDir::ArgMin))
    {
        TTI_SFPCONFIG(0x104, 0xF, 1); // Reverse the max/min behaviour of SWAP
        // SFPCONFIG is a 2-cycle op; per Quasar errata TEN-4581 ("any 2-cycle op
        // followed by SFPSWAP") at least 1 SFPNOP must separate it from the next
        // SFPSWAP. The recorded replay body starts with SFPSWAPs, so insert the NOP
        // before TTI_REPLAY (which is itself the SFPSWAP issue point).
        TTI_SFPNOP(0, 0, 0);
    }
    // Both executions of the buffered body. Caller pre-recorded the slot via
    // load_bitonic_topk_ph3_st4_to_1_replay<STABLE_SORT, replay_start>().
    TTI_REPLAY(replay_start, replay_count, 0, 0, 0, 0);
    TTI_REPLAY(replay_start, replay_count, 0, 0, 0, 0);

    if (dir == static_cast<bool>(SortDir::ArgMin))
    {
        TTI_SFPCONFIG(0x004, 0xF, 1); // Restore the max/min behaviour of SWAP
        // See above — SFPCONFIG → next SFPU op needs an SFPNOP per TEN-4581.
        TTI_SFPNOP(0, 0, 0);
    }
}

// Top-level local-sort orchestrator. For each (face, col) sub-region of the dest tile,
// walks phases [i_start_phase, i_end_phase]; phases 0..3 each emit 4 groups of
// [load16, swap-body, store16] sequences; phase 4+ falls back to inline step-N..5
// loops followed by phase-3 replay for steps 4..1.
//
// Replay-buffer slot allocation:
//   [0,  8)  = bitonic_topk_load16<is_fp32>(4, 8) body  (4 value loads + 4 index loads)
//   [8, 16)  = bitonic_topk_store16<is_fp32, alt_addr_mod=true>(4, 8) body
//   [16, 16+replay_count) = per-phase swap body (re-recorded at start of each new phase)
// Maximum slot used is 29 (stable phase 2). Quasar's replay buffer is 32 deep.
//
// All recordings happen up front: the [0, 16) static slots at function entry, and the
// [16, …) phase-body slot at the top of each phase-loop iteration. Inner loops then
// only emit TTI_REPLAY, matching the canonical Quasar load/replay split (TEN-4690
// forbids `execute_while_loading=true`, so record-then-replay is the only option).
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool STABLE_SORT = false>
inline void _bitonic_topk_phases_steps(const int idir, const int i_end_phase, const int i_start_phase, const int i_end_step, const int i_start_step)
{
    // Replay setup: record the static slots (load16 at [0, 8), store16 at [8, 16))
    // once at function entry. These bodies don't depend on dir/ph/face/col so a
    // single recording is reused across the entire function.
    load_replay_buf<0, 8, false>([] { bitonic_topk_load16<is_fp32_dest_acc_en>(4, 8); });
    load_replay_buf<8, 8, false>([] { bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, 8); });

    // Loop nest: ph is the OUTER loop so the per-phase recording at slot 16+ happens
    // ONCE per phase instead of 4× (once per face × col). The (face, col) sweep is
    // nested inside and pure-replay.
    for (int ph = i_start_phase; ph < (i_end_phase + 1); ph++)
    {
        // Record the per-phase body into slot 16+ once for this phase.
        switch (ph)
        {
            case 0:
                load_replay_buf<16, STABLE_SORT ? 6 : 4, false>([] { bitonic_topk_ph0_st1_to_1<STABLE_SORT>(); });
                break;
            case 1:
                load_replay_buf<16, STABLE_SORT ? 10 : 6, false>([] { bitonic_topk_ph1_st2_to_1<STABLE_SORT>(); });
                break;
            case 2:
                load_replay_buf<16, STABLE_SORT ? 14 : 9, false>([] { bitonic_topk_ph2_st3_to_1<STABLE_SORT>(); });
                break;
            case 3:
            default:
                // case 3 uses the ph3 helper directly; default's Part 2 (steps 4..1) also uses it.
                load_bitonic_topk_ph3_st4_to_1_replay<STABLE_SORT, 16>();
                break;
        }

        // Reset dest pointer to 0 (caller's initial state). The (face, col) sweep below
        // walks dest offsets 0, 2, 16, 18 — same as a fresh function entry. Required
        // for every phase except the first because the previous phase's sweep left
        // dest at 16.
        set_dst_write_addr(0);

        std::uint32_t dst_addr_offset = 0;
        for (int face = 0; face < 2; face++)
        {
            for (int col = 0; col < 2; col++)
            {
                bool dir = idir;
                TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, p_setrwc::SET_D);
                switch (ph)
                {
                    case 0:
                    {
                        constexpr int replay_count = STABLE_SORT ? 6 : 4;
                        for (int d = 0; d < 4; d++)
                        {
                            TTI_REPLAY(0, 8, 0, 0, 0, 0);
                            TTI_REPLAY(16, replay_count, 0, 0, 0, 0);
                            TTI_REPLAY(8, 8, 0, 0, 0, 0);
                        }
                        break;
                    }

                    case 1:
                    {
                        constexpr int replay_count = STABLE_SORT ? 10 : 6;
                        for (int d = 0; d < 4; d++)
                        {
                            TTI_REPLAY(0, 8, 0, 0, 0, 0);
                            TTI_REPLAY(16, replay_count, 0, 0, 0, 0);
                            TTI_REPLAY(8, 8, 0, 0, 0, 0);
                        }
                        break;
                    }

                    case 2:
                    {
                        constexpr int replay_count = STABLE_SORT ? 14 : 9;
                        for (int d = 0; d < 4; d++)
                        {
                            TTI_REPLAY(0, 8, 0, 0, 0, 0);
                            TTI_REPLAY(16, replay_count, 0, 0, 0, 0);
                            TTI_REPLAY(8, 8, 0, 0, 0, 0);
                        }
                        break;
                    }

                    case 3:
                    {
                        for (int d = 0; d < 4; d++)
                        {
                            TTI_REPLAY(0, 8, 0, 0, 0, 0);
                            bitonic_topk_ph3_st4_to_1<STABLE_SORT, 16>(dir);
                            TTI_REPLAY(8, 8, 0, 0, 0, 0);
                            dir = !dir;
                        }
                        break;
                    }

                    default:
                    {
                        // Phases 4..N: steps `num_steps`..5 are emitted inline (no replay);
                        // steps 4..1 fall back to the same phase-4 helper as case 3.
                        std::uint32_t num_steps               = ph + 1;
                        std::uint32_t start_step              = (i_start_phase == i_end_phase) ? i_start_step : num_steps;
                        std::uint32_t end_step                = (i_start_phase == i_end_phase) ? i_end_step : 4;
                        std::uint32_t sorted_seq_length       = 1 << num_steps;
                        std::uint32_t datums_compared         = 0;
                        std::uint32_t total_datums_to_compare = 64;

                        for (std::uint32_t ss = start_step; ss > end_step; ss--)
                        {
                            // Steps N..5 (inline)
                            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, p_setrwc::SET_D);
                            dir                      = idir;
                            std::uint32_t dist       = (ss == 5) ? 16 : 32;
                            std::uint32_t inner_d    = dist >> 3;
                            datums_compared          = 0;
                            std::uint32_t dst_offset = 0;

                            while (datums_compared < total_datums_to_compare)
                            {
                                for (std::uint32_t ii = 0; ii < inner_d; ii++)
                                {
                                    bitonic_topk_load16<is_fp32_dest_acc_en>(4, 2 * dist);
                                    bitonic_topk_step_N<STABLE_SORT>(dir);
                                    bitonic_topk_store16<is_fp32_dest_acc_en, false>(4, 2 * dist);

                                    std::uint32_t dst_inc = 8;
                                    dst_offset += dst_inc;
                                    bool dst_cr = false;
                                    if (ii == (inner_d - 1))
                                    {
                                        dst_cr     = true;
                                        dst_inc    = 4 * dist;
                                        dst_offset = 2 * dist;
                                    }
                                    else if (dst_offset == 16)
                                    {
                                        dst_cr  = true;
                                        dst_inc = 32;
                                    }
                                    bitonic_topk_inc_x8_dest(dst_inc, dst_cr);
                                    datums_compared += 16;
                                }
                                dir = (datums_compared == sorted_seq_length) ? !dir : dir;
                            }
                        }

                        // Steps 4..1 (replay - same as case 3, ph3 helper slot pre-loaded above)
                        dir = idir;
                        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, p_setrwc::SET_D);
                        datums_compared = 0;
                        while (datums_compared < total_datums_to_compare)
                        {
                            TTI_REPLAY(0, 8, 0, 0, 0, 0);
                            bitonic_topk_ph3_st4_to_1<STABLE_SORT, 16>(dir);
                            TTI_REPLAY(8, 8, 0, 0, 0, 0);
                            datums_compared += 16;
                            dir = (datums_compared == sorted_seq_length) ? !dir : dir;
                        }
                        break;
                    }
                }
                dst_addr_offset += 2;
                set_dst_write_addr(dst_addr_offset);
            }
            dst_addr_offset = 16;
            set_dst_write_addr(dst_addr_offset);
        }
    }
}

// Thin forwarding wrapper: the test passes this as the `fn` argument to
// _llk_math_eltwise_unary_sfpu_params_, which calls it with the runtime int args.
// ENABLE_DEST_INDEX (LaneConfig bit [2]) is set once by `_init_topk()` and
// LaneConfig persists on Quasar — no per-call re-assertion needed. A previous
// defensive SFPCONFIG re-assertion here clobbered LaneConfig bit [8]
// (EXCHANGE_SRCB_SRCC) that `bitonic_topk_ph3_st4_to_1` sets for the ArgMin
// path, breaking the index swap for multi-tile-row Ascending inputs. Matches BH.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool STABLE_SORT = false>
inline void calculate_bitonic_topk_phases_steps(const int idir, const int i_end_phase, const int i_start_phase, const int i_end_step, const int i_start_step)
{
    _bitonic_topk_phases_steps<APPROXIMATION_MODE, is_fp32_dest_acc_en, STABLE_SORT>(idir, i_end_phase, i_start_phase, i_end_step, i_start_step);
}

// Merge: one bitonic merge pass over already-sorted runs in Dest. For each (face, col)
// sub-region, repeatedly load 8 lanes (LREG0,1 values + LREG4,5 indices) via
// bitonic_topk_load8, run an SFPSWAP ALL_ROWS_MAX whose operand order is selected at
// compile time by `top_min` (false → ArgMax: LREG0, LREG1; true → ArgMin: LREG1, LREG0),
// then store back via bitonic_topk_store8. STABLE_SORT issues a duplicate SFPSWAP on the
// same operands — the value-swap is a no-op but with ENABLE_DEST_INDEX it re-evaluates
// the index conditional swap, breaking ties stably (1-cycle stall on the duplicate is
// intentional and matches the Blackhole reference).
//
// Merge does NOT touch the replay buffer; the inner body is short enough that recording
// would not pay off, and avoiding replay here means rebuild is free to own the slot range.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool top_min, bool STABLE_SORT = false>
inline void _bitonic_topk_merge(const int m_iter, const int k)
{
    std::uint32_t dst_addr_offset = 0;
    for (int face = 0; face < 2; face++)
    {
        for (int col = 0; col < 2; col++)
        {
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, p_setrwc::SET_D);

            int k_max                             = k > 32 ? 32 : k;
            std::uint32_t inner_d                 = k_max >> 2; // inner-loop comparisons to sort a length-K sequence
            std::uint32_t total_datums_to_compare = ((64 >> m_iter) < 2 * k_max) ? 2 * k_max : (64 >> m_iter);
            std::uint32_t dist                    = (k_max << m_iter) > 32 ? 32 : (k_max << m_iter);
            std::uint32_t ld_dist                 = (dist < 16) ? dist : 2 * dist; // accounts for face offsets within a tile
            std::uint32_t datums_compared         = 0;
            std::uint32_t dst_offset              = 0;
            std::uint32_t dst_cr                  = 0;

            while (datums_compared < total_datums_to_compare)
            {
                for (std::uint32_t ii = 0; ii < inner_d; ii++)
                {
                    bitonic_topk_load8<is_fp32_dest_acc_en>(dst_offset, ld_dist);
                    TTI_SFPSWAP(0, top_min ? p_sfpu::LREG1 : p_sfpu::LREG0, top_min ? p_sfpu::LREG0 : p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
                    if constexpr (STABLE_SORT)
                    {
                        // Duplicate swap on identical operands: with ENABLE_DEST_INDEX
                        // the value-swap is a no-op but the index conditional swap
                        // re-evaluates for stable tie-break (1-cycle stall is intentional).
                        TTI_SFPSWAP(0, top_min ? p_sfpu::LREG1 : p_sfpu::LREG0, top_min ? p_sfpu::LREG0 : p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
                    }
                    bitonic_topk_store8<is_fp32_dest_acc_en>(dst_offset, ld_dist);

                    datums_compared += 8;
                    if (ii == (inner_d - 1))
                    {
                        dst_cr += 2 * dist;
                        dst_offset = dst_cr;
                    }
                    else
                    {
                        dst_offset += 4;
                    }
                }
            }
            dst_addr_offset += 2;
            set_dst_write_addr(dst_addr_offset);
        }
        dst_addr_offset = 16;
        set_dst_write_addr(dst_addr_offset);
    }
}

// Thin forwarding wrapper: matches the call signature of
// `_llk_math_eltwise_unary_sfpu_params_`. `idir` is named to mirror the BH metal-side
// wrapper's third template parameter (the sort-direction bit forwarded as `top_min`).
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool idir = false, bool STABLE_SORT = false>
inline void calculate_bitonic_topk_merge(const int m_iter, const int k)
{
    // SFPCONFIG re-assertion removed — see comment on calculate_bitonic_topk_phases_steps.
    _bitonic_topk_merge<APPROXIMATION_MODE, is_fp32_dest_acc_en, idir, STABLE_SORT>(m_iter, k);
}

// Rebuild: re-runs phase (logK-1) on a merged tile pair to reextract sorted runs.
// A switch on `(logk - 1)` selects one of cases 0/1/2/3/default; each emits a different
// load/swap-sequence/store pattern. For non-stable sort, most cases are wrapped in a
// replay buffer with INCRWC counter advances embedded in the lambda body.
//
// Replay-slot allocation:
//   case 1 m_iter>=2 (non-stable) : slots [0, 22) — load8 + ph1_st2_to_1 + store8 + 8 INCRWC
//   case 1 m_iter<2  (non-stable) : slots [0, 26) — load16 + ph1_st2_to_1 + store16 + 4 INCRWC
//   case 2          (non-stable)  : slots [0, 29) — load16 + ph2_st3_to_1 + store16 + 4 INCRWC
//   case 3          (non-stable)  : slots [0,  8) load16, [8, 13) ph3 helper, [13, 25) store16 + 4 INCRWC
//   default Part 2  (non-stable)  : slots [0,  8) load16, [8, 13) ph3 helper, [17, 25) store16
//
// All recordings happen at the top of each (face, col) iteration, before the inner
// `while (datums_compared < total_datums_to_compare)` loop. The loop is pure
// TTI_REPLAY (and a call to the pre-loaded `bitonic_topk_ph3_st4_to_1` helper for
// the case-3 and default branches). This matches the canonical Quasar load/replay
// split — TEN-4690 forbids `execute_while_loading=true`, so record-then-replay is
// the only option.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool STABLE_SORT = false>
inline void _bitonic_topk_rebuild(const bool idir, const int m_iter, const int k, const int logk, const int skip_second)
{
    // Per-call constants (don't depend on face/col) — hoisted out of the (face, col) sweep.
    const std::uint32_t total_datums_shift = (skip_second & 0x1);
    const std::uint32_t rebuild_m          = m_iter + 1;
    std::uint32_t total_datums_to_compare  = ((64 >> rebuild_m) < 2 * k) ? 2 * k : (64 >> rebuild_m);
    total_datums_to_compare >>= total_datums_shift;
    const std::uint32_t dist      = (k << rebuild_m) > 32 ? 32 : (k << rebuild_m);
    const std::uint32_t ld_offset = (dist >> 4) * 32 + (dist & 0xF);
    std::uint32_t ld_dist         = 0;
    const int ph                  = logk - 1;

    // Per-call replay-buffer recordings (depend on ph; non-stable cases only). The
    // (face, col) sweep below is then pure execute.
    switch (ph)
    {
        case 0:
        case 1:
            if (m_iter < 2)
            {
                ld_dist = (ld_offset < 16) ? 4 * ld_offset : 2 * ld_offset;
            }
            if constexpr (!STABLE_SORT)
            {
                if (m_iter >= 2)
                {
                    load_replay_buf<0, 22, false>(
                        [ld_offset]
                        {
                            bitonic_topk_load8<is_fp32_dest_acc_en>(0, ld_offset);
                            bitonic_topk_ph1_st2_to_1<STABLE_SORT>();
                            bitonic_topk_store8<is_fp32_dest_acc_en>(0, ld_offset);
                            bitonic_topk_inc_x8_dest(64, false);
                        });
                }
                else if (ph == 1)
                {
                    load_replay_buf<0, 26, false>(
                        [ld_offset, ld_dist]
                        {
                            bitonic_topk_load16<is_fp32_dest_acc_en>(ld_offset, ld_dist);
                            bitonic_topk_ph1_st2_to_1<STABLE_SORT>();
                            bitonic_topk_store16<is_fp32_dest_acc_en, true>(ld_offset, ld_dist);
                            TTI_INCRWC(0, 0, 0, 8);
                            TTI_INCRWC(0, 0, 0, 8);
                            TTI_INCRWC(0, 0, 0, 8);
                            TTI_INCRWC(0, 0, 0, 8);
                        });
                }
            }
            break;
        case 2:
            if constexpr (!STABLE_SORT)
            {
                load_replay_buf<0, 29, false>(
                    [ld_offset]
                    {
                        bitonic_topk_load16<is_fp32_dest_acc_en>(4, ld_offset);
                        bitonic_topk_ph2_st3_to_1<STABLE_SORT>();
                        bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, ld_offset);
                        TTI_INCRWC(0, 0, 0, 8);
                        TTI_INCRWC(0, 0, 0, 8);
                        TTI_INCRWC(0, 0, 0, 8);
                        TTI_INCRWC(0, 0, 0, 8);
                    });
            }
            break;
        case 3:
            load_bitonic_topk_ph3_st4_to_1_replay<STABLE_SORT, 8>();
            if constexpr (!STABLE_SORT)
            {
                // Three slots: [0, 8) load16, [8, 13) ph3 body (above), [13, 25) store16+4×INCRWC.
                load_replay_buf<0, 8, false>([] { bitonic_topk_load16<is_fp32_dest_acc_en>(4, 8); });
                load_replay_buf<13, 12, false>(
                    []
                    {
                        bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, 8);
                        TTI_INCRWC(0, 0, 0, 8);
                        TTI_INCRWC(0, 0, 0, 8);
                        TTI_INCRWC(0, 0, 0, 8);
                        TTI_INCRWC(0, 0, 0, 8);
                    });
            }
            break;
        default:
            // ph >= 4: Part 2's replay slots (Part 1 is fully inline, no replay).
            load_replay_buf<0, 8, false>([] { bitonic_topk_load16<is_fp32_dest_acc_en>(4, 8); });
            load_bitonic_topk_ph3_st4_to_1_replay<STABLE_SORT, 8>();
            load_replay_buf<17, 8, false>([] { bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, 8); });
            break;
    }

    // (face, col) sweep — pure execute.
    std::uint32_t dst_addr_offset = 0;
    for (int face = 0; face < 2; face++)
    {
        for (int col = 0; col < 2; col++)
        {
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, p_setrwc::SET_D);
            bool dir                      = idir;
            std::uint32_t datums_compared = 0;

            switch (ph)
            {
                case 0:
                    break;

                case 1:
                    if (m_iter >= 2)
                    {
                        if constexpr (STABLE_SORT)
                        {
                            while (datums_compared < total_datums_to_compare)
                            {
                                bitonic_topk_load8<is_fp32_dest_acc_en>(0, ld_offset);
                                bitonic_topk_ph1_st2_to_1<STABLE_SORT>();
                                bitonic_topk_store8<is_fp32_dest_acc_en>(0, ld_offset);
                                bitonic_topk_inc_x8_dest(64, false);
                                datums_compared += 16;
                            }
                        }
                        else
                        {
                            while (datums_compared < total_datums_to_compare)
                            {
                                TTI_REPLAY(0, 22, 0, 0, 0, 0);
                                datums_compared += 16;
                            }
                        }
                    }
                    else
                    {
                        if constexpr (STABLE_SORT)
                        {
                            while (datums_compared < total_datums_to_compare)
                            {
                                bitonic_topk_load16<is_fp32_dest_acc_en>(ld_offset, ld_dist);
                                bitonic_topk_ph1_st2_to_1<STABLE_SORT>();
                                bitonic_topk_store16<is_fp32_dest_acc_en, true>(ld_offset, ld_dist);
                                TTI_INCRWC(0, 0, 0, 8);
                                TTI_INCRWC(0, 0, 0, 8);
                                TTI_INCRWC(0, 0, 0, 8);
                                TTI_INCRWC(0, 0, 0, 8);
                                datums_compared += 16;
                            }
                        }
                        else
                        {
                            while (datums_compared < total_datums_to_compare)
                            {
                                TTI_REPLAY(0, 26, 0, 0, 0, 0);
                                datums_compared += 16;
                            }
                        }
                    }
                    break;

                case 2:
                    if constexpr (STABLE_SORT)
                    {
                        while (datums_compared < total_datums_to_compare)
                        {
                            bitonic_topk_load16<is_fp32_dest_acc_en>(4, ld_offset);
                            bitonic_topk_ph2_st3_to_1<STABLE_SORT>();
                            bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, ld_offset);
                            TTI_INCRWC(0, 0, 0, 8);
                            TTI_INCRWC(0, 0, 0, 8);
                            TTI_INCRWC(0, 0, 0, 8);
                            TTI_INCRWC(0, 0, 0, 8);
                            datums_compared += 16;
                        }
                    }
                    else
                    {
                        while (datums_compared < total_datums_to_compare)
                        {
                            TTI_REPLAY(0, 29, 0, 0, 0, 0);
                            datums_compared += 16;
                        }
                    }
                    break;

                case 3:
                    if constexpr (STABLE_SORT)
                    {
                        while (datums_compared < total_datums_to_compare)
                        {
                            bitonic_topk_load16<is_fp32_dest_acc_en>(4, 8);
                            bitonic_topk_ph3_st4_to_1<STABLE_SORT, 8>(dir);
                            bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, 8);
                            TTI_INCRWC(0, 0, 0, 8);
                            TTI_INCRWC(0, 0, 0, 8);
                            TTI_INCRWC(0, 0, 0, 8);
                            TTI_INCRWC(0, 0, 0, 8);
                            datums_compared += 16;
                            dir = !dir;
                        }
                    }
                    else
                    {
                        while (datums_compared < total_datums_to_compare)
                        {
                            TTI_REPLAY(0, 8, 0, 0, 0, 0);
                            bitonic_topk_ph3_st4_to_1<STABLE_SORT, 8>(dir);
                            TTI_REPLAY(13, 12, 0, 0, 0, 0);
                            datums_compared += 16;
                            dir = !dir;
                        }
                    }
                    break;

                default:
                {
                    // ph >= 4: two-part sort.
                    // Part 1: steps `num_steps`..5 emitted inline.
                    // Part 2: steps 4..1 via replay (slots pre-loaded above).
                    std::uint32_t num_steps            = ph + 1;
                    std::uint32_t start_step           = num_steps;
                    std::uint32_t end_step             = 4;
                    std::uint32_t sorted_seq_length    = 1 << num_steps;
                    std::uint32_t total_datums_default = 64; // shadows outer; intentional, matches BH

                    for (std::uint32_t ss = start_step; ss > end_step; ss--)
                    {
                        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, p_setrwc::SET_D);
                        dir                      = idir;
                        datums_compared          = 0;
                        std::uint32_t dist_inner = (ss == 5) ? 16 : 32;
                        std::uint32_t inner_d    = dist_inner >> 3;
                        std::uint32_t dst_offset = 0;

                        while (datums_compared < total_datums_default)
                        {
                            for (std::uint32_t ii = 0; ii < inner_d; ii++)
                            {
                                bitonic_topk_load16<is_fp32_dest_acc_en>(4, 2 * dist_inner);
                                bitonic_topk_step_N<STABLE_SORT>(dir);
                                bitonic_topk_store16<is_fp32_dest_acc_en, false>(4, 2 * dist_inner);

                                std::uint32_t dst_inc = 8;
                                dst_offset += dst_inc;
                                bool dst_cr = false;
                                if (ii == (inner_d - 1))
                                {
                                    dst_cr     = true;
                                    dst_inc    = 4 * dist_inner;
                                    dst_offset = 2 * dist_inner;
                                }
                                else if (dst_offset == 16)
                                {
                                    dst_cr  = true;
                                    dst_inc = 32;
                                }
                                bitonic_topk_inc_x8_dest(dst_inc, dst_cr);
                                datums_compared += 16;
                            }
                            dir = (datums_compared == sorted_seq_length) ? !dir : dir;
                        }
                    }

                    // Part 2: steps 4..1 via replay (slots already loaded outside the (face, col) sweep).
                    dir             = idir;
                    datums_compared = 0;
                    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, p_setrwc::SET_D);
                    while (datums_compared < total_datums_default)
                    {
                        TTI_REPLAY(0, 8, 0, 0, 0, 0);
                        bitonic_topk_ph3_st4_to_1<STABLE_SORT, 8>(dir);
                        TTI_REPLAY(17, 8, 0, 0, 0, 0);
                        datums_compared += 16;
                        dir = (datums_compared == sorted_seq_length) ? !dir : dir;
                    }
                    break;
                }
            }
            dst_addr_offset += 2;
            set_dst_write_addr(dst_addr_offset);
        }
        dst_addr_offset = 16;
        set_dst_write_addr(dst_addr_offset);
    }
}

// Thin forwarding wrapper. `idir` is `int` because the variadic
// `_llk_math_eltwise_unary_sfpu_params_` forwards arguments unchanged from the test, where
// it is an `int` (TOPK_SORT_DIRECTION value). The implicit int->bool conversion happens at
// the `_bitonic_topk_rebuild(idir, …)` call site.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool STABLE_SORT = false>
inline void calculate_bitonic_topk_rebuild(const int idir, const int m_iter, const int k, const int logk, const int skip_second)
{
    // SFPCONFIG re-assertion removed — see comment on calculate_bitonic_topk_phases_steps.
    // Removing this re-assertion is the primary fix for the multi-tile-row Ascending
    // failure: the SFPCONFIG was clobbering LaneConfig bit [8] (EXCHANGE_SRCB_SRCC)
    // that `bitonic_topk_ph3_st4_to_1` sets for the ArgMin path.
    _bitonic_topk_rebuild<APPROXIMATION_MODE, is_fp32_dest_acc_en, STABLE_SORT>(idir, m_iter, k, logk, skip_second);
}

} // namespace sfpu
} // namespace ckernel
