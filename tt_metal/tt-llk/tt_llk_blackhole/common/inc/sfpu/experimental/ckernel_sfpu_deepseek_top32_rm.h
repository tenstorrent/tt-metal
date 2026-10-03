// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "lltt.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_load_config.h"

namespace ckernel
{
namespace sfpu
{

// Currently unused: 8-datum load/store variants kept alongside the *16
// versions below for future top32 configurations that process half-width
// strips (e.g. single-LREG-pair sorts); not referenced by any kernel today.
template <bool is_fp32_dest_acc_en>
inline void bitonic_top32_load8(std::uint32_t offset, std::uint32_t dist)
{
    constexpr std::uint32_t dst_indices_offset  = 128; // 2 tile x 64 rows per tile
    constexpr InstrModLoadStore instr_mod_index = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;

    std::uint32_t face_offset = offset >> 4;
    std::uint32_t ld_offset   = (offset & 0xF) + face_offset * 32;

    // Load 16 consecutive numbers
    TT_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, ld_offset);
    TT_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, ld_offset + dist);

    // Load 16 consecutive indices
    TT_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, dst_indices_offset + ld_offset);
    TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, dst_indices_offset + ld_offset + dist);
}

template <bool is_fp32_dest_acc_en>
inline void bitonic_top32_store8(std::uint32_t offset, std::uint32_t dist)
{
    constexpr std::uint32_t dst_indices_offset  = 128; // 2 tile x 64 rows per tile
    constexpr InstrModLoadStore instr_mod_index = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;

    std::uint32_t face_offset = offset >> 4;
    std::uint32_t ld_offset   = (offset & 0xF) + face_offset * 32;

    // Load 16 consecutive numbers
    TT_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, ld_offset);
    TT_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, ld_offset + dist);

    // Load 16 consecutive indices
    TT_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, dst_indices_offset + ld_offset + 0);
    TT_SFPSTORE(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, dst_indices_offset + ld_offset + dist);
}

// Dest layout shared by every kernel below: values in rows [0, 128) of the current tile pair,
// indices a fixed 128 rows (2 tiles x 64 rows) further on. `base` is added to every load/store
// immediate: 0 addresses the even Dest columns, 2 the odd ones. All distances are template
// parameters so every access is an immediate-encoded TTI_ instruction; the kernels used to pass
// them at runtime (TT_SFPLOAD/TT_SFPSTORE through the instruction buffer) and to reach the odd
// columns by rebasing the Dest write pointer with set_dst_write_addr_offset (TT_SETC16).
template <bool is_fp32_dest_acc_en, std::uint32_t dist0, std::uint32_t dist1, std::uint32_t base = 0>
inline void bitonic_top32_load16()
{
    constexpr std::uint32_t dst_indices_offset  = 128; // 2 tile x 64 rows per tile
    constexpr InstrModLoadStore instr_mod_index = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;

    // Load 16 consecutive numbers
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, base + 0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, base + dist0);
    TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_7, base + dist1);
    TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_7, base + dist1 + dist0);

    // Load 16 consecutive indices
    TTI_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, dst_indices_offset + base + 0);
    TTI_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, dst_indices_offset + base + dist0);
    TTI_SFPLOAD(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_7, dst_indices_offset + base + dist1);
    TTI_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_7, dst_indices_offset + base + dist1 + dist0);
}

// alt_addr_mod: the last store uses ADDR_MOD_6 (Dest RWC += 16) to step to the next 16 rows.
template <bool is_fp32_dest_acc_en, bool alt_addr_mod, std::uint32_t dist0, std::uint32_t dist1, std::uint32_t base = 0>
inline void bitonic_top32_store16()
{
    constexpr std::uint32_t dst_indices_offset  = 128; // 2 tile x 64 rows per tile
    constexpr InstrModLoadStore instr_mod_index = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;

    // Store 16 consecutive numbers
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, base + 0);
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, base + dist0);
    TTI_SFPSTORE(p_sfpu::LREG2, 0, ADDR_MOD_7, base + dist1);
    TTI_SFPSTORE(p_sfpu::LREG3, 0, ADDR_MOD_7, base + dist1 + dist0);

    // Store 16 consecutive indices
    TTI_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, dst_indices_offset + base + 0);
    TTI_SFPSTORE(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, dst_indices_offset + base + dist0);
    TTI_SFPSTORE(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_7, dst_indices_offset + base + dist1);
    TTI_SFPSTORE(p_sfpu::LREG7, instr_mod_index, alt_addr_mod ? ADDR_MOD_6 : ADDR_MOD_7, dst_indices_offset + base + dist1 + dist0);
}

// Steps 4..1 of a 16-element bitonic merge, in direction `dir`.
//
// ArgMin used to set LaneConfig.EXCHANGE_SRCB_SRCC with SFPCONFIG(0x104) (+2 SFPNOP) around the
// ALL_ROWS_MAX swaps and restore it with SFPCONFIG(0x004) (+2 SFPNOP). EXCHANGE_SRCB_SRCC turns
// VEC_MIN_MAX's "swap if VC < VD" into "swap if !(VC < VD)", which is exactly the predicate of
// SFPSWAP_MOD1_VEC_MAX_MIN on the same operands, so the ArgMin arm now uses that mode and needs
// no lane-config write. This keeps ties bit-identical, indices included: swapping the operands
// of ALL_ROWS_MAX instead (the step_N idiom below) does not swap equal keys, whereas the
// SFPCONFIG form does.
template <bool dir>
inline void bitonic_top32_ph3_st4_to_1()
{
    constexpr std::uint32_t mode = (dir == static_cast<bool>(SortDir::ArgMin)) ? sfpi::SFPSWAP_MOD1_VEC_MAX_MIN : p_sfpswap::ALL_ROWS_MAX;

    // Step 4
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, mode);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, mode);

    // Step 3
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, mode);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, mode);

    TTI_SFPTRANSP(0, 0, 0, 0);

    // Step 4
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, mode);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, mode);

    // Step 3
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, mode);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, mode);

    TTI_SFPTRANSP(0, 0, 0, 0);
}

inline void bitonic_top32_ph2_st3_to_1()
{
    // Step 3
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::UNCONDITIONALLY);

    TTI_SFPTRANSP(0, 0, 0, 0);

    // Step 2
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_01_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX);

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_01_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX);

    TTI_SFPTRANSP(0, 0, 0, 0);
}

inline void bitonic_top32_ph1_st2_to_1()
{
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Step 2
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX);

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX);

    TTI_SFPTRANSP(0, 0, 0, 0);
}

inline void bitonic_top32_ph0_st1_to_1()
{
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);

    TTI_SFPTRANSP(0, 0, 0, 0);
}

template <bool dir>
inline void bitonic_top32_step_N()
{
    // Step N
    if constexpr (dir == static_cast<bool>(SortDir::ArgMax))
    {
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    }
    else
    {
        // Min
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    }
}
inline void bitonic_top32_inc_x8_dest(std::uint32_t inc)
{
    std::uint32_t inc_grp8 = inc >> 3;
    for (std::uint32_t i = 0; i < inc_grp8; i++)
    {
        TTI_INCRWC(0, 8, 0, 0);
    }
}

inline void bitonic_top32_inc_x4_dest(std::uint32_t inc, bool cr)
{
    std::uint32_t inc_grp4 = inc >> 2;
    if (cr)
    {
        for (std::uint32_t i = 0; i < inc_grp4; i++)
        {
            TTI_INCRWC(0b100, 4, 0, 0);
        }
    }
    else
    {
        for (std::uint32_t i = 0; i < inc_grp4; i++)
        {
            TTI_INCRWC(0, 4, 0, 0);
        }
    }
}

// Rotate every LREG0-7 lane group right by `num_shifts` SFPU instances.
template <std::uint32_t num_shifts>
inline void bitonic_top32_shift_instances_right()
{
#pragma GCC unroll 4
    for (std::uint32_t i = 0; i < num_shifts; i++)
    {
        TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
        TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
        TTI_SFPSHFT2(0, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
        TTI_SFPSHFT2(0, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
        TTI_SFPSHFT2(0, p_sfpu::LREG4, p_sfpu::LREG4, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
        TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
        TTI_SFPSHFT2(0, p_sfpu::LREG6, p_sfpu::LREG6, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
        TTI_SFPSHFT2(0, p_sfpu::LREG7, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    }
}

// Full 16-element bitonic sort of one 16-row block (Dest RWC += 16).
template <bool is_fp32_dest_acc_en, bool dir, std::uint32_t base>
inline void bitonic_top32_local_sort_block()
{
    bitonic_top32_load16<is_fp32_dest_acc_en, 4, 8, base>();
    bitonic_top32_ph0_st1_to_1();
    bitonic_top32_ph1_st2_to_1();
    bitonic_top32_ph2_st3_to_1();
    bitonic_top32_ph3_st4_to_1<dir>();
    bitonic_top32_store16<is_fp32_dest_acc_en, true, 4, 8, base>();
}

// One compare-exchange step at distance `dist` rows over a 16-row block (Dest RWC += 8).
template <bool is_fp32_dest_acc_en, bool dir, std::uint32_t dist, std::uint32_t base>
inline void bitonic_top32_step_N_block()
{
    bitonic_top32_load16<is_fp32_dest_acc_en, 4, dist, base>();
    bitonic_top32_step_N<dir>();
    bitonic_top32_store16<is_fp32_dest_acc_en, false, 4, dist, base>();
    bitonic_top32_inc_x8_dest(8);
}

// Steps 4..1 over a 16-row block (Dest RWC += 16).
template <bool is_fp32_dest_acc_en, bool dir, std::uint32_t base>
inline void bitonic_top32_ph3_block()
{
    bitonic_top32_load16<is_fp32_dest_acc_en, 4, 8, base>();
    bitonic_top32_ph3_st4_to_1<dir>();
    bitonic_top32_store16<is_fp32_dest_acc_en, true, 4, 8, base>();
}

template <bool is_fp32_dest_acc_en, bool idir>
inline void bitonic_top32_phases_steps_impl()
{
    constexpr bool dir0 = idir;
    constexpr bool dir1 = !idir;

    // produce bitonic sequences len=16
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    bitonic_top32_local_sort_block<is_fp32_dest_acc_en, dir0, 0>();
    bitonic_top32_local_sort_block<is_fp32_dest_acc_en, dir1, 0>();
    bitonic_top32_local_sort_block<is_fp32_dest_acc_en, dir0, 0>();
    bitonic_top32_local_sort_block<is_fp32_dest_acc_en, dir1, 0>();

    // produce bitonic sequences len=32: step 5 (dist 16), two 16-row blocks per direction
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    bitonic_top32_step_N_block<is_fp32_dest_acc_en, dir0, 16, 0>();
    bitonic_top32_step_N_block<is_fp32_dest_acc_en, dir0, 16, 0>();
    bitonic_top32_inc_x8_dest(16);
    bitonic_top32_step_N_block<is_fp32_dest_acc_en, dir1, 16, 0>();
    bitonic_top32_step_N_block<is_fp32_dest_acc_en, dir1, 16, 0>();
    bitonic_top32_inc_x8_dest(16);

    // steps 4 to 1
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    bitonic_top32_ph3_block<is_fp32_dest_acc_en, dir0, 0>();
    bitonic_top32_ph3_block<is_fp32_dest_acc_en, dir0, 0>();
    bitonic_top32_ph3_block<is_fp32_dest_acc_en, dir1, 0>();
    bitonic_top32_ph3_block<is_fp32_dest_acc_en, dir1, 0>();
}

template <bool is_fp32_dest_acc_en, bool top_min, bool across_tiles, std::uint32_t base = 0>
TT_ALWAYS_INLINE void bitonic_top32_merge_impl()
{
    constexpr std::uint32_t dist = across_tiles ? 64 : 32;

    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    bitonic_top32_step_N_block<is_fp32_dest_acc_en, top_min, dist, base>();
    bitonic_top32_step_N_block<is_fp32_dest_acc_en, top_min, dist, base>();
    bitonic_top32_step_N_block<is_fp32_dest_acc_en, top_min, dist, base>();
    bitonic_top32_step_N_block<is_fp32_dest_acc_en, top_min, dist, base>();
}

template <bool is_fp32_dest_acc_en, bool idir, bool skip_second, std::uint32_t base = 0>
TT_ALWAYS_INLINE void bitonic_top32_rebuild_impl()
{
    constexpr std::uint32_t dist = 16;

    // Step 5
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    bitonic_top32_step_N_block<is_fp32_dest_acc_en, idir, dist, base>();
    bitonic_top32_step_N_block<is_fp32_dest_acc_en, idir, dist, base>();
    bitonic_top32_inc_x8_dest(16);
    if constexpr (!skip_second)
    {
        bitonic_top32_step_N_block<is_fp32_dest_acc_en, !idir, dist, base>();
        bitonic_top32_step_N_block<is_fp32_dest_acc_en, !idir, dist, base>();
        bitonic_top32_inc_x8_dest(16);
    }

    // steps 4 to 1
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    bitonic_top32_ph3_block<is_fp32_dest_acc_en, idir, base>();
    bitonic_top32_ph3_block<is_fp32_dest_acc_en, idir, base>();
    if constexpr (!skip_second)
    {
        bitonic_top32_ph3_block<is_fp32_dest_acc_en, !idir, base>();
        bitonic_top32_ph3_block<is_fp32_dest_acc_en, !idir, base>();
    }
}

// The runtime-argument entry points below keep their signatures (they are what the compute
// kernels and the llk_math_deepseek_top32_rm_* wrappers call); each dispatches once to a fully
// immediate-encoded instantiation, so the instruction stream no longer depends on the
// arguments through the instruction buffer.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _bitonic_top32_phases_steps_(const int idir)
{
    if (idir)
    {
        bitonic_top32_phases_steps_impl<is_fp32_dest_acc_en, true>();
    }
    else
    {
        bitonic_top32_phases_steps_impl<is_fp32_dest_acc_en, false>();
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool top_min>
TT_ALWAYS_INLINE void _bitonic_top32_merge_(const bool across_tiles)
{
    if (across_tiles)
    {
        bitonic_top32_merge_impl<is_fp32_dest_acc_en, top_min, true>();
    }
    else
    {
        bitonic_top32_merge_impl<is_fp32_dest_acc_en, top_min, false>();
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
TT_ALWAYS_INLINE void _bitonic_top32_rebuild_(const bool idir, const bool skip_second)
{
    if (idir)
    {
        if (skip_second)
        {
            bitonic_top32_rebuild_impl<is_fp32_dest_acc_en, true, true>();
        }
        else
        {
            bitonic_top32_rebuild_impl<is_fp32_dest_acc_en, true, false>();
        }
    }
    else
    {
        if (skip_second)
        {
            bitonic_top32_rebuild_impl<is_fp32_dest_acc_en, false, true>();
        }
        else
        {
            bitonic_top32_rebuild_impl<is_fp32_dest_acc_en, false, false>();
        }
    }
}

// clang-format off
/**
 * Produces bitonic top32 on 16 independent columns of 1024 elements
 * Input data must be in row major (RM) layout and pre-sorted to len 32 sub arrays
 * The data needs to be loaded into the DST register transposed, as the sorting happens on columns
 * The indices need to be loaded into the DST register in the same way, but with offset of 2 tiles
 *
 * Algorithm:
 * 1. Reduild len 32 bitonic sequences from the pre-sorted data
 *    - do on both even and odd cols
 * 2. Merge and rebuild F0/F1 sequences with F2/F3 sequences
 *    - do on both even and odd cols
 *    - even and odd cols alternate in sort direction
 *
 * dst_index is unused: the tile is addressed through the Dest base that
 * _llk_math_eltwise_sfpu_start_(dst_index) programs, and the odd columns through the +2
 * `base` of the load/store immediates. Callers pass the same tile index to both.
 */
// clang-format on

// Step 1 of _bitonic_top32_of_1024_rm_pre_sorted_prep_ is the same for both top_min
// polarities, so it is one out-of-line copy: every instruction here is an immediate-encoded
// TTI_ word, and two inlined copies (924 instructions each) overflowed the 16 KB TRISC1 code
// region of kernels that instantiate the whole family. One call per prep costs a few cycles.
template <bool is_fp32_dest_acc_en>
inline NOINLINE void bitonic_top32_pre_sorted_prep_step1()
{
    constexpr std::uint32_t even_cols = 0;
    constexpr std::uint32_t odd_cols  = 2;
    constexpr bool decreasing         = false;
    constexpr bool increasing         = true;

    /// Step 1
    // Build len 32 bitonic sequences from the pre-sorted data (even cols, then odd cols)
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    bitonic_top32_ph3_block<is_fp32_dest_acc_en, decreasing, even_cols>();
    bitonic_top32_ph3_block<is_fp32_dest_acc_en, increasing, even_cols>();
    bitonic_top32_ph3_block<is_fp32_dest_acc_en, decreasing, even_cols>();
    bitonic_top32_ph3_block<is_fp32_dest_acc_en, increasing, even_cols>();
    bitonic_top32_rebuild_impl<is_fp32_dest_acc_en, decreasing, false /* skip_second */, even_cols>();

    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    bitonic_top32_ph3_block<is_fp32_dest_acc_en, decreasing, odd_cols>();
    bitonic_top32_ph3_block<is_fp32_dest_acc_en, increasing, odd_cols>();
    bitonic_top32_ph3_block<is_fp32_dest_acc_en, decreasing, odd_cols>();
    bitonic_top32_ph3_block<is_fp32_dest_acc_en, increasing, odd_cols>();
    bitonic_top32_rebuild_impl<is_fp32_dest_acc_en, decreasing, false /* skip_second */, odd_cols>();
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool top_min>
inline void _bitonic_top32_of_1024_rm_pre_sorted_prep_([[maybe_unused]] std::uint32_t dst_index)
{
    constexpr std::uint32_t even_cols = 0;
    constexpr std::uint32_t odd_cols  = 2;
    constexpr bool decreasing         = false;

    bitonic_top32_pre_sorted_prep_step1<is_fp32_dest_acc_en>();

    /// Step 2
    // Merge and rebuild F0/F1 sequences with F2/F3 sequences; even and odd cols alternate direction
    bitonic_top32_merge_impl<is_fp32_dest_acc_en, decreasing, false /* across_tiles */, even_cols>();
    bitonic_top32_rebuild_impl<is_fp32_dest_acc_en, top_min, true /* skip_second */, even_cols>();
    bitonic_top32_merge_impl<is_fp32_dest_acc_en, decreasing, false /* across_tiles */, odd_cols>();
    bitonic_top32_rebuild_impl<is_fp32_dest_acc_en, !top_min, true /* skip_second */, odd_cols>();
}

// clang-format off
/**
 * Combines top32 sequences across F0/F1 of 2 adjacent tiles independently on 16 columns
 * Implemented with simple merge and rebuild steps
 * dst_index is unused, see _bitonic_top32_of_1024_rm_pre_sorted_prep_.
 */
// clang-format on

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _bitonic_top32_of_1024_rm_pre_sorted_combine_([[maybe_unused]] std::uint32_t dst_index)
{
    constexpr std::uint32_t even_cols = 0;
    constexpr std::uint32_t odd_cols  = 2;
    constexpr bool decreasing         = false;
    constexpr bool increasing         = true;

    bitonic_top32_merge_impl<is_fp32_dest_acc_en, decreasing, true /* across_tiles */, even_cols>();
    bitonic_top32_rebuild_impl<is_fp32_dest_acc_en, decreasing, true /* skip_second */, even_cols>();
    bitonic_top32_merge_impl<is_fp32_dest_acc_en, decreasing, true /* across_tiles */, odd_cols>();
    bitonic_top32_rebuild_impl<is_fp32_dest_acc_en, increasing, true /* skip_second */, odd_cols>();
}

// The two direction-independent parts of every _bitonic_top32_of_1024_rm_pre_sorted_final_
// round, kept out of line so the four rounds share one copy (see
// bitonic_top32_pre_sorted_prep_step1 for why code size matters here).
template <bool is_fp32_dest_acc_en>
inline NOINLINE void bitonic_top32_final_merge_even_odd()
{
    constexpr bool decreasing         = false;
    constexpr std::uint32_t even_cols = 0;
    constexpr std::uint32_t odd_cols  = 2;

    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    bitonic_top32_step_N_block<is_fp32_dest_acc_en, decreasing, odd_cols, even_cols>();
    bitonic_top32_step_N_block<is_fp32_dest_acc_en, decreasing, odd_cols, even_cols>();
    bitonic_top32_step_N_block<is_fp32_dest_acc_en, decreasing, odd_cols, even_cols>();
    bitonic_top32_step_N_block<is_fp32_dest_acc_en, decreasing, odd_cols, even_cols>();
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
}

template <bool is_fp32_dest_acc_en>
inline NOINLINE void bitonic_top32_final_step5()
{
    constexpr bool decreasing         = false;
    constexpr std::uint32_t even_cols = 0;

    bitonic_top32_step_N_block<is_fp32_dest_acc_en, decreasing, 16, even_cols>();
    bitonic_top32_step_N_block<is_fp32_dest_acc_en, decreasing, 16, even_cols>();
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
}

// One reduction round of _bitonic_top32_of_1024_rm_pre_sorted_final_: merge even and odd cols
// and rebuild, with every other `swap_dir_lane_mask` SFPU instance sorting in the opposite
// direction, then (unless `last`) store the result to the odd cols, shift it right by
// `num_shifts` SFPU instances and store that to the even cols.
//
// The shift used to be a separate pass that reloaded the odd cols it had just stored; it now
// runs on the registers that still hold them. Index tracking has to be off for SFPSHFT2, and
// the second 16-row block's steps 4..1 still need the per-instance EXCHANGE_SRCB_SRCC bits
// the lane-masked SFPCONFIG set, so the first block only toggles ENABLE_DEST_INDEX (AND/OR
// on LaneConfig). The second block writes LaneConfig = 0 / 0x004 like the old pass did,
// which also clears the direction bits for the next round.
template <bool is_fp32_dest_acc_en, std::uint32_t swap_dir_lane_mask, std::uint32_t num_shifts, bool last>
inline void bitonic_top32_final_round()
{
    constexpr bool decreasing         = false;
    constexpr std::uint32_t even_cols = 0;
    constexpr std::uint32_t odd_cols  = 2;

    // Merge even and odd cols (dist 2) and rebuild
    bitonic_top32_final_merge_even_odd<is_fp32_dest_acc_en>();
    if constexpr (!last)
    {
        // set the selected SFPU instances to the opposite SWAP direction
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0x0104);
        TTI_SFPCONFIG(swap_dir_lane_mask, 0xF, 8);
    }
    bitonic_top32_final_step5<is_fp32_dest_acc_en>();

    if constexpr (last)
    {
        // Final col is produced in even col 0 of F0/F1
        bitonic_top32_ph3_block<is_fp32_dest_acc_en, decreasing, even_cols>();
        bitonic_top32_ph3_block<is_fp32_dest_acc_en, decreasing, even_cols>();
    }
    else
    {
        constexpr std::uint32_t sfpconfig_imm_and = 1 | 4; // MOD1_IMM16_IS_VALUE | MOD1_BITWISE_AND
        constexpr std::uint32_t sfpconfig_imm_or  = 1 | 2; // MOD1_IMM16_IS_VALUE | MOD1_BITWISE_OR

        // Rows 0-15
        bitonic_top32_load16<is_fp32_dest_acc_en, 4, 8, even_cols>();
        bitonic_top32_ph3_st4_to_1<decreasing>();
        TTI_SFPCONFIG(0xFFFB, 0xF, sfpconfig_imm_and); // index tracking off, keep the per-instance direction
        bitonic_top32_store16<is_fp32_dest_acc_en, false, 4, 8, odd_cols>();
        bitonic_top32_shift_instances_right<num_shifts>();
        bitonic_top32_store16<is_fp32_dest_acc_en, true, 4, 8, even_cols>();
        TTI_SFPCONFIG(0x0004, 0xF, sfpconfig_imm_or); // index tracking back on

        // Rows 16-31
        bitonic_top32_load16<is_fp32_dest_acc_en, 4, 8, even_cols>();
        bitonic_top32_ph3_st4_to_1<decreasing>();
        TTI_SFPCONFIG(0x0000, 0xF, 1); // disable SFPU config for shifting
        bitonic_top32_store16<is_fp32_dest_acc_en, false, 4, 8, odd_cols>();
        bitonic_top32_shift_instances_right<num_shifts>();
        bitonic_top32_store16<is_fp32_dest_acc_en, true, 4, 8, even_cols>();
        TTI_SFPCONFIG(0x0004, 0xF, 1); // Restore index tracking mode
    }
}

// clang-format off
/**
 * Produces final top32 from sequences in F0/F1 with data pre-sorted to len 32 sub arrays
 * Odd cols start with decreasing, even cols increasing
 * Final output is in even col 0 of F0/F1
 *
 * Algorithm:
 * 1. Merge even and odd cols and rebuild, then store to odd cols
 *    - alternate SFPU instances with increasing/decreasing
 *    - after this step, there are 8 cols remaining (all odd cols)
 * 2. Shift odd cols by 1 SFPU instance right, and store to even cols
 * 3. Merge even and odd cols and rebuild, then store to odd cols
 *    - alternate every 2nd SFPU instance with increasing/decreasing
 *    - after this step, there are 4 cols remaining (every 2nd odd col)
 * 4. Shift odd cols by 2 SFPU instances right, and store to even cols
 * 5. Merge even and odd cols and rebuild, then store to odd cols
 *    - alternate every 4th SFPU instance with increasing/decreasing
 *    - after this step, there are 2 cols remaining (every 4th odd col)
 * 6. Shift odd cols by 4 SFPU instances right, and store to even cols
 * 7. Merge even and odd cols and rebuild, then store to even cols
 *    - after this step, final col is produced in even col 0 of F0/F1
 * Each shift (2, 4, 6) is fused into the preceding step, see bitonic_top32_final_round.
 * dst_index is unused, see _bitonic_top32_of_1024_rm_pre_sorted_prep_.
 */
// clang-format on

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _bitonic_top32_of_1024_rm_pre_sorted_final_([[maybe_unused]] std::uint32_t dst_index)
{
    bitonic_top32_final_round<is_fp32_dest_acc_en, 0x4444, 1, false>(); // steps 1-2
    bitonic_top32_final_round<is_fp32_dest_acc_en, 0x5050, 2, false>(); // steps 3-4
    bitonic_top32_final_round<is_fp32_dest_acc_en, 0x5500, 4, false>(); // steps 5-6
    bitonic_top32_final_round<is_fp32_dest_acc_en, 0x0000, 0, true>();  // step 7
}

inline void _top32_rm_configure_addrmod_()
{
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 16},
    }
        .set(ADDR_MOD_6);
}

inline void _top32_rm_init_()
{
    _sfpu_load_config32_(0xF, 0x0, 0x4); // Set bit [2] of the SFPU_CONTROL_REG to enable index tracking mode
    _top32_rm_configure_addrmod_();
}

} // namespace sfpu
} // namespace ckernel
