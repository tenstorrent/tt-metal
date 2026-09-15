// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Top-8 implementation details. Include through
// ckernel_sfpu_generic_moe_gate_topk.h so shared helpers are defined first.

#include <cstdint>

namespace ckernel
{
namespace sfpu
{

// At the end of this function, LREG0 and LREG1 contain the top 8 values per column.
inline void _generic_moe_gate_top8_local_sort_16x8_to_8x8_()
{
    _generic_moe_gate_build_bitonic8_();

    // P4 - Partial Bitonic 16 (top8), Step 4.
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
}

inline void _generic_moe_gate_top8_rebuild_and_merge_16x8_to_8x8_()
{
    _generic_moe_gate_bitonic8_steps_3_to_1_();

    // Merge top16 rows.
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
}

inline void _generic_moe_gate_top8_merge_instances_()
{
    TTI_SFPCONFIG(0, 0xF, 1);
    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG3, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);

    TTI_SFPSHFT2(0, p_sfpu::LREG4, p_sfpu::LREG6, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPCONFIG(0x4, 0xF, 1);
    _generic_moe_gate_top8_rebuild_and_merge_16x8_to_8x8_();

    TTI_SFPCONFIG(0, 0xF, 1);
    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG3, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);

    TTI_SFPSHFT2(0, p_sfpu::LREG4, p_sfpu::LREG6, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG6, p_sfpu::LREG6, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG7, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPCONFIG(0x4, 0xF, 1);
    _generic_moe_gate_top8_rebuild_and_merge_16x8_to_8x8_();

    TTI_SFPCONFIG(0, 0xF, 1);
    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG3, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);

    TTI_SFPSHFT2(0, p_sfpu::LREG4, p_sfpu::LREG6, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG6, p_sfpu::LREG6, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG7, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG6, p_sfpu::LREG6, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG7, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG6, p_sfpu::LREG6, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG7, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPCONFIG(0x4, 0xF, 1);
    _generic_moe_gate_top8_rebuild_and_merge_16x8_to_8x8_();

    TTI_SFPCONFIG(0, 0xF, 1);
    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);

    TTI_SFPSHFT2(0, p_sfpu::LREG4, p_sfpu::LREG4, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPCONFIG(0x4, 0xF, 1);
}

template <int num_selected_experts, bool full_sort>
inline void _generic_moe_gate_top8_sort_rows_()
{
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    if constexpr (num_selected_experts != 4 || full_sort)
    {
        TTI_SFPTRANSP(0, 0, 0, 0);
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_01_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX);
        if constexpr ((num_selected_experts != 6 && num_selected_experts != 2) || full_sort)
        {
            TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_01_MAX);
            TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX);
        }
        TTI_SFPTRANSP(0, 0, 0, 0);
    }
    else
    {
        TTI_NOP;
    }
}

template <std::uint32_t offset>
inline void _generic_moe_gate_top8_load_result_into_upper_lregs_()
{
    TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_7, generic_moe_gate_bias_tile + 0 + offset);
    TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_7, generic_moe_gate_bias_tile + 4 + offset);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, generic_moe_gate_indices_tile + 0 + offset);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, generic_moe_gate_indices_tile + 4 + offset);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, generic_moe_gate_scores_tile + 0 + offset);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, generic_moe_gate_scores_tile + 4 + offset);
}

template <std::uint32_t load_offset, std::uint32_t store_offset, bool store_result = true>
inline void _generic_moe_gate_top8_sort_face_()
{
    _generic_moe_gate_load_16_rows_even_odd_split_<load_offset>();
    _generic_moe_gate_top8_local_sort_16x8_to_8x8_();
    _generic_moe_gate_store_8_rows_even_odd_split_<load_offset>();

    _generic_moe_gate_load_16_rows_even_odd_split_<load_offset + 2>();
    _generic_moe_gate_top8_local_sort_16x8_to_8x8_();

    _generic_moe_gate_top8_load_result_into_upper_lregs_<load_offset>();
    _generic_moe_gate_top8_rebuild_and_merge_16x8_to_8x8_();
    if constexpr (store_result)
    {
        _generic_moe_gate_store_8_rows_even_odd_split_<store_offset>();
    }
}

template <std::uint32_t load_offset, std::uint32_t store_offset, bool store_result = true>
inline void _generic_moe_gate_top8_sort_half_face_()
{
    _generic_moe_gate_load_8_rows_even_odd_split_<load_offset>();
    _generic_moe_gate_top8_local_sort_16x8_to_8x8_();
    if constexpr (store_result)
    {
        _generic_moe_gate_store_8_rows_even_odd_split_<store_offset>();
    }
}

template <std::uint32_t face_idx, bool full_face, bool store_result>
inline void _generic_moe_gate_top8_accumulate_face_()
{
    if constexpr (full_face)
    {
        _generic_moe_gate_top8_sort_face_<face_idx * 16, 0, false>();
    }
    else
    {
        _generic_moe_gate_top8_sort_half_face_<face_idx * 16, 0, false>();
    }
    _generic_moe_gate_top8_load_result_into_upper_lregs_<0>();
    _generic_moe_gate_top8_rebuild_and_merge_16x8_to_8x8_();
    if constexpr (store_result)
    {
        _generic_moe_gate_store_8_rows_even_odd_split_<0>();
    }
}

template <int num_total_experts>
inline void _generic_moe_gate_top8_sort_to_instance_()
{
    static_assert(num_total_experts >= 128 && num_total_experts <= 1024);
    static_assert(num_total_experts % 128 == 0);

    if constexpr (num_total_experts == 128)
    {
        _generic_moe_gate_top8_sort_half_face_<0, 0, false>();
    }
    else
    {
        _generic_moe_gate_top8_sort_face_<0, 0, (num_total_experts > 256)>();
    }
    if constexpr (num_total_experts > 256)
    {
        _generic_moe_gate_top8_accumulate_face_<1, (num_total_experts >= 512), (num_total_experts > 512)>();
    }
    if constexpr (num_total_experts > 512)
    {
        _generic_moe_gate_top8_accumulate_face_<2, (num_total_experts >= 768), (num_total_experts > 768)>();
    }
    if constexpr (num_total_experts > 768)
    {
        _generic_moe_gate_top8_accumulate_face_<3, (num_total_experts >= 1024), false>();
    }
}

template <bool normalize, int num_selected_experts, int num_total_experts, bool zero_tail, bool full_sort, bool do_extra_scale = false>
inline void _generic_moe_gate_top8_(std::uint32_t eps, std::uint32_t scale, std::uint32_t extra_scale = 0)
{
    _generic_moe_gate_top8_sort_to_instance_<num_total_experts>();
    _generic_moe_gate_top8_merge_instances_();

    if constexpr (num_selected_experts < 8 || full_sort)
    {
        _generic_moe_gate_top8_sort_rows_<num_selected_experts, full_sort>();
    }

    if constexpr (zero_tail || (normalize && num_selected_experts < 8))
    {
        _generic_moe_gate_zero_tail_lregs_<num_selected_experts>();
    }

    _generic_moe_gate_store_8_rows_even_odd_split_<0>();

    if constexpr (normalize)
    {
        _generic_moe_gate_normalize_<8, generic_moe_gate_scores_tile, do_extra_scale>(eps, scale, extra_scale);
    }

    if constexpr (zero_tail)
    {
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, generic_moe_gate_scores_tile + 8);
        TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, generic_moe_gate_scores_tile + 12);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, generic_moe_gate_indices_tile + 8);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, generic_moe_gate_indices_tile + 12);
    }
}

} // namespace sfpu
} // namespace ckernel
