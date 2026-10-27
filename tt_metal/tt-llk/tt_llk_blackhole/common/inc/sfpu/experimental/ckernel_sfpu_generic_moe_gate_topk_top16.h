// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Top-16 implementation details. Include through
// ckernel_sfpu_generic_moe_gate_topk.h so shared helpers are defined first.

#include <cstdint>

namespace ckernel
{
namespace sfpu
{

template <std::uint32_t offset>
inline void _generic_moe_gate_top16_bitonic16_directional_()
{
    if constexpr (offset == 0)
    {
        // Step 4.
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

        // Step 3.
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

        // Step 2.
        TTI_SFPTRANSP(0, 0, 0, 0);
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

        // Step 1.
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    }
    else if constexpr (offset == 2)
    {
        // Odd columns sort in the opposite direction.
        // Step 4.
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);

        // Step 3.
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);

        // Step 2.
        TTI_SFPTRANSP(0, 0, 0, 0);
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);

        // Step 1.
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    }
    TTI_SFPTRANSP(0, 0, 0, 0);
}

inline void _generic_moe_gate_top16_reverse_sort_order_()
{
    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG3, p_sfpswap::UNCONDITIONALLY);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpswap::UNCONDITIONALLY);
    TTI_SFPTRANSP(0, 0, 0, 0);
}

template <std::uint32_t offset>
inline void _generic_moe_gate_top16_store_16_rows_reverse_()
{
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, generic_moe_gate_bias_tile + 12 + offset);
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, generic_moe_gate_bias_tile + 8 + offset);
    TTI_SFPSTORE(p_sfpu::LREG2, 0, ADDR_MOD_7, generic_moe_gate_bias_tile + 4 + offset);
    TTI_SFPSTORE(p_sfpu::LREG3, 0, ADDR_MOD_7, generic_moe_gate_bias_tile + 0 + offset);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, generic_moe_gate_indices_tile + 12 + offset);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, generic_moe_gate_indices_tile + 8 + offset);
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, generic_moe_gate_indices_tile + 4 + offset);
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, generic_moe_gate_indices_tile + 0 + offset);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, generic_moe_gate_scores_tile + 12 + offset);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, generic_moe_gate_scores_tile + 8 + offset);
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, generic_moe_gate_scores_tile + 4 + offset);
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, generic_moe_gate_scores_tile + 0 + offset);
}

inline void _generic_moe_gate_top16_shift_left_once_()
{
    // Index tracking must be disabled for SHFT2, so shift value and
    // score/index LREGs explicitly before restoring tracking.
    TTI_SFPCONFIG(0, 0xF, 1);
    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);

    TTI_SFPSHFT2(0, p_sfpu::LREG4, p_sfpu::LREG4, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG6, p_sfpu::LREG6, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG7, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPCONFIG(0x4, 0xF, 1);
}

template <std::uint32_t offset>
inline void _generic_moe_gate_top16_reduce_even_odd_columns_to_instance_()
{
    _generic_moe_gate_load_8_rows_even_odd_split_<offset>();

    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

    _generic_moe_gate_store_8_rows_even_odd_split_<offset>();

    _generic_moe_gate_load_8_rows_even_odd_split_<offset + 8>();

    // Keep these winners in LREG2/3 (and tracked payloads in LREG6/7),
    // then reload only the first-half winners into the remaining LREGs.
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);

    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, generic_moe_gate_bias_tile + 0 + offset);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, generic_moe_gate_bias_tile + 4 + offset);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, generic_moe_gate_indices_tile + 0 + offset);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, generic_moe_gate_indices_tile + 4 + offset);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, generic_moe_gate_scores_tile + 0 + offset);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, generic_moe_gate_scores_tile + 4 + offset);
    _generic_moe_gate_bitonic8_steps_3_to_1_();
}

template <std::uint32_t lane_shifts, bool store_result = true>
inline void _generic_moe_gate_top16_reduce_lanes_()
{
    static_assert(lane_shifts == 1 || lane_shifts == 2 || lane_shifts == 4);

    // Sort the live run descending before shifting it to the next lane.
    _generic_moe_gate_top16_bitonic16_directional_<0>();
    _generic_moe_gate_store_16_rows_even_odd_split_<0>();

#pragma GCC unroll 4
    for (std::uint32_t shift = 0; shift < lane_shifts; ++shift)
    {
        _generic_moe_gate_top16_shift_left_once_();
    }
    // Horizontal shifts preserve row order; reverse the descending run
    // into the ascending half required by the bitonic merge.
    _generic_moe_gate_top16_reverse_sort_order_();
    _generic_moe_gate_top16_store_16_rows_reverse_<2>();

    _generic_moe_gate_top16_reduce_even_odd_columns_to_instance_<0>();
    if constexpr (store_result)
    {
        _generic_moe_gate_store_16_rows_even_odd_split_<0>();
    }
}

template <std::uint32_t load_offset, std::uint32_t store_offset>
inline void _generic_moe_gate_top16_sort_face_()
{
    _generic_moe_gate_load_16_rows_even_odd_split_<load_offset>();
    _generic_moe_gate_build_bitonic8_();
    _generic_moe_gate_top16_bitonic16_directional_<0>(); // evens descending
    _generic_moe_gate_store_16_rows_even_odd_split_<load_offset>();

    _generic_moe_gate_load_16_rows_even_odd_split_<load_offset + 2>();
    _generic_moe_gate_build_bitonic8_();
    _generic_moe_gate_top16_bitonic16_directional_<2>(); // odds ascending
    _generic_moe_gate_store_16_rows_even_odd_split_<load_offset + 2>();

    _generic_moe_gate_top16_reduce_even_odd_columns_to_instance_<load_offset>();
    _generic_moe_gate_store_16_rows_even_odd_split_<store_offset>();
}

template <std::uint32_t load_offset, std::uint32_t store_offset>
inline void _generic_moe_gate_top16_merge_bitonic_face_()
{
    _generic_moe_gate_load_16_rows_even_odd_split_<load_offset>();
    _generic_moe_gate_top16_bitonic16_directional_<0>(); // evens descending
    _generic_moe_gate_store_16_rows_even_odd_split_<load_offset>();

    _generic_moe_gate_load_16_rows_even_odd_split_<load_offset + 2>();
    _generic_moe_gate_top16_bitonic16_directional_<2>(); // odds ascending
    _generic_moe_gate_store_16_rows_even_odd_split_<load_offset + 2>();

    _generic_moe_gate_top16_reduce_even_odd_columns_to_instance_<load_offset>();
    _generic_moe_gate_store_16_rows_even_odd_split_<store_offset>();
}

template <std::uint32_t load_offset, std::uint32_t store_offset>
inline void _generic_moe_gate_top16_sort_half_face_()
{
    _generic_moe_gate_load_8_rows_even_odd_split_<load_offset>();
    _generic_moe_gate_build_bitonic8_();
    _generic_moe_gate_store_16_rows_even_odd_split_<store_offset>();
}

template <std::uint32_t face_idx, bool full_face>
inline void _generic_moe_gate_top16_accumulate_face_()
{
    if constexpr (full_face)
    {
        _generic_moe_gate_top16_sort_face_<face_idx * 16, 2>();
    }
    else
    {
        _generic_moe_gate_top16_sort_half_face_<face_idx * 16, 2>();
    }
    // Both staged runs already satisfy the bitonic-8 invariant.
    _generic_moe_gate_top16_merge_bitonic_face_<0, 0>();
}

template <int num_total_experts>
inline void _generic_moe_gate_top16_sort_to_instance_()
{
    static_assert(num_total_experts >= 128 && num_total_experts <= 1024);
    static_assert(num_total_experts % 128 == 0);

    if constexpr (num_total_experts == 128)
    {
        _generic_moe_gate_top16_sort_half_face_<0, 0>();
    }
    else
    {
        _generic_moe_gate_top16_sort_face_<0, 0>();
    }
    if constexpr (num_total_experts > 256)
    {
        _generic_moe_gate_top16_accumulate_face_<1, num_total_experts >= 512>();
    }
    if constexpr (num_total_experts > 512)
    {
        _generic_moe_gate_top16_accumulate_face_<2, num_total_experts >= 768>();
    }
    if constexpr (num_total_experts > 768)
    {
        _generic_moe_gate_top16_accumulate_face_<3, num_total_experts >= 1024>();
    }
}

inline void _generic_moe_gate_top16_store_outputs_()
{
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, generic_moe_gate_indices_tile + 0);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, generic_moe_gate_indices_tile + 4);
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, generic_moe_gate_indices_tile + 8);
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, generic_moe_gate_indices_tile + 12);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, generic_moe_gate_scores_tile + 0);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, generic_moe_gate_scores_tile + 4);
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, generic_moe_gate_scores_tile + 8);
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, generic_moe_gate_scores_tile + 12);
}

inline void _generic_moe_gate_top16_merge_instances_()
{
    _generic_moe_gate_top16_reduce_lanes_<1>();
    _generic_moe_gate_top16_reduce_lanes_<2>();
    _generic_moe_gate_top16_reduce_lanes_<4, false>();

    // Move the final surviving lane into output column 0.
    _generic_moe_gate_top16_shift_left_once_();
    _generic_moe_gate_top16_store_outputs_();
}

template <bool normalize, int num_selected_experts, int num_total_experts, bool zero_tail, bool full_sort, bool do_extra_scale = false>
inline void _generic_moe_gate_top16_(std::uint32_t eps, std::uint32_t scale, std::uint32_t extra_scale = 0)
{
    static_assert(num_selected_experts >= 9 && num_selected_experts <= 16);

    _generic_moe_gate_top16_sort_to_instance_<num_total_experts>();
    _generic_moe_gate_top16_merge_instances_();

    if constexpr (num_selected_experts < 16 || full_sort)
    {
        _generic_moe_gate_top16_bitonic16_directional_<0>();
        _generic_moe_gate_top16_store_outputs_();
    }

    if constexpr (num_selected_experts < 16 && (zero_tail || normalize))
    {
        _generic_moe_gate_zero_tail_<num_selected_experts - 8, 8>();
    }

    if constexpr (normalize)
    {
        _generic_moe_gate_normalize_<16, generic_moe_gate_scores_tile, do_extra_scale>(eps, scale, extra_scale);
    }
}

} // namespace sfpu
} // namespace ckernel
