// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_sfpu_recip.h"
#include "lltt.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpu/ckernel_sfpu_load_config.h"

namespace ckernel
{
namespace sfpu
{

static constexpr std::uint32_t generic_moe_gate_dst_tile_offset = 64;
static constexpr std::uint32_t generic_moe_gate_scores_tile     = 0;
static constexpr std::uint32_t generic_moe_gate_indices_tile    = 1 * generic_moe_gate_dst_tile_offset;
static constexpr std::uint32_t generic_moe_gate_bias_tile       = 2 * generic_moe_gate_dst_tile_offset;
static constexpr std::uint32_t generic_moe_gate_interm_tile     = 3 * generic_moe_gate_dst_tile_offset;

// Load a full 16-row face. LREG0-3 hold biased scores; LREG4-7 pack
// indices into LO16 and original scores into HI16.
template <std::uint32_t offset>
inline void _generic_moe_gate_load_16_rows_even_odd_split_()
{
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, generic_moe_gate_bias_tile + 0 + offset);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, generic_moe_gate_bias_tile + 4 + offset);
    TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_7, generic_moe_gate_bias_tile + 8 + offset);
    TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_7, generic_moe_gate_bias_tile + 12 + offset);

    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, generic_moe_gate_indices_tile + 0 + offset);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, generic_moe_gate_indices_tile + 4 + offset);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, generic_moe_gate_indices_tile + 8 + offset);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, generic_moe_gate_indices_tile + 12 + offset);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, generic_moe_gate_scores_tile + 0 + offset);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, generic_moe_gate_scores_tile + 4 + offset);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, generic_moe_gate_scores_tile + 8 + offset);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, generic_moe_gate_scores_tile + 12 + offset);
}

template <std::uint32_t offset>
inline void _generic_moe_gate_store_16_rows_even_odd_split_()
{
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, generic_moe_gate_bias_tile + 0 + offset);
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, generic_moe_gate_bias_tile + 4 + offset);
    TTI_SFPSTORE(p_sfpu::LREG2, 0, ADDR_MOD_7, generic_moe_gate_bias_tile + 8 + offset);
    TTI_SFPSTORE(p_sfpu::LREG3, 0, ADDR_MOD_7, generic_moe_gate_bias_tile + 12 + offset);

    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, generic_moe_gate_indices_tile + 0 + offset);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, generic_moe_gate_indices_tile + 4 + offset);
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, generic_moe_gate_indices_tile + 8 + offset);
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, generic_moe_gate_indices_tile + 12 + offset);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, generic_moe_gate_scores_tile + 0 + offset);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, generic_moe_gate_scores_tile + 4 + offset);
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, generic_moe_gate_scores_tile + 8 + offset);
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, generic_moe_gate_scores_tile + 12 + offset);
}

template <std::uint32_t offset>
inline void _generic_moe_gate_load_8_rows_even_odd_split_()
{
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, generic_moe_gate_bias_tile + 0 + offset);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, generic_moe_gate_bias_tile + 4 + offset);
    TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_7, generic_moe_gate_bias_tile + 2 + offset);
    TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_7, generic_moe_gate_bias_tile + 6 + offset);

    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, generic_moe_gate_indices_tile + 0 + offset);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, generic_moe_gate_indices_tile + 4 + offset);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, generic_moe_gate_indices_tile + 2 + offset);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, generic_moe_gate_indices_tile + 6 + offset);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, generic_moe_gate_scores_tile + 0 + offset);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, generic_moe_gate_scores_tile + 4 + offset);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, generic_moe_gate_scores_tile + 2 + offset);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, generic_moe_gate_scores_tile + 6 + offset);
}

template <std::uint32_t offset>
inline void _generic_moe_gate_store_8_rows_even_odd_split_()
{
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, generic_moe_gate_bias_tile + 0 + offset);
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, generic_moe_gate_bias_tile + 4 + offset);

    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, generic_moe_gate_indices_tile + 0 + offset);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, generic_moe_gate_indices_tile + 4 + offset);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, generic_moe_gate_scores_tile + 0 + offset);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, generic_moe_gate_scores_tile + 4 + offset);
}

// Shared P1-P3 network: build a length-16 bitonic sequence in LREG0-3.
inline void _generic_moe_gate_build_bitonic8_()
{
    // P1 - Bitonic 2.
    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);

    // P2 - Bitonic 4.
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX);

    // P3 - Bitonic 8.
    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_01_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_01_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX);
    TTI_SFPTRANSP(0, 0, 0, 0);
}

inline void _generic_moe_gate_bitonic8_steps_3_to_1_()
{
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);

    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_01_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX);

    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_01_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX);
    TTI_SFPTRANSP(0, 0, 0, 0);
}

template <int num_rows, int scores_offset>
inline void _generic_moe_gate_normalize_(std::uint32_t eps, std::uint32_t scale)
{
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, scores_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, scores_offset + 4);
    if constexpr (num_rows > 8)
    {
        TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_7, scores_offset + 8);
        TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_7, scores_offset + 12);
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG0, 0);
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG0, 0);
    }
    else
    {
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
    }

    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
    TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG2, 0);
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG0, 0);

    TTI_SFPCONFIG(0, 0xF, 1);
    sfpi::vFloat l0                 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vFloat eps_value          = Converter::as_float(eps);
    l0                              = l0 + eps_value;
    l0                              = sfpu_reciprocal<false>(l0);
    sfpi::vFloat scale_value        = Converter::as_float(scale);
    l0                              = l0 * scale_value;
    sfpi::l_reg[sfpi::LRegs::LReg0] = l0;
    TTI_SFPNOP;

    TTI_SFPCONFIG(0, p_sfpu::LREG14, 0);
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, scores_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, scores_offset + 4);
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG14, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG14, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, scores_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, scores_offset + 4);
    if constexpr (num_rows > 8)
    {
        TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_7, scores_offset + 8);
        TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_7, scores_offset + 12);
        TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG14, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
        TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);
        TTI_SFPSTORE(p_sfpu::LREG2, 0, ADDR_MOD_7, scores_offset + 8);
        TTI_SFPSTORE(p_sfpu::LREG3, 0, ADDR_MOD_7, scores_offset + 12);
    }
}

template <int num_total_experts>
inline void _topk_moe_generate_indices_()
{
    static_assert(num_total_experts >= 128 && num_total_experts <= 1024);
    static_assert(num_total_experts % 128 == 0);
    constexpr std::uint32_t num_blocks = num_total_experts / 128;

    TTI_SFPMOV(0, p_sfpu::LTILEID, p_sfpu::LREG0, 0);

    TTI_SFPIADD(1, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(64, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(65, p_sfpu::LREG0, p_sfpu::LREG3, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);

#pragma GCC unroll 8
    for (std::uint32_t block = 0; block < num_blocks; block++)
    {
        const std::uint32_t offset = block * 8;
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::LO16, ADDR_MOD_7, generic_moe_gate_indices_tile + offset + 0);
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::LO16, ADDR_MOD_7, generic_moe_gate_indices_tile + offset + 2);
        TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::LO16, ADDR_MOD_7, generic_moe_gate_indices_tile + offset + 4);
        TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::LO16, ADDR_MOD_7, generic_moe_gate_indices_tile + offset + 6);

        TTI_SFPIADD(128, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPIADD(128, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPIADD(128, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPIADD(128, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    }
}

} // namespace sfpu
} // namespace ckernel

#include "sfpu/experimental/ckernel_sfpu_generic_moe_gate_topk_top16.h"
#include "sfpu/experimental/ckernel_sfpu_generic_moe_gate_topk_top8.h"

namespace ckernel
{
namespace sfpu
{

inline void _init_generic_moe_gate_topk_()
{
    sfpu_reciprocal_init<false>();
}

// generate_indices fills DST[1] with the position identity [0..N-1] before sorting. Set it false when the
// caller has already loaded DST[1] with its own mapping (e.g. bit-15-flagged SRAM slots) via
// copy_tile(input_indices, 0, 1): the sort's LREG4-7 LO16 load then reads those values verbatim and the
// paired sort carries them through to out_indices.
template <bool normalize, int num_selected_experts, int num_total_experts, bool zero_tail, bool full_sort, bool generate_indices = true>
inline void _generic_moe_gate_topk_(std::uint32_t eps, std::uint32_t scale)
{
    if constexpr (generate_indices)
    {
        _topk_moe_generate_indices_<num_total_experts>();
    }
    TTI_SFPCONFIG(0x4, 0xF, 1);

    if constexpr (num_selected_experts == 16)
    {
        _generic_moe_gate_top16_<normalize, num_total_experts>(eps, scale);
    }
    else
    {
        _generic_moe_gate_top8_<normalize, num_selected_experts, num_total_experts, zero_tail, full_sort>(eps, scale);
    }
}

} // namespace sfpu
} // namespace ckernel
