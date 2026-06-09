// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "cunpack_common.h"
#include "llk_assert.h"
using namespace ckernel;
using namespace ckernel::trisc;

enum class p_dim_stride_target
{
    IGNORE,
    FACE_ROW_MAJOR
};

/**
 * @brief Programs unpacker L1 info and source register format.
 *
 * @tparam UNP_SEL: Selects which unpacker to configure, values = <p_unpacr::UNP_A/UNP_B/UNP_S/UNP_DEST>
 * @param tdma_desc_src: Contains source register format.
 */
template <std::uint32_t UNP_SEL>
inline void _llk_unpack_hw_configure_(const tdma_descriptor_t& tdma_desc_src)
{
    static_assert(
        (UNP_SEL == p_unpacr::UNP_A) || (UNP_SEL == p_unpacr::UNP_B) || (UNP_SEL == p_unpacr::UNP_S) || (UNP_SEL == p_unpacr::UNP_DEST),
        "UNP_SEL can only be set to p_unpacr::UNP_A/UNP_B/UNP_S/UNP_DEST");

    // RT: make defines to aggregate the source format address, to make the below a single function
    // Program src formats
    if constexpr (UNP_SEL == p_unpacr::UNP_A || UNP_SEL == p_unpacr::UNP_DEST)
    {
        cfg_rmw(THCON_UNPACKER0_REG0_OUT_DATA_FORMAT_RMW, static_cast<std::uint8_t>(tdma_desc_src.reg_data_format));
    }
    else if constexpr (UNP_SEL == p_unpacr::UNP_B)
    {
        cfg_rmw(THCON_UNPACKER1_REG0_OUT_DATA_FORMAT_RMW, static_cast<std::uint8_t>(tdma_desc_src.reg_data_format));
    }
    else if constexpr (UNP_SEL == p_unpacr::UNP_S)
    {
        cfg_rmw(THCON_UNPACKER2_REG0_OUT_DATA_FORMAT_RMW, static_cast<std::uint8_t>(tdma_desc_src.reg_data_format));
    }
}

// RT: make defines to aggregate _llk_unpack_hw_configure_ calls into one
/**
 * @brief Programs unpacker L1 info and source register format for a unary operation.
 *
 * @tparam UNP_SEL: Selects which unpacker to configure, values = <p_unpacr::UNP_A/UNP_B/UNP_S>
 * @param tdma_desc_src: Contains L1 buffer descriptor information and source register format for the source register.
 */
template <std::uint32_t UNP_SEL>
inline void _llk_unpack_configure_unary_(const tdma_descriptor_t& tdma_desc_src)
{
    _llk_unpack_hw_configure_<UNP_SEL>(tdma_desc_src);
}

/**
 * @brief Programs unpacker L1 info and source register format for a binary operation.
 *
 * @tparam UNP_SEL_0/1: Selects which unpacker to configure, values = <p_unpacr::UNP_A/UNP_B/UNP_S>
 * @param tdma_desc_src0/1: Contains L1 buffer descriptor information and source register format for the source register.
 */
template <std::uint32_t UNP_SEL_0, std::uint32_t UNP_SEL_1>
inline void _llk_unpack_configure_binary_(const tdma_descriptor_t& tdma_desc_src0, const tdma_descriptor_t& tdma_desc_src1)
{
    _llk_unpack_hw_configure_<UNP_SEL_0>(tdma_desc_src0);
    _llk_unpack_hw_configure_<UNP_SEL_1>(tdma_desc_src1);
}

/**
 * @brief Clears the unpack-to-dest data valid for the dest section after unpacking directly into DEST.
 *
 * For DstSync::SyncFull, also clears dvalid for dest bank 1 and resets the dest bank id to 0 so the
 * next section starts from bank 0, allowing the full dest register to be used.
 *
 * @tparam DST: Destination register buffering mode, values = <DstSync::SyncHalf/DstSync::SyncFull>
 */
template <DstSync DST>
inline void _llk_unpack_dest_dvalid_section_done_()
{
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::NOTHING, p_stall::WAIT_SFPU, p_stall::UNPACK0);
    TTI_CLEARDVALID(0, 0, 0, 0, p_cleardvalid::UNPACK_TO_DEST, 0);
    if constexpr (DST == DstSync::SyncFull)
    {
        // For DstSync::SyncFull issue a CLEARDVALID instruction for dest bank1 as well in order to use full dest register
        // Reset dest bank id to 0 for the given dest client to ensure SyncFull starts from bank0
        TTI_CLEARDVALID(0, 0, 0, p_cleardvalid::UNPACK_TO_DEST, p_cleardvalid::UNPACK_TO_DEST, 0);
    }
}

/**
 * @brief Reprograms the unpacker output DataFormat at runtime.
 *
 * Quasar unpack dynamic output format: reprograms only THCON `UNPACKER*_REG0_OUT_DATA_FORMAT`.
 * L1 layout and input encoding stay in the buffer descriptor; `unpack_src_format` is the BD/L1
 * DataFormat and is not written to unpacker config here. UNP_DEST is not a valid selector: there
 * is no source register to reprogram for the dest path.
 *
 * @tparam UNP_SEL: Unpacker to update, values = <p_unpacr::UNP_A/UNP_B/UNP_S>
 * @tparam EN_32BIT_DEST: FP32 dest accumulation (validated with unpack_src_format / unpack_dst_format).
 * @param unpack_src_format: BD/L1 input DataFormat (used only for the conversion check).
 * @param unpack_dst_format: OUT_DATA_FORMAT register value to program (unpacker gasket output).
 */
template <std::uint32_t UNP_SEL, bool EN_32BIT_DEST>
inline void _llk_unpack_reconfig_data_format_src_(const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format)
{
    static_assert(
        (UNP_SEL == p_unpacr::UNP_A) || (UNP_SEL == p_unpacr::UNP_B) || (UNP_SEL == p_unpacr::UNP_S), "UNP_SEL must be p_unpacr::UNP_A, UNP_B, or UNP_S");

    LLK_ASSERT(
        ckernel::unpack::is_quasar_unpack_reconfig_pair_supported<EN_32BIT_DEST>(unpack_src_format, unpack_dst_format, false /* unpack_to_dest */),
        "Unsupported Quasar unpacker OUT_DATA_FORMAT for this L1 format and unpack path.");

    const auto out_fmt = static_cast<std::uint8_t>(unpack_dst_format);

    // No STALLWAIT needed: THCON_UNPACKER<N>_REG0_OUT_DATA_FORMAT is a shadow register on Quasar (TEN-4169 INT_DESCALE bug is unrelated and PACKER-side).
    if constexpr (UNP_SEL == p_unpacr::UNP_A)
    {
        cfg_rmw(THCON_UNPACKER0_REG0_OUT_DATA_FORMAT_RMW, out_fmt);
    }
    else if constexpr (UNP_SEL == p_unpacr::UNP_B)
    {
        cfg_rmw(THCON_UNPACKER1_REG0_OUT_DATA_FORMAT_RMW, out_fmt);
    }
    else // UNP_S
    {
        cfg_rmw(THCON_UNPACKER2_REG0_OUT_DATA_FORMAT_RMW, out_fmt);
    }
}

/**
 * @brief Sets a dummy SrcB data valid via an UNPACR NOP.
 */
inline void _llk_unpack_set_srcB_dummy_valid_()
{
    TTI_UNPACR_NOP(p_unpacr::UNP_B, 1 /*Set_Dvalid*/, 0, 0, 0, p_unpacr::UNP_NOP);
}
