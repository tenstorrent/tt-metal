// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "cmath_common.h"
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
 * @brief Programs the unpacker source register format.
 *
 * L1 buffer info is not programmed here: it lives in buffer descriptors, which op inits
 * program via the per-TRISC allocator (see llk_bfd_alloc.h).
 *
 * @tparam UNP_SEL: Selects which unpacker to configure, values = <p_unpacr::UNP_A/UNP_B/UNP_S/UNP_DEST>
 * @param reg_data_format: Source register (out) data format.
 */
template <std::uint32_t UNP_SEL>
inline void _llk_unpack_hw_configure_(const DataFormat reg_data_format)
{
    static_assert(
        (UNP_SEL == p_unpacr::UNP_A) || (UNP_SEL == p_unpacr::UNP_B) || (UNP_SEL == p_unpacr::UNP_S) || (UNP_SEL == p_unpacr::UNP_DEST),
        "UNP_SEL can only be set to p_unpacr::UNP_A/UNP_B/UNP_S/UNP_DEST");

    // RT: make defines to aggregate the source format address, to make the below a single function
    // Program src formats
    if constexpr (UNP_SEL == p_unpacr::UNP_A || UNP_SEL == p_unpacr::UNP_DEST)
    {
        cfg_rmw(THCON_UNPACKER0_REG0_OUT_DATA_FORMAT_RMW, static_cast<std::uint8_t>(reg_data_format));
    }
    else if constexpr (UNP_SEL == p_unpacr::UNP_B)
    {
        cfg_rmw(THCON_UNPACKER1_REG0_OUT_DATA_FORMAT_RMW, static_cast<std::uint8_t>(reg_data_format));
    }
    else if constexpr (UNP_SEL == p_unpacr::UNP_S)
    {
        cfg_rmw(THCON_UNPACKER2_REG0_OUT_DATA_FORMAT_RMW, static_cast<std::uint8_t>(reg_data_format));
    }
}

// RT: make defines to aggregate _llk_unpack_hw_configure_ calls into one
/**
 * @brief Programs the source register format for a unary operation.
 *
 * @tparam UNP_SEL: Selects which unpacker to configure, values = <p_unpacr::UNP_A/UNP_B/UNP_S>
 * @param reg_data_format_src: Source register (out) data format.
 */
template <std::uint32_t UNP_SEL>
inline void _llk_unpack_configure_unary_(const DataFormat reg_data_format_src)
{
    _llk_unpack_hw_configure_<UNP_SEL>(reg_data_format_src);
}

/**
 * @brief Programs the source register format for a binary operation.
 *
 * @tparam UNP_SEL_0/1: Selects which unpacker to configure, values = <p_unpacr::UNP_A/UNP_B/UNP_S>
 * @param reg_data_format_src0/1: Source register (out) data format for each unpacker.
 */
template <std::uint32_t UNP_SEL_0, std::uint32_t UNP_SEL_1>
inline void _llk_unpack_configure_binary_(const DataFormat reg_data_format_src0, const DataFormat reg_data_format_src1)
{
    _llk_unpack_hw_configure_<UNP_SEL_0>(reg_data_format_src0);
    _llk_unpack_hw_configure_<UNP_SEL_1>(reg_data_format_src1);
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
 * @brief Reset the unpack thread's DEST bank tracking to bank 0 at the start of a program.
 *
 * The unpacker places sections through the UNP_DEST tile index counter, The bank lives in this thread's
 * dest_register_offset, and the placers add it (converted to tiles) to every DEST tile index. This zeroes it;
 * @ref _llk_unpack_dest_section_advance_ toggles it per section in SyncHalf.
 *
 * @note Call once per program before the first unpack-to-dest init (@ref _llk_unpack_unary_operand_to_dest_init_,
 *       @ref _llk_unpack_tilize_to_dest_init_), never from a per-op init: op writers may re-run the per-op init inside
 *       their tile loop, and a reset there would pin unpack to bank 0 while pack keeps alternating banks. Pair with
 *       @ref _llk_math_pack_sync_init_ (T1) and @ref _llk_pack_dest_init_ (T2). The one exception is the unpack_to_dest
 *       compatibility arm of @ref _llk_unpack_unary_operand_init_, which keeps the per-init reset its unported callers
 *       rely on.
 */
inline void _llk_unpack_dest_init_()
{
    _reset_dest_register_offset_();
}

/**
 * @brief Move the unpack thread to the other DEST bank after a SyncHalf section.
 *
 * Only this thread's dest_register_offset changes; the placers fold it into the next DEST tile index. No config write
 * and no drain: instructions already issued carry the old index, the toggle only affects indices computed from here on.
 *
 * @tparam EN_32BIT_DEST: Sizes the flip, bank-1 base at row 256 when true (32-bit DEST), 512 when false. Must match
 *         what the pack thread uses, or the two sides address different halves.
 */
template <bool EN_32BIT_DEST>
inline void _llk_unpack_dest_section_advance_()
{
    _update_dest_register_offset_<EN_32BIT_DEST>();
}

/**
 * @brief The current DEST bank base of this thread, in 32x32-tile units of the UNP_DEST tile index counter.
 *
 * dest_register_offset is kept in DEST rows (0, 256 or 512); a 32x32 tile spans 1 << get_dest_tile_size_log2 rows, the
 * same stride @ref _set_dst_write_addr_ uses on the math thread, so the two sides agree on where bank 1 starts. The
 * shift assumes full 32x32 tiles: the UNP_DEST tile counter steps in buffer-descriptor tiles, and a tiny-tile operand
 * is described in one-face tiles, which would put bank 1 at the wrong rows. The API init asserts the shape.
 */
inline std::uint32_t _llk_unpack_dest_bank_tile_offset_()
{
    return _get_dest_buffer_base_() >> get_dest_tile_size_log2(DstTileShape::Tile32x32);
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
    TTI_UNPACR_NOP(p_unpacr::UNP_B, 1 /*Set_Dvalid*/, 0, 0, 0, p_unpacr::UNP_SET_DVALID);
}

/**
 * @brief Returns the NOP Tensix instruction for unused unpacker engine
 *
 * Zeros the opposite Src register and sets its dvalid (used when ELWADD needs both SrcA and SrcB).
 *
 * @tparam UNP_SEL: unpacker engine in use
 */
template <std::uint32_t UNP_SEL>
constexpr std::uint32_t nop_insn_for_unused_unpacker_engine()
{
    static_assert((UNP_SEL == p_unpacr::UNP_A) || (UNP_SEL == p_unpacr::UNP_B), "UNP_SEL must be UNP_A or UNP_B");
    constexpr auto unpacr_engine = UNP_SEL == p_unpacr::UNP_A ? p_unpacr::UNP_B : p_unpacr::UNP_A;
    return TT_OP_UNPACR_NOP(unpacr_engine, 1 /*Set_Dvalid*/, 0 /*Stall_Cntrl*/, 0 /*Bank_Clr_Ctrl*/, 0 /*Src_ClrVal_Ctrl*/, 0 /*Nop_type*/);
}
