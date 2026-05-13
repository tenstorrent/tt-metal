// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "llk_pack_common.h"

using namespace ckernel;
using namespace ckernel::packer;

namespace llk_pack_internal
{
static std::uint32_t configured_num_tiles   = 1;
static std::uint32_t configured_zero_output = 0;

template <bool zero_output = false>
inline void finalize_multitile_pack_tail()
{
    constexpr std::uint32_t ZERO_OUTPUT_FLAG = zero_output ? p_pacr::P_ZERO_OUTPUT_ENABLED : p_pacr::P_ZERO_OUTPUT_DISABLED;
    // The multi-tile MOP closes tiles 0..N-2 inside the template and advances
    // the L1 destination address after each one. The last tile still needs one
    // final PACR to close the tile and restore packer row counters to the
    // normal single-tile state for the next caller.
    TTI_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, 0xf, 0, 1, 0, 1);
}
} // namespace llk_pack_internal

inline std::uint32_t _llk_pack_output_size_bytes_(const std::uint32_t pack_dst_format, const std::uint32_t datum_count)
{
    std::uint32_t packed_tile_size_bytes = SCALE_DATUM_SIZE(pack_dst_format, datum_count);

    // SCALE_DATUM_SIZE keeps one-byte-per-datum compatibility for the sub-byte
    // BFP payload formats. Pack address programming needs the real packed L1
    // footprint instead: Bfp4 payload is 2 datums/byte, Bfp2 is 4 datums/byte,
    // and all BFP formats also store one exponent byte per 16 datums
    // alongside the mantissas.
    if (pack_dst_format == to_underlying(DataFormat::Bfp4) || pack_dst_format == to_underlying(DataFormat::Bfp4_b))
    {
        packed_tile_size_bytes /= 2;
    }
    else if (pack_dst_format == to_underlying(DataFormat::Bfp2) || pack_dst_format == to_underlying(DataFormat::Bfp2_b))
    {
        packed_tile_size_bytes /= 4;
    }

    if (IS_BFP_FORMAT(pack_dst_format))
    {
        packed_tile_size_bytes += datum_count / 16;
    }

    return packed_tile_size_bytes;
}
inline std::uint32_t _llk_pack_output_addr_offset_words_(
    const std::uint32_t pack_dst_format, const std::uint32_t face_r_dim = FACE_R_DIM, const std::uint32_t num_faces = 4)
{
    const std::uint32_t tile_elements = face_r_dim * FACE_C_DIM * num_faces;
    std::uint32_t tile_size           = _llk_pack_output_size_bytes_(pack_dst_format, tile_elements);

    return tile_size >> 4;
}

template <bool untilize = false>
inline void _llk_pack_configure_addrmod_()
{
    addr_mod_pack_t {
        .y_src = {.incr = 15}, // 4-bit value so max is 15. incadcxy will increment it by 1
        .y_dst = {.incr = 1},
    }
        .set(ADDR_MOD_0);

    if constexpr (untilize)
    {
        addr_mod_pack_t {
            .y_src = {.incr = 1, .clr = 0, .cr = 1},
            .y_dst = {.incr = 1, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_1);
    }
    else
    {
        addr_mod_pack_t {
            .y_src = {.incr = 0, .clr = 1, .cr = 0},
            .y_dst = {.incr = 0, .clr = 1, .cr = 0},
            .z_src = {.incr = 0, .clr = 0},
            .z_dst = {.incr = 0, .clr = 0},
        }
            .set(ADDR_MOD_1);
    }

    addr_mod_pack_t {
        .y_src = {.incr = 0, .clr = 1, .cr = 0},
        .y_dst = {.incr = 0, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_2);
}

template <bool untilize = false, bool zero_output = false>
inline void _llk_pack_mop_config_(
    const std::uint32_t pack_dst_format,
    const std::uint32_t face_r_dim = FACE_R_DIM,
    const std::uint32_t num_faces  = 4,
    const bool partial_face        = false,
    const bool narrow_tile         = false,
    const std::uint32_t num_tiles  = 1)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    LLK_ASSERT(num_tiles >= 1, "num_tiles must be >= 1");

    if constexpr (!untilize)
    {
        if (num_tiles > 1)
        {
            LLK_ASSERT(num_faces == 4, "multi-tile pack currently supports full 4-face tiles");
            LLK_ASSERT(!partial_face, "multi-tile pack does not support partial-face tiles");
            LLK_ASSERT(!narrow_tile, "multi-tile pack does not support narrow tiles");
            TT_SETDMAREG(
                p_setdmareg::PAYLOAD_IMMEDIATE,
                _llk_pack_output_addr_offset_words_(pack_dst_format, face_r_dim, num_faces),
                p_setdmareg::MODE_IMMEDIATE,
                LO_16(p_gpr_pack::OUTPUT_ADDR_OFFSET));
        }
    }

    const std::uint32_t PACKCNT               = (partial_face && IS_BFP_FORMAT(pack_dst_format)) ? 1 : num_faces;
    constexpr std::uint32_t MEGAROW           = 1;
    constexpr std::uint32_t ZERO_OUTPUT_FLAG  = zero_output ? p_pacr::P_ZERO_OUTPUT_ENABLED : p_pacr::P_ZERO_OUTPUT_DISABLED;
    constexpr std::uint32_t MOP_INNER_LOOP    = 1;
    llk_pack_internal::configured_num_tiles   = num_tiles;
    llk_pack_internal::configured_zero_output = ZERO_OUTPUT_FLAG;

    if constexpr (!untilize)
    {
        if (partial_face && IS_BFP_FORMAT(pack_dst_format))
        {
            LLK_ASSERT(num_tiles == 1, "multi-tile partial-face BFP pack is not supported");
            constexpr std::uint32_t MOP_OUTER_LOOP = 1;
            ckernel::ckernel_template tmp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 1));
            tmp.set_start_op(TT_OP_PACR(ADDR_MOD_0, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0)); // Don't close the tile, point to the next face
            tmp.set_loop_op0(TT_OP_INCADCXY(p_setadc::PAC, 0, 0, 1, 0));                                     // Inc ch0_y+=1 (addr_mod_0 will increment by 15)
            tmp.set_loop_op1(TT_OP_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 1)); // Close the tile
            tmp.program();
        }
        else if (num_tiles == 1)
        {
            constexpr std::uint32_t MOP_OUTER_LOOP = 1;
            ckernel::ckernel_template tmp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 1));
            tmp.program();
        }
        else
        {
            // Multi-tile blocked pack is encoded as:
            // 1. start_op: pack tile 0
            // 2. outer loop (num_tiles - 1 times): advance source/destination to
            //    the next tile and commit the new L1 destination address
            // 3. end_ops: flush the new destination address into FLOP space and
            //    reset per-tile pack counters so the final explicit PACR can
            //    close the last packed tile
            ckernel::ckernel_template tmp(
                num_tiles - 1,
                1,
                TT_OP_INCADCZW(p_setadc::PAC, 0, 0, 1, 0),
                TT_OP_ADDDMAREG(p_adddmareg::REG_PLUS_REG, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR_OFFSET));
            tmp.set_start_op(TT_OP_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 1));
            tmp.set_end_ops(
                TT_OP_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR),
                TT_OP_PACR(ADDR_MOD_2, 0, 0xf, 0, 0, 1, 0));
            tmp.program();
        }
    }
    else
    {
        const std::uint32_t MOP_OUTER_LOOP = ((face_r_dim == 1) || narrow_tile) ? 1 : (face_r_dim >> 1);
        LLK_ASSERT(num_tiles == 1, "multi-tile pack is only supported for non-untilize mode");

        if ((face_r_dim == 1) || narrow_tile)
        {
            ckernel::ckernel_template tmp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_PACR(ADDR_MOD_0, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 1));
            tmp.program();
        }
        else
        {
            // Inc ch0_y+=1 (addr_mod_0 will increment by 15)
            ckernel::ckernel_template tmp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_INCADCXY(p_setadc::PAC, 0, 0, 1, 0));
            tmp.set_start_op(TT_OP_PACR(ADDR_MOD_0, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0));
            tmp.set_end_op(TT_OP_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0));
            tmp.program();
        }
    }
}

template <bool is_fp32_dest_acc_en>
inline void _llk_pack_reconfig_data_format_(
    const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    const std::uint32_t tile_size,
    const std::uint32_t face_r_dim          = FACE_R_DIM,
    const std::uint32_t num_faces           = 4,
    const bool partial_face                 = false,
    [[maybe_unused]] const bool narrow_tile = false)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    reconfig_packer_data_format<is_fp32_dest_acc_en>(pack_src_format, pack_dst_format, tile_size, face_r_dim, num_faces, partial_face);
}

inline void _llk_pack_set_fp32_dest_acc_(bool enable)
{
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
    cfg_reg_rmw_tensix<PCK_DEST_RD_CTRL_Read_32b_data_RMW>(enable);
}

template <bool is_fp32_dest_acc_en, bool untilize = false>
inline void _llk_pack_hw_configure_(
    const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    const std::uint32_t tile_size,
    const std::uint32_t face_r_dim  = FACE_R_DIM,
    const std::uint32_t num_faces   = 4,
    const bool partial_face         = false,
    const bool narrow_tile          = false,
    const std::uint32_t relu_config = 0)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    configure_pack<is_fp32_dest_acc_en, untilize>(pack_src_format, pack_dst_format, tile_size, face_r_dim, num_faces, partial_face, narrow_tile, relu_config);
}

// TODO NC: Clean up as the part of tt-metal#34587
template <bool untilize = false, bool zero_output = false, bool tilize = false /*unused*/, bool skip_addrmod_config = false>
inline void _llk_pack_init_(
    const std::uint32_t pack_dst_format,
    const std::uint32_t face_r_dim = FACE_R_DIM,
    const std::uint32_t num_faces  = 4,
    const bool partial_face        = false,
    const bool narrow_tile         = false,
    const std::uint32_t num_tiles  = 1)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    if constexpr (!skip_addrmod_config)
    {
        _llk_pack_configure_addrmod_<untilize>();
    }
    _llk_pack_mop_config_<untilize, zero_output>(pack_dst_format, face_r_dim, num_faces, partial_face, narrow_tile, num_tiles);

    set_packer_l1_offset(pack_dst_format, face_r_dim);
    const std::uint32_t face_dim   = face_r_dim * FACE_C_DIM;
    const std::uint32_t pack_x_dim = (narrow_tile || !untilize) ? face_dim : FACE_R_DIM;
    TT_SETADCXX(p_setadc::PAC, pack_x_dim - 1, 0x0);
}

inline void _llk_pack_uninit_(const std::uint32_t face_r_dim)
{
    TT_SETADCXX(p_setadc::PAC, face_r_dim * FACE_C_DIM - 1, 0x0);
}

template <DstSync Dst, bool is_fp32_dest_acc_en, bool untilize = false>
inline void _llk_pack_(const std::uint32_t tile_index, const std::uint32_t address)
{
    if constexpr (!untilize)
    {
        if (llk_pack_internal::configured_num_tiles > 1)
        {
            set_dst_write_addr(tile_index);
            LLK_ASSERT(is_valid_L1_address(address), "L1 address must be in valid L1 memory region");
            std::uint32_t new_l1_addr = (1 << 31) | address;
            TT_SETDMAREG(0, LOWER_HALFWORD(address), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
            TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR);
            // The programmed MOP performs the blocked sequence for this whole
            // call; the explicit tail below is only the final close/reset step,
            // not a second multi-tile MOP.
            mop_run(1, 1);
            if (llk_pack_internal::configured_zero_output == p_pacr::P_ZERO_OUTPUT_ENABLED)
            {
                llk_pack_internal::finalize_multitile_pack_tail<true>();
            }
            else
            {
                llk_pack_internal::finalize_multitile_pack_tail<false>();
            }
            return;
        }
    }

    set_dst_write_addr(tile_index);

    program_packer_destination(address);

    mop_run(1, 1);

    if constexpr (untilize)
    {
        TTI_PACR(ADDR_MOD_2, 0, 0xf, 0, 0, 1, 1); // close tile
    }
}
