// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "cmath_common.h"
#include "llk_defs.h"
using namespace ckernel;
using namespace ckernel::trisc;
using namespace ckernel::math;

static DataFormatConfigSet data_format_config_set = DataFormatConfigSet::UNCONFIGURED;

/**
 * @brief Sets up ALU formats for math destination register
 * @tparam EN_IMPLIED_MATH_FORMAT: If set to true, will imply math dest format
 * from SrcA reg format
 * @tparam EN_FP32_MATH_FORMAT: Set to true to use math dest in Float32
 * otherwise default behaviour is Float16/Float16_b depending on input
 * format exponent width
 * @tparam EN_INT32_MATH_FORMAT: Set to true to use math dest in Int32
 * otherwise default behaviour is Float16/Float16_b depending on input
 * format exponent width
 * @param srcA_format: Input srcA format, used to set ALU configs if not implied math format
 * values = Dataformat enum, ex: <Float16/Float16_b/Tf32/Int8/Int16/UInt8>
 * @param srcB_format: Input srcB format, used to set ALU configs if not implied math format
 * values = Dataformat enum, ex: <Float16/Float16_b/Tf32/Int8/Int16/UInt8>
 */
template <bool EN_IMPLIED_MATH_FORMAT, bool EN_FP32_MATH_FORMAT, bool EN_INT32_MATH_FORMAT>
inline void _llk_math_srcAB_hw_configure_(DataFormat srcA_format, DataFormat srcB_format)
{
    // Turn on automatic Tensix-TRISC synchronization
    // RT: This is turned on by default by HW, this should be removed
    set_ttsync_enables<TRACK_ALL>(TRISC_ID);

    static_assert(!(EN_FP32_MATH_FORMAT && EN_INT32_MATH_FORMAT), "Cannot have Int32 dest & Float32 dest at the same time");

    // Set implied math dest format mode
    cfg[DISABLE_IMPLIED_SRCA_FMT_SEC0_Base_ADDR32 + TRISC_ID] = !EN_IMPLIED_MATH_FORMAT;
    cfg[DISABLE_IMPLIED_SRCB_FMT_SEC0_Base_ADDR32 + TRISC_ID] = !EN_IMPLIED_MATH_FORMAT;

    std::uint8_t SRCA_FORMAT_MASKED = masked_data_format(to_underlying(srcA_format));
    std::uint8_t SRCB_FORMAT_MASKED = masked_data_format(to_underlying(srcB_format));

    alu_config_u alu_config;
    for (std::uint32_t i = 0; i < NUM_WORDS_ALU_FORMAT; i++)
    {
        alu_config.val[i] = 0;
    }

    if constexpr (!EN_IMPLIED_MATH_FORMAT)
    {
        // Set ALU SrcA format since it is not implied
        // If input format has exp_width == 5, the math dest set to Float16
        // else input format has exp_width == 8, the math dest set to Float16_b
        alu_config.f.ALU_FORMAT_SPEC_REG_SrcA_val      = SRCA_FORMAT_MASKED;
        alu_config.f.ALU_FORMAT_SPEC_REG_SrcA_override = 0x1;
        alu_config.f.ALU_FORMAT_SPEC_REG_SrcB_val      = SRCB_FORMAT_MASKED;
        alu_config.f.ALU_FORMAT_SPEC_REG_SrcB_override = 0x1;

        // RT: Since SrcA & SrcB need to match exponent widths, can set them the same for now
        // Check with HW team if different mixes between Src registers are allowed
        alu_config.f.ALU_FORMAT_SPEC_REG0_SrcA = SRCA_FORMAT_MASKED;
        alu_config.f.ALU_FORMAT_SPEC_REG1_SrcB = SRCB_FORMAT_MASKED;
    }

    alu_config.f.ALU_ACC_CTRL_Fp32_enabled      = EN_FP32_MATH_FORMAT;
    alu_config.f.ALU_ACC_CTRL_SFPU_Fp32_enabled = EN_FP32_MATH_FORMAT;
    alu_config.f.ALU_ACC_CTRL_INT8_math_enabled = EN_INT32_MATH_FORMAT;

    for (std::uint32_t i = 0; i < NUM_WORDS_ALU_FORMAT; i++)
    {
        cfg[ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32 + i] = alu_config.val[i];
    }

    data_format_config_set = DataFormatConfigSet::DEFAULT;
}

/**
 * @brief Sets up ALU formats for math destination register, specifically for upk to dest
 * @tparam EN_IMPLIED_MATH_FORMAT: If set to true, will imply math dest format
 * from SrcA reg format
 * @tparam EN_FP32_MATH_FORMAT: Set to true to use math dest in Float32
 * otherwise default behaviour is Float16/Float16_b depending on input
 * format exponent width
 * @tparam EN_INT32_MATH_FORMAT: Set to true to use math dest in Int32
 * otherwise default behaviour is Float16/Float16_b depending on input
 * format exponent width
 */
template <bool EN_IMPLIED_MATH_FORMAT, bool EN_FP32_MATH_FORMAT, bool EN_INT32_MATH_FORMAT>
inline void _llk_math_upk_to_dest_hw_configure_()
{
    // Set implied math dest format mode
    cfg[DISABLE_IMPLIED_SRCA_FMT_SEC0_Base_ADDR32 + TRISC_ID] = !EN_IMPLIED_MATH_FORMAT;

    alu_config_u alu_config;
    for (std::uint32_t i = 0; i < NUM_WORDS_ALU_FORMAT; i++)
    {
        alu_config.val[i] = 0;
    }

    // Program DEST fmt
    alu_config.f.ALU_ACC_CTRL_Fp32_enabled      = EN_FP32_MATH_FORMAT;
    alu_config.f.ALU_ACC_CTRL_SFPU_Fp32_enabled = EN_FP32_MATH_FORMAT;
    alu_config.f.ALU_ACC_CTRL_INT8_math_enabled = EN_INT32_MATH_FORMAT;

    for (std::uint32_t i = 0; i < NUM_WORDS_ALU_FORMAT; i++)
    {
        cfg[ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32 + i] = alu_config.val[i];
    }
}

/**
 * @brief Determines whether the source register format and Float32 destination register format are a supported combination
 *
 * @param src_reg_fmt: The source register format
 */
inline bool _is_src_fmt_fp32_dest_compatible_(const DataFormat src_reg_fmt)
{
    return src_reg_fmt == DataFormat::Float16_b || src_reg_fmt == DataFormat::Float16 || src_reg_fmt == DataFormat::Tf32 ||
           src_reg_fmt == DataFormat::MxFp4_2x_A || src_reg_fmt == DataFormat::MxFp4_2x_B;
}

/**
 * @brief Determines whether the source register format and Int32 destination register format are a supported combination
 *
 * @param src_reg_fmt: The source register format
 */
inline bool _is_src_fmt_int32_dest_compatible_(const DataFormat src_reg_fmt)
{
    return src_reg_fmt == DataFormat::Int8 || src_reg_fmt == DataFormat::UInt8 || src_reg_fmt == DataFormat::Int8_2x || src_reg_fmt == DataFormat::UInt8_2x;
}

/**
 * @brief Sets up ALU formats
 * @tparam EN_IMPLIED_MATH_FORMAT: If set to true, will imply math dest format
 * from SrcA reg format
 * @tparam EN_FP32_DEST_FORMAT: Set to true to use math dest in Float32
 * otherwise default behaviour is Float16/Float16_b depending on input
 * format exponent width
 * @tparam EN_INT32_DEST_FORMAT: Set to true to use math dest in Int32
 * otherwise default behaviour is Float16/Float16_b depending on input
 * format exponent width
 * @param srcA_format: Input srcA format, used to set ALU configs if not implied math format
 * values = Dataformat enum, ex: <Float16/Float16_b/Tf32/Int8/Int16/UInt8>
 * @param srcB_format: Input srcB format, used to set ALU configs if not implied math format
 * values = Dataformat enum, ex: <Float16/Float16_b/Tf32/Int8/Int16/UInt8>
 */
template <bool EN_IMPLIED_MATH_FORMAT, bool EN_FP32_DEST_FORMAT, bool EN_INT32_DEST_FORMAT>
inline void _configure_alu_formats_(DataFormat srcA_format, DataFormat srcB_format)
{
    cfg[DISABLE_IMPLIED_SRCA_FMT_SEC0_Base_ADDR32 + TRISC_ID] = !EN_IMPLIED_MATH_FORMAT;
    cfg[DISABLE_IMPLIED_SRCB_FMT_SEC0_Base_ADDR32 + TRISC_ID] = !EN_IMPLIED_MATH_FORMAT;

    if constexpr (!EN_IMPLIED_MATH_FORMAT)
    {
        std::uint8_t SRCA_FORMAT_MASKED = masked_data_format(to_underlying(srcA_format));
        std::uint8_t SRCB_FORMAT_MASKED = masked_data_format(to_underlying(srcB_format));

        cfg_rmw(ALU_FORMAT_SPEC_REG_SrcA_val_RMW, SRCA_FORMAT_MASKED);
        cfg_rmw(ALU_FORMAT_SPEC_REG_SrcA_override_RMW, 0x1);
        cfg_rmw(ALU_FORMAT_SPEC_REG_SrcB_val_RMW, SRCB_FORMAT_MASKED);
        cfg_rmw(ALU_FORMAT_SPEC_REG_SrcB_override_RMW, 0x1);

        cfg_rmw(ALU_FORMAT_SPEC_REG0_SrcA_RMW, SRCA_FORMAT_MASKED);
        cfg_rmw(ALU_FORMAT_SPEC_REG1_SrcB_RMW, SRCB_FORMAT_MASKED);
    }

    cfg_rmw(ALU_ACC_CTRL_Fp32_enabled_RMW, EN_FP32_DEST_FORMAT);
    cfg_rmw(ALU_ACC_CTRL_SFPU_Fp32_enabled_RMW, EN_FP32_DEST_FORMAT);
    cfg_rmw(ALU_ACC_CTRL_INT8_math_enabled_RMW, EN_INT32_DEST_FORMAT);
}

/**
 * @brief Sets up default ALU data format state
 * @tparam EN_IMPLIED_MATH_FORMAT: If set to true, will imply math dest format
 * from SrcA reg format
 * @tparam EN_32BIT_DEST: Set to true to use 32bit math dest in Float32 or Int32 format
 * @param srcA_format: Input srcA format, used to set ALU configs if not implied math format
 * values = Dataformat enum, ex: <Float16/Float16_b/Tf32/Int8/Int16/UInt8>
 * @param srcB_format: Input srcB format, used to set ALU configs if not implied math format
 * values = Dataformat enum, ex: <Float16/Float16_b/Tf32/Int8/Int16/UInt8>
 */
template <bool EN_IMPLIED_MATH_FORMAT, bool EN_32BIT_DEST>
inline void _configure_default_data_format_state_(DataFormat srcA_format, DataFormat srcB_format)
{
    if (data_format_config_set == DataFormatConfigSet::DEFAULT)
    {
        return;
    }

    TTI_STALLWAIT(p_stall::STALL_CFG, 0, p_stall::WAIT_SFPU, p_stall::MATH);

    const bool EN_FP32_DEST_FORMAT  = _is_src_fmt_fp32_dest_compatible_(srcA_format) && _is_src_fmt_fp32_dest_compatible_(srcB_format) && EN_32BIT_DEST;
    const bool EN_INT32_DEST_FORMAT = _is_src_fmt_int32_dest_compatible_(srcA_format) && _is_src_fmt_int32_dest_compatible_(srcB_format) && EN_32BIT_DEST;
    if (EN_FP32_DEST_FORMAT)
    {
        _configure_alu_formats_<EN_IMPLIED_MATH_FORMAT, true /* EN_FP32_DEST_FORMAT */, false /* EN_INT32_DEST_FORMAT */>(srcA_format, srcB_format);
    }
    else if (EN_INT32_DEST_FORMAT)
    {
        _configure_alu_formats_<EN_IMPLIED_MATH_FORMAT, false /* EN_FP32_DEST_FORMAT */, true /* EN_INT32_DEST_FORMAT */>(srcA_format, srcB_format);
    }
    else
    {
        _configure_alu_formats_<EN_IMPLIED_MATH_FORMAT, false /* EN_FP32_DEST_FORMAT */, false /* EN_INT32_DEST_FORMAT */>(srcA_format, srcB_format);
    }

    data_format_config_set = DataFormatConfigSet::DEFAULT;
}

/**
 * @brief Sets up MOV SRC2DST 32bit ops ALU data format state
 * Used for transpose dest operations when Int32 or Fp32 dest is used.
 * Implied math format is disabled, and Int32 dest requires opposite settings that what is usually set for Int32 dest.
 * Float32 and Int32 can be set as the srcA and srcB formats.
 * @param srcA_format: Input srcA format, used to set ALU configs
 * values = Dataformat enum, ex: <Float16/Float16_b/Tf32/Float32/Int8/Int16/UInt8/Int32>
 * @param srcB_format: Input srcB format, used to set ALU configs
 * values = Dataformat enum, ex: <Float16/Float16_b/Tf32/Float32/Int8/Int16/UInt8/Int32>
 */
inline void _configure_mov_src2dst_32bit_ops_data_format_state_(DataFormat srcA_format, DataFormat srcB_format)
{
    if (data_format_config_set == DataFormatConfigSet::MOV_SRC2DST_32BIT_OPS)
    {
        return;
    }
    TTI_STALLWAIT(p_stall::STALL_CFG, 0, p_stall::WAIT_SFPU, p_stall::MATH);

    _configure_alu_formats_<false /* EN_IMPLIED_MATH_FORMAT */, true /* EN_FP32_DEST_FORMAT */, false /* EN_INT32_DEST_FORMAT */>(srcA_format, srcB_format);

    data_format_config_set = DataFormatConfigSet::MOV_SRC2DST_32BIT_OPS;
}

/**
 * @brief Sets the dest dvalid for a FPU/SFPU
 * @tparam SET_DEST_DVALID: which client to set data valid for, values = <p_cleardvalid::FPU/SFPU>
 **/
template <std::uint8_t SET_DEST_DVALID, DstSync DST>
inline void _llk_math_set_dvalid_()
{
    static_assert(SET_DEST_DVALID == p_cleardvalid::FPU || SET_DEST_DVALID == p_cleardvalid::SFPU, "Can only set dest dvalid for FPU and SFPU");

    TTI_STALLWAIT(p_stall::STALL_MATH, 0, 0, p_stall::WAIT_SFPU);
    TTI_CLEARDVALID(0, 0, 0, 0, SET_DEST_DVALID, 0);
    if constexpr (DST == DstSync::SyncFull)
    {
        // For DstSync::SyncFull issue a CLEARDVALID instruction for dest bank1 as well in order to use full dest register
        // Reset dest bank id to 0 for the given dest client to ensure SyncFull starts from bank0
        TTI_CLEARDVALID(0, 0, 0, SET_DEST_DVALID, SET_DEST_DVALID, 0);
    }
}

/**
 * All the following functions are added to enable Math <-> Pack synchronization
 * on destination register due to dest dvalid issue.
 *
 * The following functions should be removed once the above issue is resolved
 */
template <DstSync DST>
inline void _llk_math_pack_sync_init_()
{
    static_assert(DST == DstSync::SyncFull || DST == DstSync::SyncHalf, "Only Dest Sync Half and Full are supported");

    // Wait for previous packs to finish before claiming all dest
    while (semaphore_read(semaphore::MATH_PACK) > 0)
    {
    };

    _reset_dest_register_offset_();
    _set_dest_section_base_<TRISC_ID>(_get_dest_buffer_base_());

    constexpr std::uint32_t num_sem = (DST == DstSync::SyncFull) ? 1 : 2;
    TTI_SEMINIT(num_sem, 0, 0, p_stall::SEMAPHORE_1);
}

inline void _llk_math_wait_for_dest_available_()
{
    TTI_SEMWAIT(p_stall::STALL_MATH | p_stall::STALL_SFPU | p_stall::STALL_SYNC, p_stall::STALL_ON_MAX, 0, semaphore::t6_sem(semaphore::MATH_PACK));
}

template <DstSync DST, bool EN_32BIT_DEST>
inline void _llk_math_dest_section_done_()
{
    t6_semaphore_post<p_stall::MATH, p_stall::WAIT_SFPU>(semaphore::MATH_PACK);
    if constexpr (DST == DstSync::SyncHalf)
    {
        _update_dest_register_offset_<EN_32BIT_DEST>();
        std::uint32_t base_addr = _get_dest_buffer_base_();
        TTI_STALLWAIT(p_stall::STALL_CFG, 0, p_stall::MATH, p_stall::WAIT_SFPU);
        _set_dest_section_base_<TRISC_ID>(base_addr);
    }
}
