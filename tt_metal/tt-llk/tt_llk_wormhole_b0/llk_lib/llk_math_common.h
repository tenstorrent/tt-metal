// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_defs.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "cmath_common.h"

using namespace ckernel::math;

inline void _llk_math_dbg_feature_disable_()
{
    reg_write(RISCV_DEBUG_REG_DBG_FEATURE_DISABLE, 1 << 11); // Set debug feature disable bit 11
                                                             // workaround for bug tenstorrent/budabackend#1372
}

inline void _llk_math_dbg_feature_enable_()
{
    tensix_sync();
    reg_write(RISCV_DEBUG_REG_DBG_FEATURE_DISABLE, 0); // Clear debug feature disable bit 11
                                                       // workaround for bug tenstorrent/budabackend#1372
}

inline void _llk_math_set_fp32_dest_acc_(bool enable)
{
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(enable);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_SFPU_Fp32_enabled_RMW>(enable);
}

template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_hw_configure_(const std::uint32_t srca_data_format, const std::uint32_t srcb_data_format)
{
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU);
    std::uint32_t int8_math_enabled = (masked_data_format(srca_data_format) == to_underlying(DataFormat::Int8)) ||
                                      (masked_data_format(srcb_data_format) == to_underlying(DataFormat::Int8)) ||
                                      (srca_data_format == to_underlying(DataFormat::Int32)) || (srcb_data_format == to_underlying(DataFormat::Int32));
    std::uint32_t config_data = (srca_data_format << ALU_FORMAT_SPEC_REG0_SrcA_SHAMT) | (srcb_data_format << ALU_FORMAT_SPEC_REG1_SrcB_SHAMT) |
                                (int8_math_enabled << ALU_ACC_CTRL_INT8_math_enabled_SHAMT);
    constexpr std::uint32_t config_mask = ALU_FORMAT_SPEC_REG0_SrcA_MASK | ALU_FORMAT_SPEC_REG1_SrcB_MASK | ALU_ACC_CTRL_INT8_math_enabled_MASK;
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, 0, config_mask>(config_data);

    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(is_fp32_dest_acc_en);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_SFPU_Fp32_enabled_RMW>(is_fp32_dest_acc_en);

    // Workaround for HW bugs:
    // budabackend#1948: int32 dest and movd2a/b with int8 srcA/B
    // budabackend#1948: fp32 dest and movd2a/b with UInt16 srcA/B
    bool uint16_with_fp32_dest =
        is_fp32_dest_acc_en && ((srca_data_format == to_underlying(DataFormat::UInt16)) || (srcb_data_format == to_underlying(DataFormat::UInt16)));

    if (int8_math_enabled || uint16_with_fp32_dest)
    {
        _llk_math_dbg_feature_disable_();
    }
}

template <DstSync Dst>
inline void _llk_math_wait_for_dest_available_()
{
    // These lightweight functions for sync with packer imply
    // no mode change - entire epoch is either double buffer or single buffer
    math_dest_wait();
}

template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_dest_section_done_()
{
    set_math_semaphores();
    if constexpr (Dst == DstSync::SyncHalf)
    {
        math_sync_tile_dst_index = 0;
        dest_section_flip();
    }
}

template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_pack_sync_init_()
{
    tensix_sync();
    while (semaphore_read(semaphore::MATH_PACK) > 0)
    {
    }; // Wait for previous packs to finish before claiming all dest
    if constexpr (Dst == DstSync::SyncFull)
    {
        TTI_SEMINIT(1, 0, p_stall::SEMAPHORE_1);
        reset_dest_offset_id();
        set_dest_section_base<StartZero>();
    }
    else
    {
        static_assert(Dst == DstSync::SyncHalf);
        TTI_SEMINIT(2, 0, p_stall::SEMAPHORE_1);
        reset_dest_offset_id();
        set_dest_section_base<StartZero>();
    }
}

template <bool is_fp32_dest_acc_en, bool to_from_int8 = false>
inline void _llk_math_reconfig_data_format_srca_(const std::uint32_t srca_data_format)
{
    if constexpr (to_from_int8)
    {
        static_assert(is_fp32_dest_acc_en, "Reconfiguring math to/from Int8 formats requires FP32 Dest mode enabled");
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU);
        std::uint32_t int8_math_enabled =
            (masked_data_format(srca_data_format) == to_underlying(DataFormat::Int8)) || (srca_data_format == to_underlying(DataFormat::Int32));
        std::uint32_t config_data = (srca_data_format << ALU_FORMAT_SPEC_REG0_SrcA_SHAMT) | (int8_math_enabled << ALU_ACC_CTRL_INT8_math_enabled_SHAMT);
        constexpr std::uint32_t config_mask = ALU_FORMAT_SPEC_REG0_SrcA_MASK | ALU_ACC_CTRL_INT8_math_enabled_MASK;
        cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, 0, config_mask>(config_data);
    }
    else
    {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU);
        cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(srca_data_format);
    }
}

template <bool is_fp32_dest_acc_en, bool to_from_int8 = false>
inline void _llk_math_reconfig_data_format_srcb_(const std::uint32_t srcb_data_format)
{
    if constexpr (to_from_int8)
    {
        static_assert(is_fp32_dest_acc_en, "Reconfiguring math to/from Int8 formats requires FP32 Dest mode enabled");
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU);
        std::uint32_t int8_math_enabled =
            (masked_data_format(srcb_data_format) == to_underlying(DataFormat::Int8)) || (srcb_data_format == to_underlying(DataFormat::Int32));
        std::uint32_t config_data = (srcb_data_format << ALU_FORMAT_SPEC_REG1_SrcB_SHAMT) | (int8_math_enabled << ALU_ACC_CTRL_INT8_math_enabled_SHAMT);
        constexpr std::uint32_t config_mask = ALU_FORMAT_SPEC_REG1_SrcB_MASK | ALU_ACC_CTRL_INT8_math_enabled_MASK;
        cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG1_SrcB_ADDR32, 0, config_mask>(config_data);
    }
    else
    {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU);
        cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG1_SrcB_RMW>(srcb_data_format);
    }
}

template <bool is_fp32_dest_acc_en, bool to_from_int8 = false>
inline void _llk_math_reconfig_data_format_(const std::uint32_t srca_data_format, const std::uint32_t srcb_data_format)
{
    if constexpr (to_from_int8)
    {
        static_assert(is_fp32_dest_acc_en, "Reconfiguring math to/from Int8 formats requires FP32 Dest mode enabled");
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU);
        std::uint32_t int8_math_enabled = (masked_data_format(srca_data_format) == to_underlying(DataFormat::Int8)) ||
                                          (masked_data_format(srcb_data_format) == to_underlying(DataFormat::Int8)) ||
                                          (srca_data_format == to_underlying(DataFormat::Int32)) || (srcb_data_format == to_underlying(DataFormat::Int32));
        std::uint32_t config_data = (srca_data_format << ALU_FORMAT_SPEC_REG0_SrcA_SHAMT) | (srcb_data_format << ALU_FORMAT_SPEC_REG1_SrcB_SHAMT) |
                                    (int8_math_enabled << ALU_ACC_CTRL_INT8_math_enabled_SHAMT);
        constexpr std::uint32_t config_mask = ALU_FORMAT_SPEC_REG0_SrcA_MASK | ALU_FORMAT_SPEC_REG1_SrcB_MASK | ALU_ACC_CTRL_INT8_math_enabled_MASK;
        cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, 0, config_mask>(config_data);
    }
    else
    {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU);
        std::uint32_t config_data           = (srca_data_format << ALU_FORMAT_SPEC_REG0_SrcA_SHAMT) | (srcb_data_format << ALU_FORMAT_SPEC_REG1_SrcB_SHAMT);
        constexpr std::uint32_t config_mask = ALU_FORMAT_SPEC_REG0_SrcA_MASK | ALU_FORMAT_SPEC_REG1_SrcB_MASK;
        cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, 0, config_mask>(config_data);
    }
}

inline std::uint32_t _llk_math_get_compute_special_value_flags_()
{
    return reg_read(RISCV_DEBUG_REG_FPU_STICKY_BITS);
}

inline void _llk_math_clear_compute_special_value_flags_()
{
    reg_write(RISCV_DEBUG_REG_FPU_STICKY_BITS, 0);
}
