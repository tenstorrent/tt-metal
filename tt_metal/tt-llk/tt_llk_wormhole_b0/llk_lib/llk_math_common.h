// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_defs.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "sanitizer/api.h"

using namespace ckernel::math;

/**
 * @brief Enable or disable FP32 accumulation in the destination register for both FPU and SFPU.
 *
 * @param enable: True to enable FP32 dest accumulation, false to disable.
 */
inline void _llk_math_set_fp32_dest_acc_(bool enable)
{
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(enable);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_SFPU_Fp32_enabled_RMW>(enable);
}

/**
 * @brief Configure the math (FPU) thread's ALU control registers for the given source data formats.
 *
 * Programs the source A/B ALU formats, enables INT8 math when either source is Int8/Int32, and sets FP32 dest
 * accumulation mode. Always clears debug feature bit 11 (32-bit dest mode workaround) to establish a known-good
 * baseline; bit 11 is never set by any math/SFPU path (tt-llk#1568).
 *
 * @tparam is_fp32_dest_acc_en: Enable FP32 accumulation in the destination register.
 * @param srca_data_format: Data format of source A (DataFormat enum underlying value).
 * @param srcb_data_format: Data format of source B (DataFormat enum underlying value).
 * @note srcb_data_format programs ALU_FORMAT_SPEC_REG1_SrcB, which the SFPU also reads to interpret data it loads from DEST.
 *       For SFPU work, pass a srcb_data_format whose exponent family (BF16 vs FP16) matches the data in DEST (tt-llk #951).
 */
template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_hw_configure_(const std::uint32_t srca_data_format, const std::uint32_t srcb_data_format)
{
    llk::san::math_operand_configure(srca_data_format, srcb_data_format);

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU);
    std::uint32_t int8_math_enabled = is_int8_or_int32_format(srca_data_format) || is_int8_or_int32_format(srcb_data_format);
    std::uint32_t config_data       = (srca_data_format << ALU_FORMAT_SPEC_REG0_SrcA_SHAMT) | (srcb_data_format << ALU_FORMAT_SPEC_REG1_SrcB_SHAMT) |
                                (int8_math_enabled << ALU_ACC_CTRL_INT8_math_enabled_SHAMT);
    constexpr std::uint32_t config_mask = ALU_FORMAT_SPEC_REG0_SrcA_MASK | ALU_FORMAT_SPEC_REG1_SrcB_MASK | ALU_ACC_CTRL_INT8_math_enabled_MASK;
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, 0, config_mask>(config_data);

    // Establish the no-override baseline for the SrcA/SrcB ALU format-select fields (ADDR32=0),
    // consumed by the FPU (SrcA) and SFPU (SrcB) on this thread: clear SrcA_val/override and
    // SrcB_val/override so the base REG0_SrcA/REG1_SrcB formats programmed above are used. The Dstacc
    // override fields in the same word are owned by the pack thread (configure_pack); the two writers
    // touch disjoint bits and rely on per-byte RMWCIB atomicity, so no cross-thread mutex is needed.
    constexpr std::uint32_t src_fmt_override_mask =
        ALU_FORMAT_SPEC_REG_SrcA_val_MASK | ALU_FORMAT_SPEC_REG_SrcA_override_MASK | ALU_FORMAT_SPEC_REG_SrcB_val_MASK | ALU_FORMAT_SPEC_REG_SrcB_override_MASK;
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32, 0, src_fmt_override_mask>(0);

    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(is_fp32_dest_acc_en);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_SFPU_Fp32_enabled_RMW>(is_fp32_dest_acc_en);

    // Establish the operand-driven baseline for the Src zero-substitution flag.
    _configure_default_zero_flag_state_(srca_data_format, srcb_data_format);
}

/**
 * @brief Block the math thread until the destination register is available for writing.
 *
 * @tparam Dst: Destination sync mode, values = <SyncHalf/SyncFull>
 */
template <DstSync Dst>
inline void _llk_math_wait_for_dest_available_()
{
    // These lightweight functions for sync with packer imply
    // no mode change - entire epoch is either double buffer or single buffer
    math_dest_wait();
}

/**
 * @brief Signal completion of a destination section by setting the math semaphores, flipping banks in half-sync mode.
 *
 * In SyncHalf mode this resets the per-tile sync index and flips to the other DEST half so the packer can drain
 * the just-finished half while math proceeds in the other.
 *
 * @tparam Dst: Destination sync mode, values = <SyncHalf/SyncFull>
 * @tparam is_fp32_dest_acc_en: Whether FP32 accumulation in the destination register is enabled.
 */
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

/**
 * @brief Initialize the math/pack synchronization semaphore and reset the destination section base.
 *
 * Waits for any in-flight packs to finish, then seeds the MATH_PACK semaphore (max count 1 for SyncFull, 2 for
 * SyncHalf double-buffering) and resets the DEST offset and section base.
 *
 * @tparam Dst: Destination sync mode, values = <SyncHalf/SyncFull>
 * @tparam is_fp32_dest_acc_en: Whether FP32 accumulation in the destination register is enabled.
 */
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

/**
 * @brief Reconfigure the math thread for a new source A data format.
 *
 * Programs the ALU source A format register and, by default, re-derives the INT8 math enable bit from the new
 * format so an int8-boundary reconfig cannot leave it stale (tt-metal#34499). Pass skip_int8 = true only when the
 * caller guarantees no Int8/UInt8/Int32 boundary is crossed and wants to avoid the extra RMW.
 *
 * @tparam is_fp32_dest_acc_en: Whether FP32 accumulation in the destination register is enabled (required when the new format is Int8/UInt8/Int32).
 * @tparam skip_int8: Skip re-deriving the INT8 math enable bit from the new format.
 * @param srca_data_format: New data format of source A (DataFormat enum underlying value).
 */
template <bool is_fp32_dest_acc_en, bool skip_int8 = false>
inline void _llk_math_reconfig_data_format_srca_(const std::uint32_t srca_data_format)
{
    llk::san::math_operand_configure<true>(srca_data_format, llk::san::IGNORE);

    if constexpr (!skip_int8)
    {
        LLK_ASSERT(
            is_fp32_dest_acc_en || !is_int8_or_int32_format(srca_data_format),
            "Reconfiguring math to/from Int8/UInt8/Int32 formats requires FP32 Dest mode enabled");
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU);
        std::uint32_t int8_math_enabled = is_int8_or_int32_format(srca_data_format);
        std::uint32_t config_data       = (srca_data_format << ALU_FORMAT_SPEC_REG0_SrcA_SHAMT) | (int8_math_enabled << ALU_ACC_CTRL_INT8_math_enabled_SHAMT);
        constexpr std::uint32_t config_mask = ALU_FORMAT_SPEC_REG0_SrcA_MASK | ALU_ACC_CTRL_INT8_math_enabled_MASK;
        cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, 0, config_mask>(config_data);
    }
    else
    {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU);
        cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(srca_data_format);
    }

    // Re-establish the operand-driven baseline (clears stale op-state) for the new SrcA format.
    _configure_default_zero_flag_state_(srca_data_format, src_zero_flag_srcb_fmt);
}

/**
 * @brief Reconfigure the math thread for a new source B data format.
 *
 * Programs the ALU source B format register and, by default, re-derives the INT8 math enable bit from the new
 * format so an int8-boundary reconfig cannot leave it stale (tt-metal#34499). Pass skip_int8 = true only when the
 * caller guarantees no Int8/UInt8/Int32 boundary is crossed and wants to avoid the extra RMW.
 *
 * @tparam is_fp32_dest_acc_en: Whether FP32 accumulation in the destination register is enabled (required when the new format is Int8/UInt8/Int32).
 * @tparam skip_int8: Skip re-deriving the INT8 math enable bit from the new format.
 * @param srcb_data_format: New data format of source B (DataFormat enum underlying value).
 */
template <bool is_fp32_dest_acc_en, bool skip_int8 = false>
inline void _llk_math_reconfig_data_format_srcb_(const std::uint32_t srcb_data_format)
{
    llk::san::math_operand_configure<true>(llk::san::IGNORE, srcb_data_format);

    if constexpr (!skip_int8)
    {
        LLK_ASSERT(
            is_fp32_dest_acc_en || !is_int8_or_int32_format(srcb_data_format),
            "Reconfiguring math to/from Int8/UInt8/Int32 formats requires FP32 Dest mode enabled");
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU);
        std::uint32_t int8_math_enabled = is_int8_or_int32_format(srcb_data_format);
        std::uint32_t config_data       = (srcb_data_format << ALU_FORMAT_SPEC_REG1_SrcB_SHAMT) | (int8_math_enabled << ALU_ACC_CTRL_INT8_math_enabled_SHAMT);
        constexpr std::uint32_t config_mask = ALU_FORMAT_SPEC_REG1_SrcB_MASK | ALU_ACC_CTRL_INT8_math_enabled_MASK;
        cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG1_SrcB_ADDR32, 0, config_mask>(config_data);
    }
    else
    {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU);
        cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG1_SrcB_RMW>(srcb_data_format);
    }

    // Re-establish the operand-driven baseline (clears stale op-state) for the new SrcB format.
    _configure_default_zero_flag_state_(src_zero_flag_srca_fmt, srcb_data_format);
}

/**
 * @brief Reconfigure the math thread for new source A and source B data formats.
 *
 * Programs the ALU source A and source B format registers and, by default, re-derives the INT8 math enable bit
 * from both new formats so an int8-boundary reconfig cannot leave it stale (tt-metal#34499). Pass skip_int8 = true
 * only when the caller guarantees no Int8/UInt8/Int32 boundary is crossed and wants to avoid the extra RMW.
 *
 * @tparam is_fp32_dest_acc_en: Whether FP32 accumulation in the destination register is enabled (required when either new format is Int8/UInt8/Int32).
 * @tparam skip_int8: Skip re-deriving the INT8 math enable bit from the new formats.
 * @param srca_data_format: New data format of source A (DataFormat enum underlying value).
 * @param srcb_data_format: New data format of source B (DataFormat enum underlying value).
 */
template <bool is_fp32_dest_acc_en, bool skip_int8 = false>
inline void _llk_math_reconfig_data_format_(const std::uint32_t srca_data_format, const std::uint32_t srcb_data_format)
{
    llk::san::math_operand_configure<true>(srca_data_format, srcb_data_format);

    if constexpr (!skip_int8)
    {
        LLK_ASSERT(
            is_fp32_dest_acc_en || !(is_int8_or_int32_format(srca_data_format) || is_int8_or_int32_format(srcb_data_format)),
            "Reconfiguring math to/from Int8/UInt8/Int32 formats requires FP32 Dest mode enabled");
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU);
        std::uint32_t int8_math_enabled = is_int8_or_int32_format(srca_data_format) || is_int8_or_int32_format(srcb_data_format);
        std::uint32_t config_data       = (srca_data_format << ALU_FORMAT_SPEC_REG0_SrcA_SHAMT) | (srcb_data_format << ALU_FORMAT_SPEC_REG1_SrcB_SHAMT) |
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

    // Re-establish the operand-driven baseline (clears stale op-state) for the new formats.
    _configure_default_zero_flag_state_(srca_data_format, srcb_data_format);
}

/**
 * @brief Read the FPU sticky special-value flags (e.g. NaN/Inf detection) accumulated since the last clear.
 *
 * @return Raw value of the FPU sticky bits register.
 */
inline std::uint32_t _llk_math_get_compute_special_value_flags_()
{
    return reg_read(RISCV_DEBUG_REG_FPU_STICKY_BITS);
}

/**
 * @brief Clear the FPU sticky special-value flags register.
 *
 * @note Read with @ref _llk_math_get_compute_special_value_flags_ to observe flags accumulated after this clear.
 */
inline void _llk_math_clear_compute_special_value_flags_()
{
    reg_write(RISCV_DEBUG_REG_FPU_STICKY_BITS, 0);
}
