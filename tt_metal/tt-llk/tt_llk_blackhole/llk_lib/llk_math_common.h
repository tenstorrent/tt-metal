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
 * @brief Set debug feature disable bit 11 to work around an FPU HW bug.
 *
 * @note Workaround for bug tt-metal#46219. Paired with @ref _llk_math_dbg_feature_enable_ to restore.
 */
inline void _llk_math_dbg_feature_disable_()
{
    reg_write(RISCV_DEBUG_REG_DBG_FEATURE_DISABLE, 1 << 11); // Set debug feature disable bit 11
                                                             // workaround for bug tt-metal#46219
}

/**
 * @brief Clear debug feature disable bit 11, restoring default FPU behavior.
 *
 * @note Reverses @ref _llk_math_dbg_feature_disable_ (workaround for bug tt-metal#46219). Issues a tensix_sync() first.
 */
inline void _llk_math_dbg_feature_enable_()
{
    tensix_sync();
    reg_write(RISCV_DEBUG_REG_DBG_FEATURE_DISABLE, 0); // Clear debug feature disable bit 11
                                                       // workaround for bug tt-metal#46219
}

/**
 * @brief Enable or disable FP32 accumulation in the destination register for both FPU and SFPU.
 *
 * @param enable: True to enable FP32 dest accumulation, false to disable.
 */
inline void _llk_math_set_fp32_dest_acc_(bool enable)
{
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(enable);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_SFPU_Fp32_enabled_RMW>(enable);
}

/**
 * @brief Configure the math (FPU) thread's ALU control registers for the given source data formats.
 *
 * Sets ZEROACC bank auto-detect, enables INT8 math when either source is Int8/Int32, and programs FP32 dest
 * accumulation mode. Applies HW-bug workarounds (disables debug feature bit 11) for INT8 math and for the
 * UInt16-with-FP32-dest combination.
 *
 * @tparam is_fp32_dest_acc_en: Enable FP32 accumulation in the destination register.
 * @param srca_data_format: Data format of source A (DataFormat enum underlying value).
 * @param srcb_data_format: Data format of source B (DataFormat enum underlying value).
 * @note May disable debug feature bit 11 via @ref _llk_math_dbg_feature_disable_ for INT8/UInt16 workarounds (budabackend#1948).
 * @note The SFPU reads ALU_FORMAT_SPEC_REG1_SrcB (the SrcB ALU format, programmed unpack-side on Blackhole, not here) to
 *       interpret data it loads from DEST. For SFPU work, ensure that format's exponent family (BF16 vs FP16) matches
 *       the data in DEST (tt-llk #951).
 */
template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_hw_configure_(const std::uint32_t srca_data_format, const std::uint32_t srcb_data_format)
{
    // LLK sanitizer hooks
    llk::san::math_operand_configure(srca_data_format, srcb_data_format);

    // Configure ZEROACC to auto-detect destination bank (non-legacy mode).
    cfg_reg_rmw_tensix<DEST_ACCESS_CFG_zeroacc_absolute_tile_mode_RMW>(0);
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH);
    std::uint32_t int8_math_enabled = (masked_data_format(srca_data_format) == ckernel::to_underlying(DataFormat::Int8)) ||
                                      (masked_data_format(srcb_data_format) == ckernel::to_underlying(DataFormat::Int8)) ||
                                      (srca_data_format == ckernel::to_underlying(DataFormat::Int32)) ||
                                      (srcb_data_format == ckernel::to_underlying(DataFormat::Int32));
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_INT8_math_enabled_RMW>(int8_math_enabled);

    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(is_fp32_dest_acc_en);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_SFPU_Fp32_enabled_RMW>(is_fp32_dest_acc_en);

    // Establish the operand-driven baseline for the Src zero-substitution flag.
    _configure_default_zero_flag_state_(srca_data_format, srcb_data_format);
}

/**
 * @brief Enable or disable the destination read-address remap (stride-of-16) used by untilize mode.
 *
 * Waits for all in-flight DEST accesses and pending packs to finish before toggling the remap_addrs and
 * swizzle_32b config bits, since changing them mid-access would corrupt reads.
 *
 * @param remap_enable: True to enable the stride-of-16 remap (untilize), false to restore linear addressing.
 */
inline void _llk_math_reconfig_remap_(const bool remap_enable)
{
    // Need to wait for all DEST accesses to be finished before changing
    // remap_addrs and swizzle_32b bits
    tensix_sync();
    while (semaphore_read(semaphore::MATH_PACK) > 0)
    {
    }; // Wait for previous packs to finish before claiming all dest

    // Untilize mode needs dest read access with a stride of 16
    // Following bits are needed for enabling stride of 16
    cfg_reg_rmw_tensix<DEST_ACCESS_CFG_remap_addrs_RMW>(remap_enable);
    cfg_reg_rmw_tensix<DEST_ACCESS_CFG_swizzle_32b_RMW>(remap_enable);
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

// Following functions do not need to program ALU_FORMAT_SPEC_REG0_SrcA/ALU_FORMAT_SPEC_REG1_SrcB
// for blackhole since ALU format is inferred
/**
 * @brief Reconfigure the math thread for a new source A data format.
 *
 * On Blackhole the ALU source format is inferred, so this is a no-op unless the reconfiguration crosses an
 * Int8/Int32 boundary (to_from_int8), in which case it re-evaluates and programs the INT8 math enable bit.
 *
 * @tparam is_fp32_dest_acc_en: Whether FP32 accumulation in the destination register is enabled (required when to_from_int8 is set).
 * @tparam to_from_int8: Set when the reconfiguration switches to or from an Int8/Int32 format.
 * @param srca_data_format: New data format of source A (DataFormat enum underlying value).
 */
template <bool is_fp32_dest_acc_en, bool to_from_int8 = false>
inline void _llk_math_reconfig_data_format_srca_(const std::uint32_t srca_data_format)
{
    llk::san::math_operand_configure<true>(srca_data_format, llk::san::IGNORE);

    if constexpr (to_from_int8)
    {
        static_assert(is_fp32_dest_acc_en, "Reconfiguring math to/from Int8 formats requires FP32 Dest mode enabled");
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH);
        std::uint32_t int8_math_enabled = (masked_data_format(srca_data_format) == ckernel::to_underlying(DataFormat::Int8)) ||
                                          (srca_data_format == ckernel::to_underlying(DataFormat::Int32));
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_INT8_math_enabled_RMW>(int8_math_enabled);
    }

    // Re-establish the operand-driven baseline (clears stale op-state) for the new SrcA format.
    _configure_default_zero_flag_state_(srca_data_format, src_zero_flag_srcb_fmt);
}

/**
 * @brief Reconfigure the math thread for a new source B data format.
 *
 * On Blackhole the ALU source format is inferred, so this is a no-op unless the reconfiguration crosses an
 * Int8/Int32 boundary (to_from_int8), in which case it re-evaluates and programs the INT8 math enable bit.
 *
 * @tparam is_fp32_dest_acc_en: Whether FP32 accumulation in the destination register is enabled (required when to_from_int8 is set).
 * @tparam to_from_int8: Set when the reconfiguration switches to or from an Int8/Int32 format.
 * @param srcb_data_format: New data format of source B (DataFormat enum underlying value).
 */
template <bool is_fp32_dest_acc_en, bool to_from_int8 = false>
inline void _llk_math_reconfig_data_format_srcb_(const std::uint32_t srcb_data_format)
{
    llk::san::math_operand_configure<true>(llk::san::IGNORE, srcb_data_format);

    if constexpr (to_from_int8)
    {
        static_assert(is_fp32_dest_acc_en, "Reconfiguring math to/from Int8 formats requires FP32 Dest mode enabled");
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH);
        std::uint32_t int8_math_enabled = (masked_data_format(srcb_data_format) == ckernel::to_underlying(DataFormat::Int8)) ||
                                          (srcb_data_format == ckernel::to_underlying(DataFormat::Int32));
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_INT8_math_enabled_RMW>(int8_math_enabled);
    }

    // Re-establish the operand-driven baseline (clears stale op-state) for the new SrcB format.
    _configure_default_zero_flag_state_(src_zero_flag_srca_fmt, srcb_data_format);
}

/**
 * @brief Reconfigure the math thread for new source A and source B data formats.
 *
 * On Blackhole the ALU source format is inferred, so this is a no-op unless the reconfiguration crosses an
 * Int8/Int32 boundary (to_from_int8), in which case it re-evaluates and programs the INT8 math enable bit
 * from both source formats.
 *
 * @tparam is_fp32_dest_acc_en: Whether FP32 accumulation in the destination register is enabled (required when to_from_int8 is set).
 * @tparam to_from_int8: Set when the reconfiguration switches to or from an Int8/Int32 format.
 * @param srca_data_format: New data format of source A (DataFormat enum underlying value).
 * @param srcb_data_format: New data format of source B (DataFormat enum underlying value).
 */
template <bool is_fp32_dest_acc_en, bool to_from_int8 = false>
inline void _llk_math_reconfig_data_format_(const std::uint32_t srca_data_format, const std::uint32_t srcb_data_format)
{
    llk::san::math_operand_configure<true>(srca_data_format, srcb_data_format);

    if constexpr (to_from_int8)
    {
        static_assert(is_fp32_dest_acc_en, "Reconfiguring math to/from Int8 formats requires FP32 Dest mode enabled");
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH);
        std::uint32_t int8_math_enabled = (masked_data_format(srca_data_format) == ckernel::to_underlying(DataFormat::Int8)) ||
                                          (masked_data_format(srcb_data_format) == ckernel::to_underlying(DataFormat::Int8)) ||
                                          (srca_data_format == ckernel::to_underlying(DataFormat::Int32)) ||
                                          (srcb_data_format == ckernel::to_underlying(DataFormat::Int32));
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_INT8_math_enabled_RMW>(int8_math_enabled);
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
