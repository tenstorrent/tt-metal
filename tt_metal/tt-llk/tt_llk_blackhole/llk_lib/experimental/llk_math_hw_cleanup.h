// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_template.h"
#include "experimental/llk_hw_cleanup.h"
#include "llk_math_common.h"

using namespace ckernel::math;

/**
 * Bank 0 as _llk_math_hw_configure_ leaves it for Float16_b operands, with the Blackhole dest remap enabled. The
 * Float16_b operands take the src zero flag clear.
 */
template <bool is_fp32_dest_acc_en>
inline void _llk_math_hw_cleanup_configure_current_bank_()
{
    constexpr std::uint32_t canonical_format      = to_underlying(DataFormat::Float16_b);
    constexpr std::uint32_t src_fmt_override_mask = ALU_FORMAT_SPEC_REG_SrcA_val_MASK | ALU_FORMAT_SPEC_REG_SrcA_override_MASK |
                                                    ALU_FORMAT_SPEC_REG_SrcB_val_MASK | ALU_FORMAT_SPEC_REG_SrcB_override_MASK;

    hw_cleanup::cfg_rmw<DEST_ACCESS_CFG_zeroacc_absolute_tile_mode_RMW, 0>();
    hw_cleanup::cfg_rmw<ALU_ACC_CTRL_INT8_math_enabled_RMW, 0>();
    hw_cleanup::cfg_rmw<ALU_ACC_CTRL_Fp32_enabled_RMW, is_fp32_dest_acc_en>();
    hw_cleanup::cfg_rmw<ALU_ACC_CTRL_SFPU_Fp32_enabled_RMW, is_fp32_dest_acc_en>();
    hw_cleanup::cfg_rmw<ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32, 0, src_fmt_override_mask, 0>();
    src_zero_flag_srca_fmt = canonical_format;
    src_zero_flag_srcb_fmt = canonical_format;
    src_zero_flag_hw       = 0;
    hw_cleanup::cfg_rmw<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW, 0>();
    hw_cleanup::cfg_rmw<DEST_ACCESS_CFG_remap_addrs_RMW, 1>();
    hw_cleanup::cfg_rmw<DEST_ACCESS_CFG_swizzle_32b_RMW, 1>();
}

// _llk_math_pack_sync_init_ without its wait: MATH_PACK is already zero once both other threads sent READY.
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_hw_cleanup_init_dest_sync_()
{
    TTI_SEMINIT(Dst == DstSync::SyncFull ? 1 : 2, 0, p_stall::SEMAPHORE_1);
    reset_dest_offset_id();
    set_dest_section_base<StartZero>();
}

/** Zero math/unpack-style ADDR_MOD_0..7 so stale increments cannot survive. */
inline void _llk_math_hw_cleanup_poison_addr_mods_()
{
    for (std::uint8_t mod_index = 0; mod_index < 8; ++mod_index)
    {
        addr_mod_t {}.set(mod_index);
    }
}

/**
 * @brief Quiesces all TRISCs and restores math cfg bank 0 to canonical Float16_b formats.
 * @tparam Dst Kernel DST_SYNC_MODE (compile-time; not modified by cleanup).
 * @tparam is_fp32_dest_acc_en Kernel DST_ACCUM_MODE (re-asserted, not changed).
 * @note On return cfg bank 0 uses Float16_b formats and is selected, and the global Blackhole
 *       dest remap is enabled.
 * @note On return dest half-sync is on section 0, MOP CFG is NOP-poisoned, and ADDR_MOD_0..7
 *       are zero.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_hw_cleanup_canonical_()
{
    hw_cleanup::start<MathThreadId>();

    _llk_math_hw_cleanup_init_dest_sync_<Dst, is_fp32_dest_acc_en>();

    hw_cleanup::select_cfg_state(0);
    _llk_math_hw_cleanup_configure_current_bank_<is_fp32_dest_acc_en>();

    // 1x1 NOP double-loop so accidental ckernel_template::run() is a no-op.
    hw_cleanup::program_nop_mop();
    _llk_math_hw_cleanup_poison_addr_mods_();

    hw_cleanup::finish<MathThreadId>();
}
