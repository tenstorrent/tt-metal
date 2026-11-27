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

template <bool is_fp32_dest_acc_en>
inline void _llk_math_hw_cleanup_configure_current_bank_()
{
    constexpr std::uint32_t canonical_format = to_underlying(DataFormat::Float16_b);
    _invalidate_src_zero_flag_state_();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(canonical_format, canonical_format);
}

template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_hw_cleanup_init_dest_sync_()
{
    _llk_math_pack_sync_init_<Dst, is_fp32_dest_acc_en>();
}

/**
 * @brief Zero the transient math/unpack-style ADDR_MOD_0..5 so stale increments cannot survive.
 *
 * ADDR_MOD_6/7 are left at their @ref _llk_math_hw_configure_ invariants (dest += 2 / zero advance),
 * which the SFPU ops after cleanup assume.
 */
inline void _llk_math_hw_cleanup_poison_addr_mods_()
{
    for (std::uint8_t mod_index = 0; mod_index < 6; ++mod_index)
    {
        addr_mod_t {}.set(mod_index);
    }
    addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
}

/**
 * @brief Quiesces all TRISCs and restores math cfg bank 0 to canonical Float16_b formats.
 * @tparam Dst Kernel DST_SYNC_MODE (compile-time; not modified by cleanup).
 * @tparam is_fp32_dest_acc_en Kernel DST_ACCUM_MODE (re-asserted, not changed).
 * @note On return cfg bank 0 uses Float16_b formats, the global Blackhole dest remap is
 *       enabled, and bank 0 is selected.
 * @note On return dest half-sync is on section 0, MOP CFG is NOP-poisoned, ADDR_MOD_0..5 are
 *       zero, and ADDR_MOD_6/7 hold their hw_configure invariants (dest += 2 / zero advance).
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_hw_cleanup_canonical_()
{
    hw_cleanup::start<MathThreadId>();

    _llk_math_hw_cleanup_init_dest_sync_<Dst, is_fp32_dest_acc_en>();

    hw_cleanup::select_cfg_state(0);
    _llk_math_hw_cleanup_configure_current_bank_<is_fp32_dest_acc_en>();

    _llk_math_reconfig_remap_(true);

    // 1x1 NOP double-loop so accidental ckernel_template::run() is a no-op.
    ckernel::ckernel_template(1, 1).program();
    _llk_math_hw_cleanup_poison_addr_mods_();

    hw_cleanup::finish<MathThreadId>();
}
