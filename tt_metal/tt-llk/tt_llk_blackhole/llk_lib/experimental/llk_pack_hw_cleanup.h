// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_pack_hw_cleanup.h. Same instruction sequence and access
// kinds as production, expressed through the HAL.

#pragma once

#include <cstdint>

#include "ckernel_template.h"
#include "experimental/llk_hw_cleanup.h"
#include "hal/cfg.h"
#include "hal/nop.h"
#include "hal/scalar.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "llk_defs.h"
#include "llk_pack.h"
#include "llk_pack_common.h"

using namespace ckernel;
using namespace ckernel::packer;

template <bool is_fp32_dest_acc_en>
inline void _llk_pack_hw_cleanup_configure_current_bank_()
{
    constexpr std::uint32_t canonical_format = to_underlying(DataFormat::Float16_b);
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, PackMode::Default>(
        canonical_format, canonical_format, HW_CLEANUP_CANONICAL_TILE_SIZE_16B, FACE_R_DIM, TILE_C_DIM, 4, false, 0);
}

/**
 * Poison pack MOP / strides / PAC X so a following MicroOp cannot rely on
 * ambient Default pack state for those fields. Ops must call pack_init or
 * pack_reconfig_data_format<true> (or an equivalent full pack init) before packing.
 *
 * Default pack ADDR_MOD is restored: pack_reconfig_data_format<true> skips
 * addrmod config, and pack_block_contiguous_init only replaces the MOP.
 *
 * hw_configure above programs correct Default strides; overwrite them here.
 */
inline void _llk_pack_hw_cleanup_poison_pack_ambient_()
{
    namespace cfg      = hal::cfg;
    namespace sync     = hal::sync;
    namespace adc      = hal::counters::adc;
    constexpr auto Ch0 = cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg0];
    static_assert(Ch0.Xstride.addr32(cfg::Sec::S0) == PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32, "packer XY stride word");
    static_assert(Ch0.Zstride.addr32(cfg::Sec::S0) == PCK0_ADDR_CTRL_ZW_REG_0_Zstride_ADDR32, "packer ZW stride word");

    _llk_pack_configure_addrmod_<PackMode::Default>();

    // 1x1 NOP double-loop so accidental MOP runs are inert.
    // HAL-GAP: hal::mop::program rejects a Template 1 configuration that issues nothing, so the inert all-NOP
    // MOP is programmed through production's ckernel_template.
    ckernel_template(1, 1).program();

    // Zero packer strides (X unused; Y/Z/W cleared): TT_SETDMAREG(0, 0, 0, LO_16/HI_16(TMP0/TMP1)).
    hal::scalar::set_low<0>(hal::gpr<p_gpr_pack::TMP0>());
    hal::scalar::set_high<0>(hal::gpr<p_gpr_pack::TMP0>());
    hal::scalar::set_low<0>(hal::gpr<p_gpr_pack::TMP1>());
    hal::scalar::set_high<0>(hal::gpr<p_gpr_pack::TMP1>());
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::ScalarIdle>();
    cfg::write<cfg::Access::TensixCfgUnit, Ch0.Xstride, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
        hal::gpr<p_gpr_pack::TMP0>());
    cfg::write<cfg::Access::TensixCfgUnit, Ch0.Zstride, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
        hal::gpr<p_gpr_pack::TMP1>());
    hal::nop::thread();
    hal::nop::thread();

    // Wrong PAC X (Default pack_init programs FACE_C_DIM - 1): TTI_SETADCXX(p_setadc::PAC, 0, 0x0).
    adc::set<adc::AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = 0}}.for_clients<adc::Client::Packers>()>();
}

/**
 * @brief Quiesces all TRISCs and restores pack cfg bank 0 to canonical Float16_b 32x32 geometry.
 * @tparam Dst Kernel DST_SYNC_MODE (compile-time; not modified by cleanup).
 * @tparam is_fp32_dest_acc_en Kernel DST_ACCUM_MODE (re-asserted, not changed).
 * @note On return cfg bank 0 uses Float16_b pack formats with 32x32 tiles and four faces,
 *       and bank 0 is selected.
 * @note On return the pack-dest counters and Dest sync match startup and the Default pack
 *       ADDR_MOD is restored, while the pack MOP, strides and PAC X are deliberately poisoned.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_hw_cleanup_canonical_()
{
    hw_cleanup::start<PackThreadId>();

    hw_cleanup::select_cfg_state(0);
    _llk_pack_hw_cleanup_configure_current_bank_<is_fp32_dest_acc_en>();

    _llk_pack_dest_init_<Dst, is_fp32_dest_acc_en>();

    _llk_pack_hw_cleanup_poison_pack_ambient_();

    hw_cleanup::finish<PackThreadId>();
}
