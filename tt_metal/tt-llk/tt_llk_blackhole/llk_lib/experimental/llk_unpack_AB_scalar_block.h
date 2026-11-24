// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_unpack_AB_scalar_block.h. Same instruction sequence and
// access kinds as production, expressed through the HAL.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "cunpack_common.h"
#include "hal/cfg.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "hal/unpack.h"
#include "llk_unpack_AB_reduce.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"

namespace ckernel
{
using namespace unpacker;

inline void _llk_unpack_AB_scalar_block_init_(const ckernel::TensorShape& tensor_shape_a, const ckernel::TensorShape& tensor_shape_b)
{
    // Two-context handshake op: an earlier op's init may have left Auto TTSync on, and tt-metal calls no uninit.
    hal_port::unpack_sync_mode<hal_port::UnpackSyncMode::Handshake>();
    namespace cfg = hal::cfg;
    namespace adc = hal::counters::adc;

    // cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0)
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg2].Haloize_mode, cfg::Sec::S0, 0>();
    // TT_SETADCXX(p_setadc::UNP0, tensor_shape_a.total_tensor_size() - 1, 0x0)
    adc::set(adc::AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = tensor_shape_a.total_tensor_size() - 1u}}.for_clients<adc::Client::Unpacker0>());
    // TT_SETADCXX(p_setadc::UNP1, tensor_shape_b.total_tensor_size() - 1, 0x0)
    adc::set(adc::AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = tensor_shape_b.total_tensor_size() - 1u}}.for_clients<adc::Client::Unpacker1>());
}

inline void _llk_unpack_AB_scalar_block_(const std::uint32_t address_a, const std::uint32_t address_b, const std::uint32_t block_size)
{
    namespace adc  = hal::counters::adc;
    namespace sync = hal::sync;
    using hal::unpack::Engine;
    using hal::unpack::SourceHandoff;

    // TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111)
    adc::set<adc::AddrCounter {.channel0 = {.z = 0, .w = 0}, .channel1 = {.z = 0, .w = 0}}.for_clients<adc::Client::Unpacker0, adc::Client::Unpacker1>()>();

    volatile std::uint32_t tt_reg_ptr* cfg = hal::cfg::detail::state_cfg_bank();
    hal_port::wait_for_next_context(2);
    _llk_unpack_configure_addresses_(address_a, address_b, cfg);
    sync::semaphore::post<sync::Access::MMIO, hal_port::UNPACK_SYNC_SEM>();
    sync::wait::stall<sync::StallTarget::Unpack, sync::StallCondition::RiscvAccessProcessed>();

    // TTI_UNPACR(Src, kNoAddressIncrement, 0, 0, 0, 1, 1, RAREFYB_DISABLE, 0, 0, 0, 0, 1)
    constexpr std::uint32_t unpack_srcb = hal_port::reduce_unpack(Engine::Unpacker1, 0, 0, SourceHandoff::FlipAndSetDataValid);
    constexpr std::uint32_t unpack_srca = hal_port::reduce_unpack(Engine::Unpacker0, 0, 0, SourceHandoff::FlipAndSetDataValid);
    TTI_INSN(unpack_srcb);

    for (std::uint32_t i = 0; i < block_size; ++i)
    {
        TTI_INSN(unpack_srca);
        // TTI_INCADCZW(p_setadc::UNP_A, 0, 0, 1, 0): channel 0 W += 1
        adc::set<adc::AddrCounter {.action = adc::Action::Increment, .channel0 = {.w = 1}}.for_clients<adc::Client::Unpacker0>()>();
    }

    sync::semaphore::get<hal_port::UNPACK_SYNC_SEM>();
    hal_port::switch_config_context(unp_cfg_context);
}

} // namespace ckernel
