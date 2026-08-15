// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "cpack_common.h"
#include "llk_defs.h"

namespace ckernel
{
namespace dest_dvalid
{

enum class client : std::uint32_t
{
    UNPACK = 0,
    FPU    = 1,
    PACK   = 2,
};

struct client_config
{
    std::uint32_t bit;
    std::uint32_t ctrl_addr32;
    std::uint32_t drain_res;
};

constexpr client_config CLIENT_CONFIGS[] = {
    {p_cleardvalid::UNPACK_TO_DEST, UNPACK_TO_DEST_DVALID_CTRL_wait_mask_ADDR32, p_stall::UNPACK0},
    {p_cleardvalid::FPU, MATH_DEST_DVALID_CTRL_wait_mask_ADDR32, p_stall::NOTHING},
    {p_cleardvalid::PACK, PACK_DEST_DVALID_CTRL_wait_mask_ADDR32, p_stall::PACK},
};

template <client CLIENT>
constexpr client_config config_of = CLIENT_CONFIGS[to_underlying(CLIENT)];

constexpr std::uint32_t ALL_CLIENTS = p_cleardvalid::UNPACK_TO_DEST | p_cleardvalid::FPU | p_cleardvalid::SFPU | p_cleardvalid::PACK;

constexpr std::uint32_t CTRL_MASK = UNPACK_TO_DEST_DVALID_CTRL_wait_mask_MASK | UNPACK_TO_DEST_DVALID_CTRL_wait_polarity_MASK |
                                    UNPACK_TO_DEST_DVALID_CTRL_toggle_mask_MASK | UNPACK_TO_DEST_DVALID_CTRL_disable_auto_bank_id_toggle_MASK;

static_assert(MATH_DEST_DVALID_CTRL_wait_mask_SHAMT == UNPACK_TO_DEST_DVALID_CTRL_wait_mask_SHAMT);
static_assert(MATH_DEST_DVALID_CTRL_wait_polarity_SHAMT == UNPACK_TO_DEST_DVALID_CTRL_wait_polarity_SHAMT);
static_assert(MATH_DEST_DVALID_CTRL_toggle_mask_SHAMT == UNPACK_TO_DEST_DVALID_CTRL_toggle_mask_SHAMT);
static_assert(PACK_DEST_DVALID_CTRL_wait_mask_SHAMT == UNPACK_TO_DEST_DVALID_CTRL_wait_mask_SHAMT);
static_assert(PACK_DEST_DVALID_CTRL_wait_polarity_SHAMT == UNPACK_TO_DEST_DVALID_CTRL_wait_polarity_SHAMT);
static_assert(PACK_DEST_DVALID_CTRL_toggle_mask_SHAMT == UNPACK_TO_DEST_DVALID_CTRL_toggle_mask_SHAMT);

template <client CLIENT>
inline void wait_client_idle()
{
    wait_mop_idle();

    switch (CLIENT)
    {
        case client::UNPACK:
            return wait_unpack_idle();
        case client::FPU:
            return wait_fpu_idle();
        case client::PACK:
            return wait_pack_idle();
    }
}

} // namespace dest_dvalid

template <dest_dvalid::client CLIENT, dest_dvalid::client NEXT, bool FIRST = false>
inline void _llk_dest_dvalid_configure_()
{
    static_assert(NEXT != dest_dvalid::client::UNPACK, "UNPACK is always first in the chain, so it can never be a successor");
    static_assert(CLIENT != dest_dvalid::client::UNPACK || FIRST, "UNPACK is always first in the chain");
    static_assert(CLIENT != dest_dvalid::client::FPU || NEXT == dest_dvalid::client::PACK, "PACK always follows FPU in the chain");
    static_assert(CLIENT != dest_dvalid::client::PACK || NEXT == dest_dvalid::client::PACK, "PACK is always last in the chain");
    static_assert(CLIENT != dest_dvalid::client::PACK || !FIRST, "PACK is always last in the chain, so it can never be first");
    static_assert(CLIENT != NEXT || CLIENT == dest_dvalid::client::PACK, "UNPACK and FPU must hand off to a different client");

    dest_dvalid::wait_client_idle<CLIENT>();

    constexpr std::uint32_t OWN  = dest_dvalid::config_of<CLIENT>.bit;
    constexpr std::uint32_t SUCC = dest_dvalid::config_of<NEXT>.bit;

    constexpr std::uint32_t WAIT_MASK     = FIRST ? dest_dvalid::ALL_CLIENTS : OWN;
    constexpr std::uint32_t WAIT_POLARITY = FIRST ? 0u : OWN;
    constexpr std::uint32_t TOGGLE_MASK   = FIRST ? SUCC : (OWN | SUCC);

    constexpr std::uint32_t CTRL = (WAIT_MASK << UNPACK_TO_DEST_DVALID_CTRL_wait_mask_SHAMT) |
                                   (WAIT_POLARITY << UNPACK_TO_DEST_DVALID_CTRL_wait_polarity_SHAMT) |
                                   (TOGGLE_MASK << UNPACK_TO_DEST_DVALID_CTRL_toggle_mask_SHAMT);

    cfg_rmw(dest_dvalid::config_of<CLIENT>.ctrl_addr32, 0, dest_dvalid::CTRL_MASK, CTRL);
    TTI_STALLWAIT(p_stall::STALL_THREAD, p_stall::NOTHING, p_stall::CFGEXU, p_stall::TRISC_CFG);
}

template <dest_dvalid::client CLIENT>
inline void _llk_dest_dvalid_disable_()
{
    TTI_STALLWAIT(p_stall::STALL_MATH | p_stall::STALL_CFG, p_stall::NOTHING, p_stall::WAIT_SFPU, dest_dvalid::config_of<CLIENT>.drain_res);
    dest_dvalid::wait_client_idle<CLIENT>();
    TTI_CLEARDVALID(0, 0, 0, dest_dvalid::config_of<CLIENT>.bit, 0, 0);
    cfg_rmw(dest_dvalid::config_of<CLIENT>.ctrl_addr32, 0, dest_dvalid::CTRL_MASK, 0);
    TTI_STALLWAIT(p_stall::STALL_THREAD, p_stall::NOTHING, p_stall::CFGEXU, p_stall::TRISC_CFG);
}

template <dest_dvalid::client CLIENT, DstSync DST, bool EN_32BIT_DEST>
inline void _llk_dest_dvalid_signal_()
{
    constexpr std::uint32_t OWN = dest_dvalid::config_of<CLIENT>.bit;

    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::NOTHING, p_stall::WAIT_SFPU, dest_dvalid::config_of<CLIENT>.drain_res);

    if constexpr (CLIENT == dest_dvalid::client::PACK)
    {
        if constexpr (DST == DstSync::SyncFull)
        {
            TTI_ZEROACC(p_zeroacc::CLR_ALL, EN_32BIT_DEST, 0, ADDR_MOD_0, 0);
        }
        else
        {
            TT_ZEROACC(p_zeroacc::CLR_HALF, EN_32BIT_DEST, 0, ADDR_MOD_0, ckernel::pack::clear_dest_bank_id);
            ckernel::pack::_update_clear_dest_bank_id_();
        }
    }

    TTI_CLEARDVALID(0, 0, 0, 0, OWN, 0);
    if constexpr (DST == DstSync::SyncFull)
    {
        TTI_CLEARDVALID(0, 0, 0, OWN, OWN, 0);
    }
}

} // namespace ckernel
