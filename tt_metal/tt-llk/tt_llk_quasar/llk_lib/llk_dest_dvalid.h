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
enum class dest_dvalid_client : std::uint32_t
{
    UNPACK = 0,
    FPU    = 1,
    SFPU   = 2,
    PACK   = 3,
};

constexpr std::uint32_t dest_dvalid_bit(dest_dvalid_client client)
{
    return 1u << static_cast<std::uint32_t>(client);
}

constexpr std::uint32_t dest_dvalid_ctrl_addr32(dest_dvalid_client client)
{
    return UNPACK_TO_DEST_DVALID_CTRL_wait_mask_ADDR32 + static_cast<std::uint32_t>(client);
}

constexpr std::uint32_t dest_dvalid_idle_res(dest_dvalid_client client)
{
    return (client == dest_dvalid_client::UNPACK) ? p_stall::UNPACK0 : (client == dest_dvalid_client::PACK) ? p_stall::PACK : p_stall::NOTHING;
}

constexpr std::uint32_t DEST_DVALID_ALL_BITS = dest_dvalid_bit(dest_dvalid_client::UNPACK) | dest_dvalid_bit(dest_dvalid_client::FPU) |
                                               dest_dvalid_bit(dest_dvalid_client::SFPU) | dest_dvalid_bit(dest_dvalid_client::PACK);

constexpr std::uint32_t DEST_DVALID_WAIT_CTRL_MASK = UNPACK_TO_DEST_DVALID_CTRL_wait_mask_MASK | UNPACK_TO_DEST_DVALID_CTRL_wait_polarity_MASK;

static_assert(dest_dvalid_bit(dest_dvalid_client::UNPACK) == p_cleardvalid::UNPACK_TO_DEST);
static_assert(dest_dvalid_bit(dest_dvalid_client::FPU) == p_cleardvalid::FPU);
static_assert(dest_dvalid_bit(dest_dvalid_client::SFPU) == p_cleardvalid::SFPU);
static_assert(dest_dvalid_bit(dest_dvalid_client::PACK) == p_cleardvalid::PACK);

static_assert(MATH_DEST_DVALID_CTRL_wait_mask_ADDR32 == UNPACK_TO_DEST_DVALID_CTRL_wait_mask_ADDR32 + 1);
static_assert(SFPU_DEST_DVALID_CTRL_wait_mask_ADDR32 == UNPACK_TO_DEST_DVALID_CTRL_wait_mask_ADDR32 + 2);
static_assert(PACK_DEST_DVALID_CTRL_wait_mask_ADDR32 == UNPACK_TO_DEST_DVALID_CTRL_wait_mask_ADDR32 + 3);
static_assert(MATH_DEST_DVALID_CTRL_wait_mask_MASK == UNPACK_TO_DEST_DVALID_CTRL_wait_mask_MASK);
static_assert(PACK_DEST_DVALID_CTRL_wait_polarity_SHAMT == UNPACK_TO_DEST_DVALID_CTRL_wait_polarity_SHAMT);
static_assert(MATH_DEST_DVALID_CTRL_toggle_mask_SHAMT == UNPACK_TO_DEST_DVALID_CTRL_toggle_mask_SHAMT);
static_assert(PACK_DEST_DVALID_CTRL_toggle_mask_MASK == UNPACK_TO_DEST_DVALID_CTRL_toggle_mask_MASK);

static_assert((DEST_DVALID_WAIT_CTRL_MASK & UNPACK_TO_DEST_DVALID_CTRL_toggle_mask_MASK) == 0);
static_assert(DEST_DVALID_ALL_BITS == UNPACK_TO_DEST_DVALID_CTRL_wait_mask_MASK);

template <dest_dvalid_client CLIENT>
std::uint32_t dest_dvalid_toggle_mask = 0;

template <dest_dvalid_client CLIENT, bool IS_FIRST = false>
inline void _llk_dest_dvalid_enable_()
{
    constexpr std::uint32_t OWN_BIT       = dest_dvalid_bit(CLIENT);
    constexpr std::uint32_t WAIT_MASK     = IS_FIRST ? DEST_DVALID_ALL_BITS : OWN_BIT;
    constexpr std::uint32_t WAIT_POLARITY = IS_FIRST ? 0u : OWN_BIT;

    constexpr std::uint32_t CTRL =
        (WAIT_MASK << UNPACK_TO_DEST_DVALID_CTRL_wait_mask_SHAMT) | (WAIT_POLARITY << UNPACK_TO_DEST_DVALID_CTRL_wait_polarity_SHAMT);

    cfg_rmw(dest_dvalid_ctrl_addr32(CLIENT), 0, DEST_DVALID_WAIT_CTRL_MASK, CTRL);
    TTI_STALLWAIT(p_stall::STALL_THREAD, p_stall::NOTHING, p_stall::CFGEXU, p_stall::TRISC_CFG);
}

template <dest_dvalid_client CLIENT, dest_dvalid_client NEXT, DstSync DST, bool EN_32BIT_DEST = false>
inline void _llk_dest_dvalid_signal_()
{
    static_assert(CLIENT != NEXT, "A dest dvalid chain has to hand its sections over to another client");

    constexpr std::uint32_t CLIENT_SEL  = dest_dvalid_bit(CLIENT);
    constexpr std::uint32_t TOGGLE_MASK = CLIENT_SEL | dest_dvalid_bit(NEXT);

    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::NOTHING, p_stall::WAIT_SFPU, dest_dvalid_idle_res(CLIENT));

    if (dest_dvalid_toggle_mask<CLIENT> != TOGGLE_MASK)
    {
        dest_dvalid_toggle_mask<CLIENT> = TOGGLE_MASK;
        cfg_rmw(dest_dvalid_ctrl_addr32(CLIENT), UNPACK_TO_DEST_DVALID_CTRL_toggle_mask_SHAMT, UNPACK_TO_DEST_DVALID_CTRL_toggle_mask_MASK, TOGGLE_MASK);
        TTI_STALLWAIT(p_stall::STALL_THREAD, p_stall::NOTHING, p_stall::CFGEXU, p_stall::TRISC_CFG);
    }

    if constexpr (CLIENT == dest_dvalid_client::PACK)
    {
        constexpr std::uint32_t CLR_MODE = (DST == DstSync::SyncHalf) ? p_zeroacc::CLR_HALF : p_zeroacc::CLR_ALL;
        if constexpr (DST == DstSync::SyncFull)
        {
            TTI_ZEROACC(CLR_MODE, EN_32BIT_DEST, 0, ADDR_MOD_0, 0);
        }
        else
        {
            TT_ZEROACC(CLR_MODE, EN_32BIT_DEST, 0, ADDR_MOD_0, ckernel::pack::clear_dest_bank_id);
        }
    }

    TTI_CLEARDVALID(0, 0, 0, 0, CLIENT_SEL, 0);
    if constexpr (DST == DstSync::SyncFull)
    {
        TTI_CLEARDVALID(0, 0, 0, CLIENT_SEL, CLIENT_SEL, 0);
    }

    if constexpr (CLIENT == dest_dvalid_client::PACK && DST == DstSync::SyncHalf)
    {
        ckernel::pack::_update_clear_dest_bank_id_();
    }
}

template <dest_dvalid_client CLIENT>
inline void _llk_dest_dvalid_disable_()
{
    TTI_STALLWAIT(p_stall::STALL_MATH | p_stall::STALL_CFG, p_stall::NOTHING, p_stall::WAIT_SFPU, dest_dvalid_idle_res(CLIENT));
    TTI_CLEARDVALID(0, 0, 0, dest_dvalid_bit(CLIENT), 0, 0);
    cfg_rmw(dest_dvalid_ctrl_addr32(CLIENT), 0, DEST_DVALID_WAIT_CTRL_MASK, 0);
}

} // namespace ckernel
