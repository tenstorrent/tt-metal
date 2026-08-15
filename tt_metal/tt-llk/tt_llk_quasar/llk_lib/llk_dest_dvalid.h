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

struct dest_dvalid_config
{
    std::uint32_t ctrl_addr32;
    std::uint32_t bit;
    std::uint32_t drain_res;
};

constexpr dest_dvalid_config dest_dvalid_configs[] = {
    {UNPACK_TO_DEST_DVALID_CTRL_wait_mask_ADDR32, p_cleardvalid::UNPACK_TO_DEST, p_stall::UNPACK0},
    {MATH_DEST_DVALID_CTRL_wait_mask_ADDR32, p_cleardvalid::FPU, p_stall::NOTHING},
    {SFPU_DEST_DVALID_CTRL_wait_mask_ADDR32, p_cleardvalid::SFPU, p_stall::NOTHING},
    {PACK_DEST_DVALID_CTRL_wait_mask_ADDR32, p_cleardvalid::PACK, p_stall::PACK},
};

template <dest_dvalid_client CLIENT>
constexpr dest_dvalid_config dest_dvalid_config_of = dest_dvalid_configs[to_underlying(CLIENT)];

constexpr std::uint32_t DEST_DVALID_ALL_CLIENTS =
    dest_dvalid_configs[0].bit | dest_dvalid_configs[1].bit | dest_dvalid_configs[2].bit | dest_dvalid_configs[3].bit;

constexpr std::uint32_t DEST_DVALID_CTRL_MASK = UNPACK_TO_DEST_DVALID_CTRL_wait_mask_MASK | UNPACK_TO_DEST_DVALID_CTRL_wait_polarity_MASK |
                                                UNPACK_TO_DEST_DVALID_CTRL_toggle_mask_MASK | UNPACK_TO_DEST_DVALID_CTRL_disable_auto_bank_id_toggle_MASK;

static_assert(MATH_DEST_DVALID_CTRL_wait_polarity_SHAMT == UNPACK_TO_DEST_DVALID_CTRL_wait_polarity_SHAMT);
static_assert(MATH_DEST_DVALID_CTRL_toggle_mask_SHAMT == UNPACK_TO_DEST_DVALID_CTRL_toggle_mask_SHAMT);
static_assert(PACK_DEST_DVALID_CTRL_toggle_mask_MASK == UNPACK_TO_DEST_DVALID_CTRL_toggle_mask_MASK);

static std::uint32_t dest_dvalid_chain = DEST_DVALID_ALL_CLIENTS;

constexpr std::uint32_t dest_dvalid_lowest(std::uint32_t mask)
{
    return mask & (~mask + 1);
}

inline std::uint32_t dest_dvalid_successor(std::uint32_t bit)
{
    const std::uint32_t after = dest_dvalid_chain & ~((bit << 1) - 1);
    return dest_dvalid_lowest(after != 0 ? after : dest_dvalid_chain);
}

template <dest_dvalid_client CLIENT>
inline void dest_dvalid_wait_client_idle()
{
    wait_mop_idle();

    if constexpr (CLIENT == dest_dvalid_client::UNPACK)
    {
        wait_unpack_idle();
    }
    else if constexpr (CLIENT == dest_dvalid_client::FPU)
    {
        wait_fpu_idle();
    }
    else if constexpr (CLIENT == dest_dvalid_client::SFPU)
    {
        wait_sfpu_idle();
    }
    else
    {
        wait_pack_idle();
    }
}

inline void dest_dvalid_wait_chain_idle()
{
    bstatus_u busy;
    busy.val         = 0;
    busy.global_fpu  = 1;
    busy.global_sfpu = 1;
    busy.global_pack = 1;
    wait_bstatus_low(busy.val);
}

template <dest_dvalid_client CLIENT>
inline void _llk_dest_dvalid_include_()
{
    dest_dvalid_chain |= dest_dvalid_config_of<CLIENT>.bit;
}

template <dest_dvalid_client CLIENT>
inline void _llk_dest_dvalid_exclude_()
{
    dest_dvalid_chain &= ~dest_dvalid_config_of<CLIENT>.bit;
}

template <dest_dvalid_client CLIENT>
inline void _llk_dest_dvalid_enable_()
{
    constexpr dest_dvalid_config CFG = dest_dvalid_config_of<CLIENT>;

    dest_dvalid_chain |= CFG.bit;
    dest_dvalid_wait_client_idle<CLIENT>();

    const bool is_first = dest_dvalid_lowest(dest_dvalid_chain) == CFG.bit;
    if (is_first)
    {
        dest_dvalid_wait_chain_idle();
    }

    const std::uint32_t wait_mask     = is_first ? DEST_DVALID_ALL_CLIENTS : CFG.bit;
    const std::uint32_t wait_polarity = is_first ? 0u : CFG.bit;
    const std::uint32_t toggle_mask   = CFG.bit | dest_dvalid_successor(CFG.bit);

    const std::uint32_t ctrl = (wait_mask << UNPACK_TO_DEST_DVALID_CTRL_wait_mask_SHAMT) | (wait_polarity << UNPACK_TO_DEST_DVALID_CTRL_wait_polarity_SHAMT) |
                               (toggle_mask << UNPACK_TO_DEST_DVALID_CTRL_toggle_mask_SHAMT);

    cfg_rmw(CFG.ctrl_addr32, 0, DEST_DVALID_CTRL_MASK, ctrl);
    TTI_STALLWAIT(p_stall::STALL_THREAD, p_stall::NOTHING, p_stall::CFGEXU, p_stall::TRISC_CFG);
}

template <dest_dvalid_client CLIENT, DstSync DST, bool EN_32BIT_DEST = false>
inline void _llk_dest_dvalid_signal_()
{
    constexpr dest_dvalid_config CFG = dest_dvalid_config_of<CLIENT>;

    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::NOTHING, p_stall::WAIT_SFPU, CFG.drain_res);

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

    TTI_CLEARDVALID(0, 0, 0, 0, CFG.bit, 0);
    if constexpr (DST == DstSync::SyncFull)
    {
        TTI_CLEARDVALID(0, 0, 0, CFG.bit, CFG.bit, 0);
    }

    if constexpr (CLIENT == dest_dvalid_client::PACK && DST == DstSync::SyncHalf)
    {
        ckernel::pack::_update_clear_dest_bank_id_();
    }
}

template <dest_dvalid_client CLIENT>
inline void _llk_dest_dvalid_disable_()
{
    constexpr dest_dvalid_config CFG = dest_dvalid_config_of<CLIENT>;

    dest_dvalid_chain &= ~CFG.bit;

    TTI_STALLWAIT(p_stall::STALL_MATH | p_stall::STALL_CFG, p_stall::NOTHING, p_stall::WAIT_SFPU, CFG.drain_res);
    dest_dvalid_wait_client_idle<CLIENT>();
    TTI_CLEARDVALID(0, 0, 0, CFG.bit, 0, 0);
    cfg_rmw(CFG.ctrl_addr32, 0, DEST_DVALID_CTRL_MASK, 0);
}

} // namespace ckernel
