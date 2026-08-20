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
    std::uint8_t trisc_id;
};

constexpr client_config CLIENT_CONFIGS[] = {
    {p_cleardvalid::UNPACK_TO_DEST, UNPACK_TO_DEST_DVALID_CTRL_wait_mask_ADDR32, p_stall::UNPACK0, to_underlying(trisc::TriscID::Unpack)},
    {p_cleardvalid::FPU, MATH_DEST_DVALID_CTRL_wait_mask_ADDR32, p_stall::MATH, to_underlying(trisc::TriscID::Math)},
    {p_cleardvalid::PACK, PACK_DEST_DVALID_CTRL_wait_mask_ADDR32, p_stall::PACK, to_underlying(trisc::TriscID::Pack)},
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
            wait_sfpu_idle();
            return wait_fpu_idle();
        case client::PACK:
            return wait_pack_idle();
    }
}

enum class mode : std::uint32_t
{
    UNPACK_TO_DEST_DIS = 0,
    UNPACK_TO_DEST_EN,
};

inline mode configured_mode = mode::UNPACK_TO_DEST_DIS;

} // namespace dest_dvalid

namespace dest_dvalid
{

constexpr std::uint32_t make_ctrl(std::uint32_t wait_mask, std::uint32_t wait_polarity, std::uint32_t toggle_mask)
{
    return (wait_mask << UNPACK_TO_DEST_DVALID_CTRL_wait_mask_SHAMT) | (wait_polarity << UNPACK_TO_DEST_DVALID_CTRL_wait_polarity_SHAMT) |
           (toggle_mask << UNPACK_TO_DEST_DVALID_CTRL_toggle_mask_SHAMT);
}

template <client CLIENT, bool UNPACK_TO_DEST>
constexpr std::uint32_t ctrl_value()
{
    constexpr std::uint32_t F = p_cleardvalid::FPU;
    constexpr std::uint32_t P = p_cleardvalid::PACK;

    if constexpr (CLIENT == client::UNPACK)
    {
        return make_ctrl(P, 0, F | P);
    }
    else if constexpr (CLIENT == client::FPU)
    {
        return UNPACK_TO_DEST ? make_ctrl(F, F, F) : make_ctrl(F | P, 0, P);
    }
    else
    {
        return make_ctrl(F | P, P, P);
    }
}

template <bool UNPACK_TO_DEST>
constexpr std::uint32_t sfpu_ctrl_value()
{
    constexpr std::uint32_t F = p_cleardvalid::FPU;
    constexpr std::uint32_t P = p_cleardvalid::PACK;
    constexpr std::uint32_t S = p_cleardvalid::SFPU;

    return UNPACK_TO_DEST ? make_ctrl(F, F, S) : make_ctrl(F | P, 0, S);
}

} // namespace dest_dvalid

template <dest_dvalid::client CLIENT, bool UNPACK_TO_DEST = false>
inline void _llk_dest_dvalid_configure_()
{
    static_assert(CLIENT != dest_dvalid::client::UNPACK || UNPACK_TO_DEST, "UNPACK is only in the chain when it unpacks to DEST");
    static_assert(CLIENT != dest_dvalid::client::PACK || !UNPACK_TO_DEST, "PACK is configured the same way for both chains");

    dest_dvalid::wait_client_idle<CLIENT>();

    if constexpr (CLIENT != dest_dvalid::client::PACK)
    {
        constexpr dest_dvalid::mode TARGET = UNPACK_TO_DEST ? dest_dvalid::mode::UNPACK_TO_DEST_EN : dest_dvalid::mode::UNPACK_TO_DEST_DIS;
        if (dest_dvalid::configured_mode == TARGET)
        {
            return;
        }
        dest_dvalid::configured_mode = TARGET;
    }

    cfg_rmw(dest_dvalid::config_of<CLIENT>.ctrl_addr32, 0, dest_dvalid::CTRL_MASK, dest_dvalid::ctrl_value<CLIENT, UNPACK_TO_DEST>());
    if constexpr (CLIENT == dest_dvalid::client::FPU)
    {
        cfg_rmw(SFPU_DEST_DVALID_CTRL_wait_mask_ADDR32, 0, dest_dvalid::CTRL_MASK, dest_dvalid::sfpu_ctrl_value<UNPACK_TO_DEST>());
    }
    TTI_STALLWAIT(p_stall::STALL_THREAD, p_stall::NOTHING, p_stall::CFGEXU, p_stall::TRISC_CFG);

    if constexpr (CLIENT == dest_dvalid::client::UNPACK)
    {
        ckernel::trisc::_set_dest_section_base_<dest_dvalid::config_of<CLIENT>.trisc_id>(0);
    }
}

inline void _llk_dest_dvalid_math_init_()
{
    dest_dvalid::configured_mode = dest_dvalid::mode::UNPACK_TO_DEST_DIS;
    cfg_rmw(
        dest_dvalid::config_of<dest_dvalid::client::FPU>.ctrl_addr32, 0, dest_dvalid::CTRL_MASK, dest_dvalid::ctrl_value<dest_dvalid::client::FPU, false>());
    cfg_rmw(SFPU_DEST_DVALID_CTRL_wait_mask_ADDR32, 0, dest_dvalid::CTRL_MASK, dest_dvalid::sfpu_ctrl_value<false>());
    TTI_STALLWAIT(p_stall::STALL_THREAD, p_stall::NOTHING, p_stall::CFGEXU, p_stall::TRISC_CFG);
}

template <dest_dvalid::client CLIENT>
inline void _llk_dest_dvalid_disable_()
{
    dest_dvalid::wait_client_idle<CLIENT>();

    if constexpr (CLIENT == dest_dvalid::client::UNPACK)
    {
        if (dest_dvalid::configured_mode == dest_dvalid::mode::UNPACK_TO_DEST_DIS)
        {
            return;
        }
        dest_dvalid::configured_mode = dest_dvalid::mode::UNPACK_TO_DEST_DIS;
    }

    TTI_STALLWAIT(p_stall::STALL_MATH | p_stall::STALL_CFG, p_stall::NOTHING, p_stall::WAIT_SFPU, dest_dvalid::config_of<CLIENT>.drain_res);
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

    if constexpr (CLIENT == dest_dvalid::client::FPU)
    {
        TTI_CLEARDVALID(0, 0, 0, 0, p_cleardvalid::SFPU, 0);
        if constexpr (DST == DstSync::SyncFull)
        {
            TTI_CLEARDVALID(0, 0, 0, p_cleardvalid::SFPU, p_cleardvalid::SFPU, 0);
        }
    }

    TTI_CLEARDVALID(0, 0, 0, 0, OWN, 0);
    if constexpr (DST == DstSync::SyncFull)
    {
        TTI_CLEARDVALID(0, 0, 0, OWN, OWN, 0);
    }
}

} // namespace ckernel
