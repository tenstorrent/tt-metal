// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_trisc_common.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_dest_region.h"

namespace ckernel
{

enum class DestClient : std::uint8_t
{
    UNPACK = 0,
    FPU    = 1,
    SFPU   = 2,
    PACK   = 3,
};

namespace dst_sync
{

constexpr std::uint32_t UNPACK = 0;
constexpr std::uint32_t FPU    = 1;
constexpr std::uint32_t SFPU   = 2;
constexpr std::uint32_t PACK   = 3;
constexpr std::uint32_t NONE   = 4;

#if defined(LLK_TRISC_UNPACK) || defined(TRISC_UNPACK)
constexpr std::uint32_t ME = UNPACK;
#elif defined(LLK_TRISC_MATH) || defined(TRISC_MATH)
constexpr std::uint32_t ME = FPU;
#elif defined(LLK_TRISC_ISOLATE_SFPU) || defined(TRISC_ISOLATE_SFPU)
constexpr std::uint32_t ME = SFPU;
#elif defined(LLK_TRISC_PACK) || defined(TRISC_PACK)
constexpr std::uint32_t ME = PACK;
#else
constexpr std::uint32_t ME = NONE;
#endif

constexpr std::uint32_t BANK      = 1;
constexpr std::uint32_t FIRST     = 4;
constexpr std::uint32_t FULL      = 1u << (FIRST + PACK);
constexpr std::uint32_t PRODUCERS = 0x7u << FIRST;

constexpr std::uint32_t sem(const std::uint32_t client)
{
    return client == PACK ? FULL : 1u << (FIRST + client);
}

constexpr std::uint32_t STALL[] = {
    p_stall::STALL_UNPACK | p_stall::STALL_SYNC,
    p_stall::STALL_MATH | p_stall::STALL_SYNC,
    p_stall::STALL_SFPU | p_stall::STALL_MATH | p_stall::STALL_SYNC,
    p_stall::STALL_PACK | p_stall::STALL_SYNC,
    p_stall::STALL_SYNC};

constexpr std::uint32_t DONE[] = {p_stall::UNPACK0, p_stall::MATH, p_stall::WAIT_SFPU, p_stall::PACK, p_stall::NOTHING};

constexpr std::uint32_t DRAIN_STALL = ME == PACK ? p_stall::STALL_MATH | p_stall::STALL_SYNC : p_stall::STALL_SYNC;

struct State
{
    std::uint8_t holder;
    std::uint8_t last;
    std::uint8_t region;
    std::uint8_t base;
};

inline State state;

inline void drain()
{
    TTI_STALLWAIT(DRAIN_STALL, p_stall::NOTHING, p_stall::NOTHING, DONE[ME]);
}

template <std::uint32_t SEM>
inline void take()
{
    TTI_SEMWAIT(STALL[ME], p_stall::STALL_ON_ZERO, BANK, SEM);
    TTI_SEMGET(BANK, SEM);
}

inline void program_base(const std::uint32_t row)
{
    trisc::dest_register_offset = row;
    if constexpr (ME == PACK)
    {
        cfg_rmw(THCON_PACKER0_REG0_SRC_ADDR_OFFSET_RMW, row);
    }
    if constexpr (ME == FPU)
    {
        cfg_rmw(DEST_TARGET_REG_CFG_MATH_SEC1_Offset_RMW, row);
    }
    if constexpr (ME == SFPU)
    {
        cfg_rmw(DEST_TARGET_REG_CFG_MATH_SEC3_Offset_RMW, row);
    }
}

template <typename Region>
inline void retarget(State& s)
{
    if (s.base == s.region)
    {
        return;
    }
    s.base = s.region;
    program_base(Region::base(s.region));
}

template <typename Region, std::uint32_t TAKER>
inline void hand_over(State& s)
{
    if (s.holder == ME)
    {
        drain();
        TTI_SEMPOST(BANK, sem(TAKER));
    }
    if constexpr (ME == TAKER && TAKER == PACK)
    {
        retarget<Region>(s);
        TTI_SEMWAIT(STALL[ME], p_stall::STALL_ON_ZERO, BANK, FULL);
    }
    if constexpr (ME == TAKER && TAKER != PACK)
    {
        retarget<Region>(s);
        take<sem(ME)>();
    }
}

template <typename Region, std::uint32_t TAKER>
inline void open(State& s)
{
    if (ME == PACK && s.holder == PACK)
    {
        drain();
        TT_INSN(Region::clear_word(s.region));
        TTI_SEMGET(BANK, FULL);
    }
    s.region           = Region::next(s.region);
    const bool chained = s.last != NONE && s.last != TAKER;
    if (chained && s.last == ME)
    {
        TTI_SEMPOST(BANK, sem(TAKER));
    }
    if constexpr (ME == TAKER)
    {
        retarget<Region>(s);
        if (chained)
        {
            take<sem(ME)>();
        }
        TTI_SEMWAIT(STALL[ME], p_stall::STALL_ON_MAX, BANK, FULL);
    }
}
}

template <DstSync DST, bool EN_32BIT_DEST>
inline void _llk_dest_sync_init_()
{
    using namespace dst_sync;
    using Region = DestRegionOf<DST, EN_32BIT_DEST>;

    if constexpr (ME == PACK)
    {
        drain();
        TTI_ZEROACC(p_zeroacc::CLR_ALL, EN_32BIT_DEST, 0, ADDR_MOD_7, 0);
        TTI_SEMINIT(Region::REGIONS, 0, BANK, FULL);
        TTI_SEMPOST(BANK, PRODUCERS);
    }
    else
    {
        take<sem(ME)>();
    }
    program_base(0);
    state = {NONE, NONE, Region::REGIONS - 1, 0};
}

template <DstSync DST, DestClient CLIENT, bool EN_32BIT_DEST>
NOINLINE void _llk_dest_sync_take_()
{
    using namespace dst_sync;
    using Region                  = DestRegionOf<DST, EN_32BIT_DEST>;
    constexpr std::uint32_t TAKER = to_underlying(CLIENT);

    State s = state;
    if (s.holder == TAKER)
    {
        return;
    }
    LLK_ASSERT(TAKER != PACK || s.holder != NONE, "PACK only reads dest, so it cannot open a use");
    if (TAKER != PACK && (s.holder == NONE || s.holder == PACK))
    {
        open<Region, TAKER>(s);
    }
    else
    {
        hand_over<Region, TAKER>(s);
    }
    s.holder = TAKER;
    s.last   = TAKER == PACK ? s.last : TAKER;
    state    = s;
}

} // namespace ckernel
