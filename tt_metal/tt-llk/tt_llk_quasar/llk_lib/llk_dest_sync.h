// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_trisc_common.h"
#include "llk_assert.h"
#include "llk_defs.h"

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

struct Slot
{
    friend constexpr auto region_mode(Slot);
};

// Dest is split into two regions (one in SyncFull). Two are enough for two clients, but with a third client, for example
// FPU, SFPU and PACK, one client always has to wait for a free region. In the future dest should hold more tensors.
template <DstSync DST, bool EN_32BIT_DEST>
struct DestRegion
{
    friend constexpr auto region_mode(Slot)
    {
        return DestRegion {};
    }

    static_assert(DST == DstSync::SyncHalf || DST == DstSync::SyncFull, "Unknown destination synchronization mode");

    static constexpr std::uint32_t REGIONS = DST == DstSync::SyncHalf ? 2 : 1;
    static constexpr std::uint32_t ROWS    = (trisc::DEST_REGISTER_FULL_SIZE / REGIONS) >> EN_32BIT_DEST;

    static constexpr std::uint32_t base(const std::uint32_t index)
    {
        return index * ROWS;
    }

    static constexpr std::uint32_t next(const std::uint32_t index)
    {
        return index ^ (REGIONS - 1);
    }

    static constexpr std::uint32_t clear_word(const std::uint32_t index)
    {
        return TT_OP_ZEROACC(REGIONS == 2 ? p_zeroacc::CLR_HALF : p_zeroacc::CLR_ALL, EN_32BIT_DEST, 0, ADDR_MOD_7, index);
    }
};

template <std::uint32_t TAKER>
using Region = decltype(region_mode(std::conditional_t<(TAKER < NONE), Slot, Slot> {}));

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

template <std::uint32_t TAKER>
inline void retarget(State& s)
{
    if (s.base == s.region)
    {
        return;
    }
    s.base = s.region;
    program_base(Region<TAKER>::base(s.region));
}

template <std::uint32_t TAKER>
inline void hand_over(State& s)
{
    if (s.holder == ME)
    {
        drain();
        TTI_SEMPOST(BANK, sem(TAKER));
    }
    if constexpr (ME == TAKER && TAKER == PACK)
    {
        retarget<TAKER>(s);
        TTI_SEMWAIT(STALL[ME], p_stall::STALL_ON_ZERO, BANK, FULL);
    }
    if constexpr (ME == TAKER && TAKER != PACK)
    {
        retarget<TAKER>(s);
        take<sem(ME)>();
    }
}

template <std::uint32_t TAKER>
inline void open(State& s)
{
    if (ME == PACK && s.holder == PACK)
    {
        drain();
        TT_INSN(Region<TAKER>::clear_word(s.region));
        TTI_SEMGET(BANK, FULL);
    }
    s.region           = Region<TAKER>::next(s.region);
    const bool chained = s.last != NONE && s.last != TAKER;
    if (chained && s.last == ME)
    {
        TTI_SEMPOST(BANK, sem(TAKER));
    }
    if constexpr (ME == TAKER)
    {
        retarget<TAKER>(s);
        if (chained)
        {
            take<sem(ME)>();
        }
        TTI_SEMWAIT(STALL[ME], p_stall::STALL_ON_MAX, BANK, FULL);
    }
}
}

template <DstSync DST, bool EN_32BIT_DEST>
inline decltype(void(sizeof(dst_sync::DestRegion<DST, EN_32BIT_DEST>))) _llk_dest_sync_configure_()
{
    using namespace dst_sync;
    using Mode = DestRegion<DST, EN_32BIT_DEST>;

    if constexpr (ME == PACK)
    {
        drain();
        TTI_ZEROACC(p_zeroacc::CLR_ALL, EN_32BIT_DEST, 0, ADDR_MOD_7, 0);
        TTI_SEMINIT(Mode::REGIONS, 0, BANK, FULL);
        TTI_SEMPOST(BANK, PRODUCERS);
    }
    else
    {
        take<sem(ME)>();
    }
    program_base(0);
    state = {NONE, NONE, Mode::REGIONS - 1, 0};
}

template <DestClient CLIENT>
NOINLINE void _llk_dest_sync_take_()
{
    using namespace dst_sync;
    constexpr std::uint32_t TAKER = to_underlying(CLIENT);

    if (state.holder == TAKER)
    {
        return;
    }
    State s = state;
    LLK_ASSERT(TAKER != PACK || s.holder != NONE, "PACK only reads dest, so it cannot open a use");
    const bool opens    = TAKER != PACK && (s.holder == NONE || s.holder == PACK);
    const bool clears   = ME == PACK && s.holder == PACK;
    const bool involved = ME == TAKER || (opens ? clears || s.last == ME : s.holder == ME);
    if (!involved)
    {
        state.holder = TAKER;
        if (TAKER != PACK)
        {
            state.last = TAKER;
        }
        if (opens)
        {
            state.region = Region<TAKER>::next(s.region);
        }
        return;
    }
    if (opens)
    {
        open<TAKER>(s);
    }
    else
    {
        hand_over<TAKER>(s);
    }
    s.holder = TAKER;
    s.last   = TAKER == PACK ? s.last : TAKER;
    state    = s;
}

} // namespace ckernel
