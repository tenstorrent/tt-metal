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

constexpr std::uint32_t UNP  = 0;
constexpr std::uint32_t FPU  = 1;
constexpr std::uint32_t SFPU = 2;
constexpr std::uint32_t PACK = 3;
constexpr std::uint32_t NONE = 4;

#if defined(LLK_TRISC_UNPACK) || defined(TRISC_UNPACK)
constexpr std::uint32_t THIS = UNP;
#elif defined(LLK_TRISC_MATH) || defined(TRISC_MATH)
constexpr std::uint32_t THIS = FPU;
#elif defined(LLK_TRISC_ISOLATE_SFPU) || defined(TRISC_ISOLATE_SFPU)
constexpr std::uint32_t THIS = SFPU;
#elif defined(LLK_TRISC_PACK) || defined(TRISC_PACK)
constexpr std::uint32_t THIS = PACK;
#else
constexpr std::uint32_t THIS = NONE;
#endif

constexpr bool TRACK_READS = THIS == FPU || THIS == SFPU || THIS == PACK;
constexpr bool TRACK_SFPU  = THIS == FPU || THIS == SFPU;
constexpr bool TRACK_COUNT = THIS == UNP || THIS == FPU;

constexpr std::uint32_t SEM_BANK    = 1;
constexpr std::uint32_t SEM_MAX     = 15;
constexpr std::uint32_t FREE_COUNTS = 2;

constexpr std::uint32_t WAIT_RES[]    = {p_stall::UNPACK0, p_stall::MATH, p_stall::WAIT_SFPU, p_stall::PACK};
constexpr std::uint32_t STALL_CLASS[] = {p_stall::STALL_UNPACK, p_stall::STALL_MATH, p_stall::STALL_SFPU | p_stall::STALL_MATH, p_stall::STALL_PACK};

constexpr std::uint32_t bit(const std::uint32_t client)
{
    return 1u << client;
}

struct State
{
    std::uint8_t holder;
    std::uint8_t members;
    std::uint8_t leaving;
    std::uint8_t head;
    std::uint8_t bank;
    std::uint8_t bank_flip;
    std::uint8_t per_use;
    std::uint8_t early;
    std::uint8_t reads_only;
    std::uint8_t sfpu_posted_pack;
    std::uint8_t acks;
    std::uint16_t base;
    std::uint32_t zeroacc;
};

inline State state;

inline void wait(std::uint32_t n = 1)
{
    for (; n > 0; n--)
    {
        TTI_SEMWAIT(STALL_CLASS[THIS] | p_stall::STALL_SYNC, p_stall::STALL_ON_ZERO, SEM_BANK, bit(THIS));
        TTI_SEMGET(SEM_BANK, bit(THIS));
    }
}

inline void signal(const std::uint32_t to_mask, std::uint32_t n = 1)
{
    for (; n > 0; n--)
    {
        TT_SEMPOST(SEM_BANK, to_mask);
    }
}

inline void drain()
{
    TTI_STALLWAIT(p_stall::STALL_SYNC, p_stall::NOTHING, p_stall::NOTHING, WAIT_RES[THIS]);
}

inline std::uint32_t fpu_next(const std::uint32_t members)
{
    return bit(PACK) >> ((members >> SFPU) & 1);
}

inline void take_acks()
{
    if (const std::uint32_t acks = state.acks)
    {
        wait(acks);
        state.acks = 0;
    }
}

inline void receive()
{
    if constexpr (THIS == FPU)
    {
        take_acks();
    }
    wait();
}

inline void receive_head(const std::uint32_t to)
{
    if constexpr (TRACK_COUNT)
    {
        if (to == THIS)
        {
            if (state.head == THIS)
            {
                const std::uint32_t free = FREE_COUNTS - state.per_use;
                wait(free);
                receive();
                signal(bit(THIS), free);
            }
            else
            {
                receive();
            }
        }
    }
}

template <std::uint32_t C>
inline void turn(const std::uint32_t from)
{
    State& s                    = state;
    const std::uint32_t members = s.members;
    const std::uint32_t sfpu_in = members & bit(SFPU);
    std::uint32_t reads_to_pack = 0;
    if constexpr (C == PACK && TRACK_READS)
    {
        reads_to_pack = s.reads_only & ((from == SFPU) | ((from == FPU) & !sfpu_in));
        if (reads_to_pack)
        {
            s.early = from;
        }
    }
    if constexpr (C == PACK && TRACK_SFPU)
    {
        s.sfpu_posted_pack |= sfpu_in != 0;
    }
    if constexpr (THIS == UNP)
    {
        if (from == UNP)
        {
            drain();
            TTI_SEMPOST(SEM_BANK, bit(FPU));
        }
    }
    else if constexpr (THIS == FPU)
    {
        if (from == FPU)
        {
            if (!reads_to_pack)
            {
                drain();
            }
            TT_SEMPOST(SEM_BANK, fpu_next(members));
            return;
        }
    }
    else if constexpr (THIS == SFPU)
    {
        if constexpr (C == SFPU)
        {
            wait();
        }
        else if constexpr (C == PACK)
        {
            if (from == SFPU)
            {
                if (!reads_to_pack)
                {
                    drain();
                }
                TTI_SEMPOST(SEM_BANK, bit(PACK));
            }
            else if (sfpu_in)
            {
                wait();
                TTI_SEMPOST(SEM_BANK, bit(PACK));
            }
        }
    }
    else if constexpr (THIS == PACK)
    {
        if constexpr (C == PACK)
        {
            wait();
        }
    }
    if constexpr (THIS == FPU)
    {
        if (from == UNP)
        {
            receive_head(FPU);
            if constexpr (C > FPU)
            {
                TT_SEMPOST(SEM_BANK, fpu_next(members));
            }
        }
    }
}

inline void transfer(const std::uint32_t old, const std::uint32_t head, const std::uint32_t counts)
{
    if constexpr (THIS == FPU)
    {
        if (old == FPU)
        {
            take_acks();
        }
    }
    if (old == THIS)
    {
        for (std::uint32_t n = FREE_COUNTS - counts; n > 0; n--)
        {
            wait();
            signal(bit(head));
        }
    }
}

inline std::uint32_t boundary(const std::uint32_t counts)
{
    State& s                = state;
    const std::uint32_t old = s.head;
    std::uint32_t members   = s.members;
    if (const std::uint32_t gone = s.leaving)
    {
        if constexpr (TRACK_SFPU)
        {
            if ((gone & bit(SFPU)) && s.sfpu_posted_pack)
            {
                if constexpr (THIS == SFPU)
                {
                    TTI_SEMPOST(SEM_BANK, bit(FPU));
                }
                else
                {
                    s.acks++;
                }
            }
        }
        members &= ~gone;
        s.members = members;
        s.leaving = 0;
    }
    const std::uint32_t head = ~members & bit(UNP);
    s.head                   = head;
    if constexpr (THIS == PACK)
    {
        signal(bit(head), counts);
    }
    if constexpr (TRACK_COUNT)
    {
        if (old != head)
        {
            asm volatile("");
            transfer(old, head, counts);
        }
    }
    return head;
}

inline void close()
{
    State& s = state;
    if constexpr (TRACK_READS)
    {
        const std::uint32_t early = s.early;
        if (early != NONE)
        {
            if constexpr (THIS == PACK)
            {
                wait();
            }
            else if (early == THIS)
            {
                drain();
                TTI_SEMPOST(SEM_BANK, bit(PACK));
            }
            s.early = NONE;
        }
    }
    if constexpr (THIS == PACK)
    {
        TTI_STALLWAIT(p_stall::STALL_THREAD, p_stall::NOTHING, p_stall::NOTHING, p_stall::PACK);
        TT_INSN(s.zeroacc | s.bank);
    }
}

template <std::uint32_t C>
inline void open(const std::uint32_t counts)
{
    State& s                 = state;
    const std::uint32_t head = boundary(counts);
    s.bank ^= s.bank_flip;
    if constexpr (TRACK_COUNT)
    {
        if (head == THIS)
        {
            if constexpr (THIS == FPU)
            {
                take_acks();
            }
            wait(s.per_use);
        }
    }
    if constexpr (C != UNP)
    {
        if (head == UNP)
        {
            if constexpr (THIS == UNP)
            {
                TTI_SEMPOST(SEM_BANK, bit(FPU));
            }
            if constexpr (THIS == FPU)
            {
                receive();
            }
        }
        if constexpr (C == SFPU)
        {
            if constexpr (THIS == FPU)
            {
                TTI_SEMPOST(SEM_BANK, bit(SFPU));
            }
            if constexpr (THIS == SFPU)
            {
                wait();
            }
        }
    }
}

__attribute__((noinline, cold)) inline void wrap(std::uint32_t from, const std::uint32_t taker)
{
    State& s = state;
    for (bool had_turn = true; from != taker; had_turn = false)
    {
        const std::uint32_t members = s.members;
        const std::uint32_t to      = from == PACK ? ~members & bit(UNP) : from == FPU && !(members & bit(SFPU)) ? PACK : from + 1;
        if constexpr (THIS != NONE)
        {
            if (from == THIS)
            {
                if (had_turn)
                {
                    drain();
                }
                signal(bit(to));
            }
        }
        if constexpr (TRACK_SFPU)
        {
            s.sfpu_posted_pack |= from == SFPU && to == PACK;
        }
        if (to <= FPU)
        {
            receive_head(to);
        }
        else if constexpr (THIS != NONE)
        {
            if (to == THIS)
            {
                wait();
            }
        }
        from = to;
    }
}

inline void program_base(const std::uint32_t base)
{
    state.base                  = base;
    trisc::dest_register_offset = base;
    if constexpr (THIS == FPU)
    {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::NOTHING, p_stall::NOTHING, p_stall::MATH);
        cfg_rmw(DEST_TARGET_REG_CFG_MATH_SEC1_Offset_RMW, base);
    }
    else if constexpr (THIS == SFPU)
    {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::NOTHING, p_stall::NOTHING, p_stall::WAIT_SFPU);
        cfg_rmw(DEST_TARGET_REG_CFG_MATH_SEC3_Offset_RMW, base);
    }
    else if constexpr (THIS == PACK)
    {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::NOTHING, p_stall::NOTHING, p_stall::PACK);
        trisc::_set_packer_dest_registers_<p_pacr::PACK0, DstSync::SyncHalf>();
    }
}

template <std::uint32_t C, bool EN_32BIT_DEST>
inline void own_base()
{
    if constexpr (C == THIS)
    {
        constexpr std::uint32_t BANK1_BASE = EN_32BIT_DEST ? trisc::DEST_REGISTER_HALF_SIZE >> 1 : trisc::DEST_REGISTER_HALF_SIZE;
        const std::uint32_t base           = -static_cast<std::uint32_t>(state.bank) & BANK1_BASE;
        if (state.base != base)
        {
            program_base(base);
        }
    }
}

template <std::uint32_t C, bool EN_32BIT_DEST, bool READ_ONLY>
__attribute__((noinline, cold)) void revisit(const std::uint32_t holder)
{
    wrap(holder, C);
    state.holder = C;
    if constexpr (TRACK_READS)
    {
        state.reads_only = READ_ONLY;
    }
    own_base<C, EN_32BIT_DEST>();
}

} // namespace dst_sync

inline void _llk_dest_sync_boot_init_()
{
    using namespace dst_sync;
    TTI_SEMINIT(SEM_MAX, 0, SEM_BANK, bit(UNP));
    TTI_SEMINIT(SEM_MAX, FREE_COUNTS, SEM_BANK, bit(FPU));
    TTI_SEMINIT(SEM_MAX, 0, SEM_BANK, bit(SFPU));
    TTI_SEMINIT(SEM_MAX, 0, SEM_BANK, bit(PACK));
}

template <DstSync DST, bool EN_32BIT_DEST>
inline void _llk_dest_sync_init_()
{
    using namespace dst_sync;
    constexpr bool SYNC_FULL = DST == DstSync::SyncFull;
    State& s                 = state;
    s                        = State {};
    s.holder                 = NONE;
    s.members                = bit(FPU) | bit(PACK);
    s.head                   = FPU;
    s.bank_flip              = !SYNC_FULL;
    s.bank                   = !SYNC_FULL;
    s.per_use                = SYNC_FULL ? FREE_COUNTS : 1;
    s.early                  = NONE;
    s.zeroacc                = TT_OP_ZEROACC(SYNC_FULL ? p_zeroacc::CLR_ALL : p_zeroacc::CLR_HALF, EN_32BIT_DEST, 0, ADDR_MOD_7, 0);
    if constexpr (THIS == PACK)
    {
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 0},
        }
            .set(ADDR_MOD_7);
    }
    if constexpr (THIS != NONE)
    {
        program_base(0);
    }
}

template <DestClient CLIENT>
inline void _llk_dest_sync_join_()
{
    using namespace dst_sync;
    static_assert(CLIENT == DestClient::UNPACK || CLIENT == DestClient::SFPU, "only UNPACK and SFPU join and leave");
    constexpr std::uint32_t C = to_underlying(CLIENT);
    State& s                  = state;
    if constexpr (C == SFPU && TRACK_SFPU)
    {
        if (!(s.members & bit(SFPU)))
        {
            s.sfpu_posted_pack = false;
        }
    }
    s.members |= bit(C);
    s.leaving &= ~bit(C);
}

template <DestClient CLIENT>
inline void _llk_dest_sync_leave_()
{
    using namespace dst_sync;
    static_assert(CLIENT == DestClient::UNPACK || CLIENT == DestClient::SFPU, "only UNPACK and SFPU join and leave");
    state.leaving |= state.members & bit(to_underlying(CLIENT));
}

template <DestClient CLIENT, bool EN_32BIT_DEST, bool READ_ONLY = (CLIENT == DestClient::PACK)>
void _llk_dest_sync_take_()
{
    using namespace dst_sync;
    static_assert(CLIENT != DestClient::UNPACK || !READ_ONLY, "unpack to dest only writes");
    static_assert(CLIENT != DestClient::PACK || READ_ONLY, "PACK only reads dest");
    constexpr std::uint32_t C = to_underlying(CLIENT);
    State& s                  = state;
    LLK_ASSERT((s.members & ~s.leaving) & bit(C), "a client must join before it takes dest");
    const std::uint32_t holder = s.holder;
    if (holder != C)
    {
        if constexpr (C == PACK)
        {
            LLK_ASSERT(holder != NONE, "PACK only reads dest, so it cannot open a use");
            turn<C>(holder);
        }
        else if (holder < C)
        {
            turn<C>(holder);
        }
        else if (holder == PACK)
        {
            close();
            open<C>(s.per_use);
        }
        else if (C == SFPU || holder == NONE)
        {
            open<C>(0);
        }
        else
        {
            return revisit<C, EN_32BIT_DEST, READ_ONLY>(holder);
        }
        s.holder = C;
        if constexpr (TRACK_READS)
        {
            s.reads_only = READ_ONLY;
        }
    }
    else if constexpr (!READ_ONLY && TRACK_READS)
    {
        s.reads_only = false;
    }
    own_base<C, EN_32BIT_DEST>();
}

inline void _llk_dest_sync_finish_()
{
    using namespace dst_sync;
    State& s                   = state;
    const std::uint32_t holder = s.holder;
    std::uint32_t counts       = 0;
    if (holder != NONE)
    {
        if (holder != PACK)
        {
            turn<PACK>(holder);
        }
        close();
        s.holder = NONE;
        counts   = s.per_use;
    }
    s.leaving = s.members & (bit(UNP) | bit(SFPU));
    boundary(counts);
    if constexpr (THIS == FPU)
    {
        take_acks();
        wait(FREE_COUNTS);
        signal(bit(FPU), FREE_COUNTS);
    }
    LLK_ASSERT(s.head == FPU, "dest sync did not return to rest");
}

} // namespace ckernel
