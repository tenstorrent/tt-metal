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
    NONE   = 4,
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

constexpr std::uint32_t CHANNEL_COUNT = 12;
constexpr std::uint32_t BOOT_SEM      = 12;

constexpr std::uint32_t channel(const std::uint32_t from, const std::uint32_t to)
{
    return 3 * from + to - (to > from);
}

constexpr std::uint32_t DONE[] = {p_stall::UNPACK0, p_stall::MATH, p_stall::WAIT_SFPU, p_stall::PACK, p_stall::NOTHING};

constexpr std::uint32_t MAX_HELD = 16;
inline std::uint32_t held_slots[MAX_HELD];
inline std::uint8_t held_cnt   = 0;
inline std::uint32_t dest_base = 0xFFFFFFF;

inline std::uint32_t slots_claimed  = 0;
inline std::uint32_t alloc_next_row = 0;

inline void drain()
{
    TTI_STALLWAIT(p_stall::STALL_SYNC | p_stall::STALL_MATH, p_stall::NOTHING, p_stall::NOTHING, DONE[ME]);
}

inline void spin_get(const std::uint8_t sem)
{
    while (semaphore_read(sem) == 0)
    {
    }
    TT_SEMGET(0, semaphore::t6_sem(sem));
}

inline void drain_post(const std::uint8_t sem)
{
    drain();
    TT_SEMPOST(0, semaphore::t6_sem(sem));
}

inline void program_base(const std::uint32_t row)
{
    if (dest_base == row)
    {
        return;
    }
    dest_base = row;
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::NOTHING, p_stall::NOTHING, DONE[ME]);
    trisc::dest_register_offset = row;
    if (ME == PACK)
    {
        cfg_rmw(THCON_PACKER0_REG0_SRC_ADDR_OFFSET_RMW, row);
    }
    if (ME == FPU)
    {
        cfg_rmw(DEST_TARGET_REG_CFG_MATH_SEC1_Offset_RMW, row);
    }
    if (ME == SFPU)
    {
        cfg_rmw(DEST_TARGET_REG_CFG_MATH_SEC3_Offset_RMW, row);
    }
}

inline bool held(const std::uint32_t slot)
{
    for (std::uint8_t i = 0; i < held_cnt; ++i)
    {
        if (held_slots[i] == slot)
        {
            return true;
        }
    }
    return false;
}

inline void hold(const std::uint32_t slot)
{
    if (!held(slot) && held_cnt < MAX_HELD)
    {
        held_slots[held_cnt++] = slot;
    }
}

inline void clear_held()
{
    if (held_cnt != 0)
    {
        drain();
        for (std::uint8_t i = 0; i < held_cnt; ++i)
        {
            const std::uint8_t counter = held_slots[i] >> 16;
            const std::uint8_t mask    = (ME == PACK) ? 0x3 : 0;
            TT_PUSH_TILES(mask, 1, counter);
        }
        held_cnt = 0;
    }
}

} // namespace dst_sync

template <std::uint32_t ROWS, std::uint8_t SLOTS>
struct DestTensor
{
    static constexpr std::uint32_t rows = ROWS;
    static constexpr std::uint8_t slots = SLOTS;
    static_assert(ROWS % 16 == 0, "Dest rows must be a multiple of the 16-row face");

    template <DestClient CLIENT>
    static inline __attribute__((always_inline)) void acquire()
    {
        using namespace dst_sync;
        constexpr auto client   = to_underlying(CLIENT);
        const std::uint8_t slot = uses % slots;
        std::uint8_t& owner     = owners[slot];
        const bool same         = owner == client;
        const bool fresh        = owner == NONE;

        clear_held();

        if (!same && ME == client)
        {
            program_base(Info::address + slot * rows);
            if (fresh)
            {
                TT_WAIT_TILES(1 /*stall the thread*/, 1, Info::counter);
                TT_POP_TILES(0, 1, Info::counter);
            }
            else
            {
                spin_get(channel(owner, client));
            }
        }
        if (!same && !fresh && ME == owner)
        {
            drain_post(channel(owner, client));
        }
        owner = client;
        if ((client == PACK || client == SFPU) && ME == client)
        {
            hold((Info::counter << 16) | Info::slot_base + slot);
        }
    }

    static inline void destroy()
    {
        dst_sync::clear_held();
        for (std::uint8_t s = 0; s < slots; ++s)
        {
            owners[s] = dst_sync::NONE;
        }
        ++uses;
    }

    static void allocate()
    {
        claim();
    }

    static constexpr std::uint32_t slot_count()
    {
        return slots;
    }

    static void seed()
    {
        claim();
        if (dst_sync::ME == dst_sync::PACK)
        {
            TT_PUSH_TILES(0, slots, Info::counter);
        }
    }

private:
    struct Info
    {
        inline static std::uint32_t address   = 0xFFFFFFF;
        inline static std::uint32_t slot_base = 0;
        inline static std::uint8_t counter    = 0;
    };

    inline static std::uint8_t owners[SLOTS ? SLOTS : 1] = {};
    inline static std::uint32_t uses                     = 0;

    static void claim()
    {
        if (Info::address != 0xFFFFFFF)
        {
            return;
        }
        LLK_ASSERT(dst_sync::ME != dst_sync::NONE, "DestTensor used on a non-client thread");
        LLK_ASSERT(dst_sync::alloc_next_row + rows * slots <= trisc::DEST_REGISTER_FULL_SIZE, "Dest tensors exceed the Dest capacity");
        for (std::uint8_t s = 0; s < slots; ++s)
        {
            owners[s] = dst_sync::NONE;
        }
        Info::address   = dst_sync::alloc_next_row;
        Info::slot_base = dst_sync::slots_claimed;
        Info::counter   = dst_sync::tensors_claimed;
        dst_sync::alloc_next_row += rows * slots;
        dst_sync::slots_claimed += slots;
        dst_sync::tensors_claimed += 1;
    }
};

inline void _llk_dest_sync_configure_()
{
    using namespace dst_sync;
    if (ME == PACK)
    {
        for (std::uint32_t id = 0; id < CHANNEL_COUNT; ++id)
        {
            t6_semaphore_init(id, 0, 15);
        }
        t6_semaphore_init(BOOT_SEM, 0, 1);
        t6_semaphore_post(BOOT_SEM);
    }
    else
    {
        while (semaphore_read(BOOT_SEM) == 0)
        {
        }
    }
    dest_base = 0xFFFFFFF;
}

inline void _llk_dest_sync_flush_()
{
    dst_sync::clear_held();
}

} // namespace ckernel
