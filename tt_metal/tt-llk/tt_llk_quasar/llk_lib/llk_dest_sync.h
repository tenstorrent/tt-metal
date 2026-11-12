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

// Name each allocation with a constexpr object; aliases must use the same object.
struct DestTensor
{
    std::uint32_t address; // Absolute Dest row, like LLK2 Tensor::address.
    std::uint32_t rows;
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

// IDs 16..27: one poster and one getter per directed client pair, shared in program order.
constexpr std::uint32_t FIRST_SEMAPHORE = 16;
constexpr std::uint32_t MAX_CREDITS     = 15;

constexpr std::uint32_t semaphore_id(const std::uint32_t from, const std::uint32_t to)
{
    return FIRST_SEMAPHORE + 3 * from + to - (to > from);
}

constexpr std::uint32_t STALL[] = {
    p_stall::STALL_UNPACK | p_stall::STALL_MATH | p_stall::STALL_SYNC,
    p_stall::STALL_MATH | p_stall::STALL_SYNC,
    p_stall::STALL_SFPU | p_stall::STALL_MATH | p_stall::STALL_SYNC,
    p_stall::STALL_PACK | p_stall::STALL_MATH | p_stall::STALL_SYNC,
    p_stall::STALL_SYNC};

constexpr std::uint32_t DONE[] = {p_stall::UNPACK0, p_stall::MATH, p_stall::WAIT_SFPU, p_stall::PACK, p_stall::NOTHING};

struct State
{
    std::uint32_t epoch       = 0;
    std::uint32_t owner       = NONE;
    std::uint32_t released_to = NONE;
};

template <const DestTensor& TENSOR>
inline State state;

inline std::uint32_t epoch = 0;
inline bool fp32           = false;

inline void drain()
{
    TTI_STALLWAIT(p_stall::STALL_SYNC | p_stall::STALL_MATH, p_stall::NOTHING, p_stall::NOTHING, DONE[ME]);
}

template <std::uint32_t FROM, std::uint32_t TO>
inline void post()
{
    static_assert(FROM < NONE && TO < NONE && FROM != TO);
    constexpr auto id = semaphore_id(FROM, TO);
    TTI_SEMWAIT(STALL[ME], p_stall::STALL_ON_MAX, id / 8, 1u << (id % 8));
    TTI_SEMPOST(id / 8, 1u << (id % 8));
}

inline void take(const std::uint32_t from, const std::uint32_t to)
{
    const auto id = semaphore_id(from, to);
    TT_SEMWAIT(STALL[ME], p_stall::STALL_ON_ZERO, id / 8, 1u << (id % 8));
    TT_SEMGET(id / 8, 1u << (id % 8));
}

inline void program_base(const std::uint32_t row)
{
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::NOTHING, p_stall::NOTHING, DONE[ME]);
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

} // namespace dst_sync

/**
 * @brief Initialize the twelve Dest channels and clear Dest once per kernel on all four TRISCs.
 * @note Start with empty channels and Dest dvalid disabled; use one compatible Dest width/encoding with SFPU on TRISC3.
 */
template <DstSync DST, bool EN_32BIT_DEST>
inline void _llk_dest_sync_configure_()
{
    using namespace dst_sync;
    static_assert(DST == DstSync::SyncHalf || DST == DstSync::SyncFull);
    if constexpr (ME == PACK)
    {
        drain();
        TTI_ZEROACC(p_zeroacc::CLR_ALL, EN_32BIT_DEST, 0, ADDR_MOD_7, 0);
        TTI_STALLWAIT(p_stall::STALL_SYNC, p_stall::NOTHING, p_stall::NOTHING, p_stall::MATH);
        for (std::uint32_t id = FIRST_SEMAPHORE; id < FIRST_SEMAPHORE + 12; ++id)
        {
            TT_SEMINIT(MAX_CREDITS, 0, id / 8, 1u << (id % 8));
        }
        post<PACK, UNPACK>();
        post<PACK, FPU>();
        post<PACK, SFPU>();
        for (std::uint32_t client = UNPACK; client < PACK; ++client)
        {
            take(client, PACK);
        }
    }
    else if constexpr (ME < PACK)
    {
        // Do not arm a Tensix wait while PACK may still SEMINIT its channel.
        while (semaphore_read(semaphore_id(PACK, ME)) == 0)
        {
        }
        take(PACK, ME);
        post<ME, PACK>();
    }
    fp32 = EN_32BIT_DEST;
    ++epoch;
    program_base(0);
}

/**
 * @brief Acquire TENSOR and select its Dest address without releasing other tensors owned by CLIENT.
 * @note All TRISCs must make the same calls in the same order; use disjoint allocations and submit work before its next handoff.
 */
template <DestClient CLIENT, const DestTensor& TENSOR>
inline __attribute__((always_inline)) void _llk_dest_sync_take_()
{
    using namespace dst_sync;
    constexpr auto taker = to_underlying(CLIENT);
    static_assert(taker < NONE);
    static_assert(TENSOR.rows > 0 && TENSOR.address + TENSOR.rows <= trisc::DEST_REGISTER_FULL_SIZE, "Tensor exceeds Dest capacity");
    static_assert(CLIENT != DestClient::UNPACK || TENSOR.address % 64 == 0, "Legacy unpack-to-Dest requires full-tile bases");
    LLK_ASSERT(TENSOR.address + TENSOR.rows <= (trisc::DEST_REGISTER_FULL_SIZE >> fp32), "Tensor exceeds configured Dest capacity");
    auto& s = state<TENSOR>;
    if (s.epoch != epoch)
    {
        s = {epoch, NONE};
    }
    LLK_ASSERT(taker != PACK || s.owner != NONE, "PACK cannot produce a Dest tensor");
    LLK_ASSERT(s.released_to == NONE || s.released_to == taker, "Take must match the tensor's scheduled next writer");
    if (s.owner != NONE && s.owner != taker)
    {
        if constexpr (ME < NONE && ME != taker)
        {
            if (ME == s.owner && s.released_to == NONE)
            {
                drain();
                post<ME, taker>();
            }
        }
        if constexpr (ME == taker)
        {
            take(s.owner, taker);
        }
    }
    if constexpr (ME == taker)
    {
        // Ownership may already be ours while another owned tensor is currently selected.
        if (trisc::dest_register_offset != TENSOR.address)
        {
            program_base(TENSOR.address);
        }
    }
    s.owner       = taker;
    s.released_to = NONE;
}

/** @brief Clear only TENSOR after its current owner's outstanding work completes, retaining ownership. */
template <const DestTensor& TENSOR>
inline void _llk_dest_sync_clear_()
{
    using namespace dst_sync;
    static_assert(TENSOR.address % 16 == 0 && TENSOR.rows % 16 == 0, "Clear requires complete Dest faces");
    LLK_ASSERT(state<TENSOR>.epoch == epoch && state<TENSOR>.owner != NONE, "Take the tensor before clearing it");
    LLK_ASSERT(state<TENSOR>.released_to == NONE, "A released tensor must be taken before clearing it");
    if (ME != state<TENSOR>.owner)
    {
        return;
    }
    drain();
    const auto capacity = trisc::DEST_REGISTER_FULL_SIZE >> fp32;
    if (TENSOR.address == 0 && TENSOR.rows == capacity)
    {
        TT_ZEROACC(p_zeroacc::CLR_ALL, fp32, 0, ADDR_MOD_7, 0);
    }
    else if (TENSOR.rows == capacity / 2 && TENSOR.address % TENSOR.rows == 0)
    {
        TT_ZEROACC(p_zeroacc::CLR_HALF, fp32, 0, ADDR_MOD_7, TENSOR.address / TENSOR.rows);
    }
    else
    {
        for (std::uint32_t face = TENSOR.address / 16; face < (TENSOR.address + TENSOR.rows) / 16; ++face)
        {
            TT_ZEROACC(p_zeroacc::CLR_16, fp32, 0, ADDR_MOD_7, face);
        }
    }
    TTI_STALLWAIT(p_stall::STALL_SYNC, p_stall::NOTHING, p_stall::NOTHING, p_stall::MATH);
}

/**
 * @brief Clear TENSOR and publish its free credit now; NEXT consumes it at its later take.
 * @note Match deferred takes to releases in FIFO order on each directed channel, including handoffs of other tensors.
 */
template <DestClient NEXT, const DestTensor& TENSOR>
inline void _llk_dest_sync_release_()
{
    using namespace dst_sync;
    constexpr auto next = to_underlying(NEXT);
    static_assert(next < PACK, "A fresh tensor needs a writer");
    LLK_ASSERT(state<TENSOR>.owner != next, "Release transfers the tensor to another client");
    _llk_dest_sync_clear_<TENSOR>();
    if constexpr (ME < NONE && ME != next)
    {
        if (ME == state<TENSOR>.owner)
        {
            post<ME, next>();
        }
    }
    state<TENSOR>.released_to = next;
}

} // namespace ckernel
