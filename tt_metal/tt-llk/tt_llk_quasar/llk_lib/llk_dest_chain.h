// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "ckernel_trisc_id.h"
#include "llk_assert.h"
#include "llk_sync.h"

// Dest handoffs between the units that use Dest, on Tensix semaphores (the SyncExu ones, private to this Tensix).
// Units: UNPACK (unpack-to-dest), FPU, SFPU (on the SFPU thread), PACK.
//   * Every thread runs every chain call in the same program order and keeps the same state; a thread issues
//     instructions only for its own unit.
//   * A section (first take .. release) uses one slot: Dest bank 0 / 1 alternately in SyncHalf, all of Dest in SyncFull.
//   * A take by a unit that does not hold the slot hands it over: the holder drains its engine and posts GO[taker][slot],
//     the taker waits for the post and takes it (SEMWAIT + SEMGET).
//   * Release hands the slot back to the section's first taker (its head) and parks the post there, so the next section
//     on the slot starts as soon as the last unit is done. At kernel start MATH owns every slot without a post.
//   * _llk_dest_chain_end_ takes the parked posts, so every chain semaphore is 0 between kernels.
//   * A wait on a semaphore that was never SEMINIT'ed is a hardware fault, so firmware boot initializes them once
//     (_llk_dest_chain_boot_); kernels never re-initialize them, as that could erase another thread's post.

namespace ckernel
{

namespace dest_chain
{

constexpr dest_dvalid_client NONE = static_cast<dest_dvalid_client>(4);

#if defined(LLK_TRISC_UNPACK) || defined(TRISC_UNPACK)
constexpr dest_dvalid_client THIS_UNIT = dest_dvalid_client::UNPACK;
#elif defined(LLK_TRISC_MATH) || defined(TRISC_MATH)
constexpr dest_dvalid_client THIS_UNIT = dest_dvalid_client::FPU;
#elif defined(LLK_TRISC_ISOLATE_SFPU) || defined(TRISC_ISOLATE_SFPU)
constexpr dest_dvalid_client THIS_UNIT = dest_dvalid_client::SFPU;
#elif defined(LLK_TRISC_PACK) || defined(TRISC_PACK)
constexpr dest_dvalid_client THIS_UNIT = dest_dvalid_client::PACK;
#else
constexpr dest_dvalid_client THIS_UNIT = NONE;
#endif

// GO[unit][slot] = Tensix semaphore 12 + 4 * slot + unit (ids 12..19; 0..7 are legacy, 8..11 the tt-llk perf barrier).
constexpr std::uint32_t GO_BASE = 12;

inline std::uint32_t go(const std::uint32_t unit, const std::uint32_t slot)
{
    return GO_BASE + 4 * slot + unit;
}

struct UnitInfo
{
    std::uint32_t wait_class; // SEMWAIT stall class: the engine that must not touch Dest before the post
    std::uint32_t drain0;     // engines to drain before posting
    std::uint32_t drain1;
};

constexpr UnitInfo UNITS[] = {
    {p_stall::STALL_UNPACK, p_stall::UNPACK0, p_stall::NOTHING},
    {p_stall::STALL_MATH, p_stall::MATH, p_stall::WAIT_SFPU},
    {p_stall::STALL_SFPU | p_stall::STALL_MATH, p_stall::WAIT_SFPU, p_stall::NOTHING},
    {p_stall::STALL_PACK, p_stall::PACK0, p_stall::NOTHING},
};

// Units are stored as unit + 1; 0 means none.
constexpr std::uint8_t code(const dest_dvalid_client unit)
{
    return static_cast<std::uint8_t>(unit) + 1;
}

// No member initializers: in metal the state is per-thread zero-init storage (.tbss); _llk_dest_chain_end_ zeroes it.
struct State
{
    std::uint8_t holder;    // unit holding the open section's slot, 0 between sections
    std::uint8_t head;      // first taker of the open section
    std::uint8_t bank;      // slot of the open (or next) section: its Dest bank in SyncHalf, 0 in SyncFull
    std::uint8_t parked[2]; // per slot, the unit whose GO holds the slot's post; 0: MATH owns it, nothing posted
    std::uint8_t members;   // joined units, one bit each
};

// One instance per hardware TRISC thread, like trisc::bfd_state (llk_bfd_alloc.h): ENV_LLK_INFRA uses a plain
// global; the metal build declares it extern thread_local, defined zero-init in tt_metal/hw/firmware/src/tt-2xx/trisc.cc
// (a non-zero .tdata image breaks Quasar device init).
#ifdef ENV_LLK_INFRA
inline State state = {};
#else
extern thread_local State state;
#endif

inline void post(const std::uint32_t id)
{
    TT_SEMPOST(id >> 3, 1u << (id & 7));
}

/**
 * @brief Waits for UNIT's engines to finish their Dest accesses.
 */
template <dest_dvalid_client UNIT>
inline void drain()
{
    constexpr UnitInfo INFO = UNITS[static_cast<int>(UNIT)];
    TTI_STALLWAIT(p_stall::STALL_SYNC, INFO.drain1, p_stall::NOTHING, INFO.drain0);
}

/**
 * @brief Waits for a post to UNIT on `slot` and takes it. UNIT's engine is held until then; the SEMGET always follows
 *        its SEMWAIT, so no thread ends a kernel on a pending wait.
 */
template <dest_dvalid_client UNIT>
inline void wait(const std::uint32_t slot)
{
    const std::uint32_t id = go(static_cast<std::uint32_t>(UNIT), slot);
    TT_SEMWAIT(UNITS[static_cast<int>(UNIT)].wait_class | p_stall::STALL_SYNC, p_stall::STALL_ON_ZERO, id >> 3, 1u << (id & 7));
    TT_SEMGET(id >> 3, 1u << (id & 7));
}

/**
 * @brief Points UNIT at the other Dest bank (SyncHalf), once UNIT's engine is done with the current one.
 */
template <dest_dvalid_client UNIT, bool EN_32BIT_DEST>
inline void flip_bank()
{
    if constexpr (UNIT == dest_dvalid_client::UNPACK)
    {
        // Folded into each unpack Dest tile index. The UNP_DEST tile counter finds bank 1 at the 16-bit stride in
        // 32-bit mode too (emulator, 2026-10-30: the 32-bit stride lands bank-1 tiles where pack reads zeros).
        _update_dest_register_offset_<false>();
    }
    else if constexpr (UNIT == dest_dvalid_client::FPU)
    {
        _llk_sync_advance_dest_section_<TRISC_ID, EN_32BIT_DEST, p_stall::MATH>();
    }
    else if constexpr (UNIT == dest_dvalid_client::SFPU)
    {
        _llk_sync_advance_dest_section_<TRISC_ID, EN_32BIT_DEST, p_stall::WAIT_SFPU>();
    }
    else
    {
        _update_dest_register_offset_<EN_32BIT_DEST>();
        _llk_stall_cfg_on_<p_stall::PACK0>();
        _set_packer_dest_registers_<p_pacr::PACK0, DstSync::SyncHalf>();
    }
}

} // namespace dest_chain

/**
 * @brief Firmware boot, one thread, before any kernel: initialize every chain semaphore to 0 (max 15).
 */
inline void _llk_dest_chain_boot_()
{
    for (std::uint32_t id = dest_chain::GO_BASE; id < dest_chain::GO_BASE + 8; ++id)
    {
        TT_SEMINIT(15, 0, id >> 3, 1u << (id & 7));
    }
}

/**
 * @brief Add UNIT to the dest chain.
 * @note Call on every thread from op init.
 */
template <dest_dvalid_client UNIT>
inline void _llk_dest_chain_join_()
{
    dest_chain::state.members |= 1u << static_cast<int>(UNIT);
}

/**
 * @brief Take dest for UNIT. Nothing happens when UNIT already holds it. Otherwise the current holder hands it over: the
 *        holder of the open section, or, for a section's first take, the unit the last release parked it with (MATH at
 *        kernel start).
 *
 * @note Call on every thread before UNIT's dest accesses, after @ref _llk_dest_chain_join_. Take per block of tiles,
 *       not per tile: every handoff drains an engine.
 */
template <dest_dvalid_client UNIT, DstSync DST, bool EN_32BIT_DEST = false>
inline void _llk_dest_chain_take_()
{
    using namespace dest_chain;
    State& s                  = state;
    constexpr std::uint8_t ME = code(UNIT);
    LLK_ASSERT(s.members & (1u << static_cast<int>(UNIT)), "a unit must join before taking dest");
    if (s.holder == ME)
    {
        return;
    }
    const std::uint32_t slot = s.bank;
    std::uint8_t from        = s.holder;
    bool queued              = false; // from's post is still in its GO
    if (from == 0)
    {
        s.head = ME;
        from   = s.parked[slot];
        queued = from != 0;
        if (!queued)
        {
            from = code(dest_dvalid_client::FPU);
        }
    }
    if constexpr (THIS_UNIT != NONE)
    {
        if (from != ME && from == code(THIS_UNIT))
        {
            if (queued)
            {
                wait<THIS_UNIT>(slot);
            }
            else
            {
                drain<THIS_UNIT>();
            }
            post(go(static_cast<std::uint32_t>(UNIT), slot));
        }
    }
    if constexpr (UNIT == THIS_UNIT)
    {
        if (from != ME || queued)
        {
            wait<UNIT>(slot);
        }
        if constexpr (DST == DstSync::SyncHalf)
        {
            if ((_get_dest_buffer_base_() != 0) != (slot != 0))
            {
                dest_chain::flip_bank<UNIT, EN_32BIT_DEST>();
            }
        }
    }
    s.holder = ME;
}

/**
 * @brief End the open section: the holder drains and parks the slot with the section's head. In SyncHalf the next
 *        section uses the other bank. Nothing happens for an empty section.
 * @note Call on every thread.
 */
template <DstSync DST>
inline void _llk_dest_chain_release_()
{
    using namespace dest_chain;
    State& s = state;
    if (s.holder == 0)
    {
        return;
    }
    if constexpr (THIS_UNIT != NONE)
    {
        if (s.holder == code(THIS_UNIT))
        {
            drain<THIS_UNIT>();
            post(go(s.head - 1u, s.bank));
        }
    }
    s.parked[s.bank] = s.head;
    s.holder         = 0;
    if constexpr (DST == DstSync::SyncHalf)
    {
        s.bank ^= 1;
    }
}

/**
 * @brief Kernel end: take the posts parked with this thread's unit, so every chain semaphore is back at 0, and zero
 *        the state for the next kernel.
 * @note Call on every thread after the kernel, before the firmware reports it done.
 */
inline void _llk_dest_chain_end_()
{
    using namespace dest_chain;
    State& s = state;
    LLK_ASSERT(s.holder == 0, "dest_acquire without dest_release at kernel end");
    if constexpr (THIS_UNIT != NONE)
    {
        for (std::uint32_t slot = 0; slot < 2; ++slot)
        {
            if (s.parked[slot] == code(THIS_UNIT))
            {
                wait<THIS_UNIT>(slot);
            }
        }
    }
    s = {};
}

} // namespace ckernel
