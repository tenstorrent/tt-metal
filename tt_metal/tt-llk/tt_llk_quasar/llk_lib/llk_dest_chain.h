// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "llk_assert.h"
#include "llk_dest_dvalid.h"

namespace ckernel
{

namespace dest_chain
{

using dest_dvalid::NONE;
using dest_dvalid::THIS_UNIT;

// Each thread only acts for its own unit, so the copies agree on the holder and the use, and on the turns of the
// units a thread tracks.
// No member initializers: in metal the state is per-thread zero-init storage (.tbss), given initial_state() by
// _llk_dest_chain_reset_.
struct State
{
    dest_dvalid_client holder;
    bool holder_has_bit;     // on the holder's thread, false while it was admitted on an empty bank
    bool gated;              // this thread's unit has a wait word, false at boot and after a leave
    std::uint8_t members;
    std::uint8_t turned;     // tracked units with a turn in the current bank use
    std::uint8_t ordered[2]; // per bank, tracked units with a turn in its last use (boot orders everyone)
    std::uint8_t use;        // bank of the current use in SyncHalf
    std::uint8_t bank;       // bank of this thread's unit
};

constexpr State initial_state()
{
    return State {NONE, false, false, 0, 0, {0xF, 0xF}, 0, 0};
}

// One instance per hardware TRISC thread, like trisc::bfd_state (llk_bfd_alloc.h): ENV_LLK_INFRA uses a plain
// global; the metal build declares it extern thread_local, defined zero-init in tt_metal/hw/firmware/src/tt-2xx/trisc.cc
// (a non-zero .tdata image breaks Quasar device init), and _llk_dest_chain_reset_ gives it its initial value.
#ifdef ENV_LLK_INFRA
inline State state = initial_state();
#else
extern thread_local State state;
#endif

/**
 * @brief Select a bank on its owner's thread, preserving the bank's ownership bits.
 */
template <dest_dvalid_client UNIT>
inline void select_bank(const std::uint32_t target)
{
    if constexpr (UNIT == THIS_UNIT)
    {
        if (state.bank == target)
        {
            return;
        }
        if (target == 0)
        {
            TTI_CLEARDVALID(0, 0, 0, dest_dvalid::bit_of(UNIT), 0, 0);
        }
        else
        {
            _llk_dest_dvalid_bank_flip_<UNIT>();
            state.gated = false;
        }
        state.bank = target;
    }
}

template <dest_dvalid_client FROM, dest_dvalid_client TO, DstSync DST>
inline void signal()
{
    if constexpr (FROM != TO)
    {
        constexpr bool HOLD = DST == DstSync::SyncFull;
        if constexpr (FROM == THIS_UNIT)
        {
            if (state.holder_has_bit)
            {
                _llk_dest_dvalid_signal_<FROM, TO, true /*HAS_BIT*/, HOLD>();
            }
            else
            {
                _llk_dest_dvalid_signal_<FROM, TO, false /*HAS_BIT*/, HOLD>();
            }
            if constexpr (!HOLD)
            {
                state.bank ^= 1;
            }
        }
    }
}

/**
 * @brief The holder hands dest to TO. Only the holder's own thread issues anything.
 */
template <dest_dvalid_client TO, DstSync DST>
inline void pass_from_holder()
{
    if constexpr (THIS_UNIT != NONE && TO != THIS_UNIT)
    {
        if (state.holder == THIS_UNIT)
        {
            signal<THIS_UNIT, TO, DST>();
        }
    }
}

/**
 * @brief PACK's turn ended the use of a bank and NEXT opens the other one.
 *
 * NEXT waits for the bank to be empty when it had a turn in the last use of that bank. Otherwise an empty bank
 * could be the state before that use, so PACK hands NEXT its bit once PACK's reads are done.
 * @return true when NEXT was handed a bit, known on the threads that track NEXT
 */
template <dest_dvalid_client NEXT, DstSync DST>
inline bool end_use()
{
    constexpr bool KNOWN     = NEXT == THIS_UNIT || THIS_UNIT == dest_dvalid_client::PACK;
    const std::uint32_t bank = state.use;
    const std::uint32_t next = bank ^ 1;
    const bool granted       = KNOWN && !(state.ordered[next] & dest_dvalid::bit_of(NEXT));
    if constexpr (THIS_UNIT == dest_dvalid_client::PACK)
    {
        _llk_dest_dvalid_end_use_();
        state.bank ^= 1;
        if (granted)
        {
            // PACK is now on the bank NEXT opens and has nothing in flight, so this pulse sets NEXT's bit at once.
            _llk_dest_dvalid_write_release_<dest_dvalid_client::PACK, dest_dvalid::bit_of(NEXT), true /*HOLD_BANK*/>();
            _llk_dest_dvalid_pulse_<dest_dvalid_client::PACK>();
        }
    }
    state.ordered[bank] = state.turned;
    state.turned        = 0;
    state.holder        = NONE;
    state.use           = next;
    return granted;
}

} // namespace dest_chain

/**
 * @brief Kernel-start reset of the dest chain. Call on every thread (UNPACK, MATH, PACK, ISOLATE_SFPU) before any join.
 *
 * Resets this thread's chain bookkeeping. The UNPACK thread then does what the tt-llk test boot does: zeroes Dest,
 * clears every dvalid bit and bank id and the four dvalid control words a previous kernel left programmed, then raises
 * DEST_CHAIN_INIT to 3. Each other thread spins until it is raised and takes one count, so no thread joins or hands
 * dest to another unit before the reset has landed, and the semaphore is back at 0 once all three passed.
 */
inline void _llk_dest_chain_reset_()
{
    dest_chain::state = dest_chain::initial_state();
    if constexpr (dest_chain::THIS_UNIT == dest_dvalid_client::UNPACK)
    {
        TTI_ZEROACC(p_zeroacc::CLR_ALL, 0 /*16-bit*/, 0, 1, 0);
        TTI_ZEROACC(p_zeroacc::CLR_ALL, 1 /*32-bit*/, 0, 1, 0);
        _llk_dest_dvalid_boot_reset_();
        _llk_dest_dvalid_write_wait_<dest_dvalid_client::UNPACK, 0, 0>(false /*stall*/);
        _llk_dest_dvalid_write_wait_<dest_dvalid_client::FPU, 0, 0>(false /*stall*/);
        _llk_dest_dvalid_write_wait_<dest_dvalid_client::SFPU, 0, 0>(false /*stall*/);
        _llk_dest_dvalid_write_wait_<dest_dvalid_client::PACK, 0, 0>(false /*stall*/);
        _llk_dest_dvalid_write_release_<dest_dvalid_client::UNPACK, 0, false /*HOLD_BANK*/>();
        _llk_dest_dvalid_write_release_<dest_dvalid_client::FPU, 0, false /*HOLD_BANK*/>();
        _llk_dest_dvalid_write_release_<dest_dvalid_client::SFPU, 0, false /*HOLD_BANK*/>();
        _llk_dest_dvalid_write_release_<dest_dvalid_client::PACK, 0, false /*HOLD_BANK*/>();
        TTI_STALLWAIT(p_stall::STALL_SYNC, p_stall::NOTHING, p_stall::MATH, p_stall::CFGEXU);
        TTI_SEMINIT(3 /*max*/, 3 /*init*/, 0, trisc::semaphore::t6_sem(trisc::semaphore::DEST_CHAIN_INIT));
    }
    else if constexpr (dest_chain::THIS_UNIT != dest_chain::NONE)
    {
        while (semaphore_read(trisc::semaphore::DEST_CHAIN_INIT) == 0)
        {
        }
        trisc::t6_semaphore_get(trisc::semaphore::DEST_CHAIN_INIT);
    }
}

/**
 * @brief Add UNIT to the dest chain.
 * @note Call on every thread from op init and pair with @ref _llk_dest_chain_leave_.
 */
template <dest_dvalid_client UNIT>
inline void _llk_dest_chain_join_()
{
    constexpr std::uint32_t BIT = dest_dvalid::bit_of(UNIT);
    dest_chain::state.members |= BIT;
}

/**
 * @brief Remove UNIT from the dest chain.
 * @note Call on every thread outside a turn, paired with @ref _llk_dest_chain_join_. The thread keeps running the
 *       chain for the other units, so a handoff still owed by UNIT is issued.
 */
template <dest_dvalid_client UNIT>
inline void _llk_dest_chain_leave_()
{
    constexpr std::uint32_t BIT = dest_dvalid::bit_of(UNIT);
    if (!(dest_chain::state.members & BIT))
    {
        return;
    }
    if constexpr (UNIT == dest_dvalid::THIS_UNIT)
    {
        _llk_dest_dvalid_leave_<UNIT>();
        dest_chain::state.gated = false;
    }
    dest_chain::state.members &= ~BIT;
}

/**
 * @brief Take dest from the current holder.
 *
 * A take after PACK's turn in SyncHalf ends the use of PACK's bank and opens the other bank for the taker.
 * In SyncFull dest is one handoff list on bank 0 and any unit, PACK included, hands to the next taker.
 *
 * @note Call on every thread before UNIT's dest accesses, after @ref _llk_dest_chain_join_.
 * @note Issue at least one gated dest access in each turn before handing off to another unit.
 */
template <dest_dvalid_client UNIT, DstSync DST, bool EN_32BIT_DEST = false>
inline void _llk_dest_chain_take_()
{
    dest_chain::State& s = dest_chain::state;
    LLK_ASSERT(s.members & dest_dvalid::bit_of(UNIT), "a unit must join before taking dest");
    if (s.holder == UNIT)
    {
        return;
    }
    bool has_bit = false;
    if (DST == DstSync::SyncHalf && s.holder == dest_dvalid_client::PACK)
    {
        has_bit = dest_chain::end_use<UNIT, DST>();
    }
    if (s.holder != dest_chain::NONE)
    {
        dest_chain::pass_from_holder<UNIT, DST>();
        has_bit = true;
    }
    LLK_ASSERT(has_bit || UNIT != dest_dvalid_client::PACK, "PACK only reads dest, so it cannot open a bank use");
    dest_chain::select_bank<UNIT>(DST == DstSync::SyncFull ? 0 : s.use);
    if constexpr (UNIT == dest_dvalid::THIS_UNIT)
    {
        // A unit handed a bit waits for its own bit. A unit that opens a bank use waits for the bank to be empty.
        const bool stall = !s.gated || (has_bit && !s.holder_has_bit);
        if (has_bit)
        {
            _llk_dest_dvalid_write_wait_<UNIT, dest_dvalid::bit_of(UNIT), dest_dvalid::bit_of(UNIT)>(stall);
        }
        else
        {
            _llk_dest_dvalid_write_wait_<UNIT, dest_dvalid::ALL_BITS, 0 /*POLARITY*/>(stall);
        }
        s.holder_has_bit = has_bit;
        s.gated          = true;
    }
    if constexpr (UNIT == dest_dvalid::THIS_UNIT || dest_dvalid::THIS_UNIT == dest_dvalid_client::PACK)
    {
        s.turned |= dest_dvalid::bit_of(UNIT);
    }
    s.holder = UNIT;
}

} // namespace ckernel
