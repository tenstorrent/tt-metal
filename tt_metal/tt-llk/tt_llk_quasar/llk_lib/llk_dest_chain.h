// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_dest_dvalid.h"

namespace ckernel
{

namespace dest_chain
{

using dest_dvalid::NONE;
using dest_dvalid::THIS_UNIT;

// Each thread only acts for its own unit, so the copies agree on the holder and the use, and on the turns of the
// units a thread tracks.
struct State
{
    dest_dvalid_client holder = NONE;
    bool holder_has_bit       = false; // on the holder's thread, false while it was admitted on an empty bank
    bool gated                = false; // this thread's unit has a wait word, false at boot and after a leave
    std::uint8_t members      = 0;
    std::uint8_t turned       = 0;          // tracked units with a turn in the current bank use
    std::uint8_t last_turned  = 0xF;        // tracked units with a turn in the previous bank use (boot orders everyone)
    std::uint8_t use          = 0;          // bank of the current use in SyncHalf
    std::uint8_t bank         = 0;          // bank of this thread's unit
};

inline State state;

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
    constexpr bool KNOWN            = NEXT == THIS_UNIT || THIS_UNIT == dest_dvalid_client::PACK;
    constexpr std::uint8_t NEXT_BIT = dest_dvalid::bit_of(NEXT);
    if constexpr (THIS_UNIT == dest_dvalid_client::PACK)
    {
        _llk_dest_dvalid_end_use_();
        state.bank ^= 1;
    }
    // Bank uses alternate, so the last use of the bank NEXT opens is the previous use.
    const bool granted = KNOWN && !(state.last_turned & NEXT_BIT);
    if constexpr (THIS_UNIT == dest_dvalid_client::PACK)
    {
        if (granted)
        {
            // PACK is now on the bank NEXT opens and has nothing in flight, so this pulse sets NEXT's bit at once.
            _llk_dest_dvalid_write_release_<dest_dvalid_client::PACK, dest_dvalid::bit_of(NEXT), true /*HOLD_BANK*/>();
            _llk_dest_dvalid_pulse_<dest_dvalid_client::PACK>();
        }
    }
    // The grant lands after this bank's release and before its next use, so NEXT may open that use on an empty check.
    state.last_turned = state.turned | (granted ? NEXT_BIT : 0);
    state.turned      = 0;
    state.holder      = NONE;
    state.use ^= 1;
    return granted;
}

} // namespace dest_chain

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
