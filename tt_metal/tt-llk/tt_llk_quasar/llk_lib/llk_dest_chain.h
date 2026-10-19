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

constexpr std::uint32_t NUM_BANKS = 2;

struct State
{
    dest_dvalid_client publisher[NUM_BANKS] = {NONE, NONE};
    dest_dvalid_client holder               = NONE;
    std::uint32_t section                   = 0;
    std::uint8_t members                    = 0;
    std::uint8_t bank                       = 0;
};

inline State state;

template <dest_dvalid_client UNIT>
constexpr bool mine()
{
    return UNIT == THIS_UNIT;
}

constexpr bool can_start(const dest_dvalid_client unit)
{
    return unit != dest_dvalid_client::PACK && unit != NONE;
}

template <DstSync DST>
inline std::uint32_t section_bank()
{
    return DST == DstSync::SyncFull ? 0 : (state.section & 1);
}

/**
 * @brief Select a bank on its owner's thread, preserving the bank's ownership bits.
 */
template <dest_dvalid_client UNIT>
inline void select_bank(const std::uint32_t target)
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
    }
    state.bank = target;
}

template <DstSync DST>
inline void after_pulse()
{
    if constexpr (DST == DstSync::SyncHalf)
    {
        state.bank ^= 1;
    }
}

/**
 * @brief Publish the actual next head after the previous section's completion, on its terminal unit's thread.
 */
template <dest_dvalid_client FROM, dest_dvalid_client TO, bool HOLD_BANK>
inline void publish()
{
    _llk_dest_dvalid_write_release_<FROM, dest_dvalid::bit_of(TO), HOLD_BANK>();
    _llk_dest_dvalid_pulse_<FROM>();
}

template <dest_dvalid_client FROM, dest_dvalid_client TO>
inline void publish_on_bank(const std::uint32_t bank)
{
    const std::uint32_t saved = state.bank;
    select_bank<FROM>(bank);
    if (saved == bank)
    {
        publish<FROM, TO, true /*HOLD_BANK*/>();
    }
    else
    {
        publish<FROM, TO, false /*HOLD_BANK*/>();
    }
    state.bank = saved;
}

/**
 * @brief Admit HEAD using the previous terminal client's stream, never a guessed head's bit.
 */
template <dest_dvalid_client HEAD, DstSync DST>
inline void collect()
{
    const std::uint32_t bank           = section_bank<DST>();
    const dest_dvalid_client publisher = state.publisher[bank];
    if (publisher == NONE)
    {
        if constexpr (mine<HEAD>())
        {
            select_bank<HEAD>(bank);
            _llk_dest_dvalid_write_release_<HEAD, dest_dvalid::bit_of(HEAD), true /*HOLD_BANK*/>();
            _llk_dest_dvalid_pulse_<HEAD>();
        }
    }
    else if constexpr (THIS_UNIT != NONE)
    {
        if (publisher == THIS_UNIT)
        {
            publish_on_bank<THIS_UNIT, HEAD>(bank);
        }
    }
}

template <dest_dvalid_client UNIT, DstSync DST>
inline void arrive()
{
    if constexpr (mine<UNIT>())
    {
        select_bank<UNIT>(section_bank<DST>());
    }
}

template <dest_dvalid_client FROM, dest_dvalid_client TO, DstSync DST>
inline void signal()
{
    if constexpr (FROM != TO && mine<FROM>())
    {
        _llk_dest_dvalid_signal_<FROM, TO, DST>();
        after_pulse<DST>();
    }
}

template <dest_dvalid_client TO, DstSync DST>
inline void pass_from_holder()
{
    if constexpr (THIS_UNIT != NONE && !mine<TO>())
    {
        if (state.holder == THIS_UNIT)
        {
            signal<THIS_UNIT, TO, DST>();
        }
    }
    arrive<TO, DST>();
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
    if (dest_chain::state.members & BIT)
    {
        return;
    }
    if constexpr (dest_chain::mine<UNIT>())
    {
        _llk_dest_dvalid_join_<UNIT>();
        _llk_dest_dvalid_write_wait_<UNIT>();
        dest_chain::state.bank = 0;
    }
    dest_chain::state.members |= BIT;
}

/**
 * @brief Remove UNIT while preserving its role as publisher for the next head's take.
 * @note Call on every thread outside a section, paired with @ref _llk_dest_chain_join_.
 */
template <dest_dvalid_client UNIT>
inline void _llk_dest_chain_leave_()
{
    constexpr std::uint32_t BIT = dest_dvalid::bit_of(UNIT);
    if (!(dest_chain::state.members & BIT))
    {
        return;
    }
    if constexpr (dest_chain::mine<UNIT>())
    {
        _llk_dest_dvalid_leave_<UNIT>();
    }
    dest_chain::state.members &= ~BIT;
}

/**
 * @brief Start runtime bookkeeping for a section.
 */
inline void _llk_dest_chain_section_begin_()
{
    dest_chain::state.holder = dest_chain::NONE;
}

/**
 * @brief Take dest from the current holder or ask the previous terminal unit to publish it.
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
    if (s.holder == dest_chain::NONE)
    {
        if constexpr (dest_chain::can_start(UNIT))
        {
            dest_chain::collect<UNIT, DST>();
            dest_chain::arrive<UNIT, DST>();
        }
        else
        {
            LLK_ASSERT(false, "PACK only reads dest, so it cannot start a section");
        }
    }
    else
    {
        dest_chain::pass_from_holder<UNIT, DST>();
    }
    s.holder = UNIT;
}

/**
 * @brief Complete the section and remember which unit will publish its bank's next head.
 */
template <DstSync DST, bool EN_32BIT_DEST = false>
inline void _llk_dest_chain_section_end_()
{
    dest_chain::State& s = dest_chain::state;
    LLK_ASSERT(s.holder != dest_chain::NONE, "a section must end with a holder");
    if constexpr (dest_chain::THIS_UNIT != dest_chain::NONE)
    {
        if (s.holder == dest_chain::THIS_UNIT)
        {
            _llk_dest_dvalid_finish_<dest_chain::THIS_UNIT, DST, EN_32BIT_DEST>(s.bank);
            dest_chain::after_pulse<DST>();
        }
    }
    s.publisher[dest_chain::section_bank<DST>()] = s.holder;
    s.section++;
    s.holder = dest_chain::NONE;
}

} // namespace ckernel
