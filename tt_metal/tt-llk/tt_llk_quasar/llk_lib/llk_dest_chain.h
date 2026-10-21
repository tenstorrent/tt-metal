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

struct State
{
    dest_dvalid_client holder     = NONE;
    dest_dvalid_client head       = NONE;
    dest_dvalid_client pending[2] = {NONE, NONE}; // per bank, the head whose token the last section posted
    std::uint32_t section         = 0;
    std::uint8_t members          = 0;
    std::uint8_t bank             = 0;
    bool holder_is_head           = false; // false once the head takes again later in the section
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
        if (state.holder_is_head)
        {
            _llk_dest_dvalid_signal_<FROM, TO, DST, true /*HEAD*/>();
        }
        else
        {
            _llk_dest_dvalid_signal_<FROM, TO, DST, false /*HEAD*/>();
        }
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
}

template <dest_dvalid_client UNIT, DstSync DST>
inline void admit(const bool head)
{
    if constexpr (mine<UNIT>())
    {
        if (head)
        {
            _llk_dest_dvalid_admit_<UNIT, true /*HEAD*/, DST>();
        }
        else
        {
            _llk_dest_dvalid_admit_<UNIT, false /*HEAD*/, DST>();
        }
    }
}

/**
 * @brief PACK's turn ended the section: PACK clears its bit and posts the token of this section's head.
 */
template <DstSync DST>
inline void close_section()
{
    const std::uint32_t bank = section_bank<DST>();
    if constexpr (mine<dest_dvalid_client::PACK>())
    {
        _llk_dest_dvalid_finish_<dest_dvalid_client::PACK>(dest_dvalid::token_sem(state.head, bank));
        after_pulse<DST>();
    }
    state.pending[bank] = state.head;
    state.section++;
    state.holder = NONE;
}

/**
 * @brief Wait for the token of the previous section on this bank. A token posted to another unit is forwarded by it.
 */
template <dest_dvalid_client UNIT, DstSync DST>
inline void claim()
{
    const std::uint32_t bank          = section_bank<DST>();
    const dest_dvalid_client expected = state.pending[bank];
    if (expected == NONE)
    {
        return;
    }
    const std::uint8_t sem = dest_dvalid::token_sem(UNIT, bank);
    if constexpr (THIS_UNIT != NONE && !mine<UNIT>())
    {
        if (expected == THIS_UNIT)
        {
            _llk_dest_dvalid_token_forward_(dest_dvalid::token_sem(THIS_UNIT, bank), sem);
        }
    }
    if constexpr (mine<UNIT>())
    {
        _llk_dest_dvalid_token_wait_(sem);
    }
}

} // namespace dest_chain

/**
 * @brief Seed the token semaphores. Call once from one thread at firmware boot, before any section.
 */
inline void _llk_dest_chain_boot_init_()
{
    _llk_dest_dvalid_token_init_();
}

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
        dest_chain::state.bank = 0;
    }
    dest_chain::state.members |= BIT;
}

/**
 * @brief Remove UNIT from the dest chain.
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
 * @brief Take dest from the current holder.
 *
 * A take after PACK's turn starts a new section: PACK's bank is released and the taker waits as the head for the
 * previous section on the bank it moves to.
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
    if (s.holder == dest_dvalid_client::PACK)
    {
        dest_chain::close_section<DST>();
    }
    const bool head = s.holder == dest_chain::NONE;
    LLK_ASSERT(!head || dest_chain::can_start(UNIT), "PACK only reads dest, so it cannot start a section");
    if (head)
    {
        s.head = UNIT;
    }
    else
    {
        dest_chain::pass_from_holder<UNIT, DST>();
    }
    dest_chain::arrive<UNIT, DST>();
    dest_chain::admit<UNIT, DST>(head);
    if (head)
    {
        dest_chain::claim<UNIT, DST>();
    }
    s.holder         = UNIT;
    s.holder_is_head = head;
}

/**
 * @brief Bank this thread's unit is on. Pass it to @ref _llk_math_dest_clear_ after the FPU's take.
 */
inline std::uint32_t _llk_dest_chain_bank_()
{
    return dest_chain::state.bank;
}

} // namespace ckernel
