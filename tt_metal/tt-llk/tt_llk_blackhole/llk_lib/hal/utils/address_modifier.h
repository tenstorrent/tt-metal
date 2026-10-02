// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "../cfg.h"

namespace hal::pack
{

/** @brief Per-transfer update of one packer address counter: step amount and optional reset. */
struct CounterUpdate
{
    std::uint32_t step = 0;
    bool clear         = false;
};

/**
 * @brief Per-transfer stepping of the packer address counters for one address-modifier selection.
 *
 * The source side is the destination-register read and the destination side is the L1 write;
 * row counters step in rows, face counters in face pairs. A cleared counter resets after the
 * transfer that selects the modifier.
 */
struct AddressModifier
{
    cfg::Sec selection = cfg::Sec::S0;
    CounterUpdate source_row {};
    CounterUpdate destination_row {};
    CounterUpdate source_face {};
    CounterUpdate destination_face {};

    constexpr bool is_valid() const
    {
        return static_cast<std::uint32_t>(selection) < 4u && source_row.step < 16u && destination_row.step < 16u && source_face.step < 2u &&
               destination_face.step < 2u;
    }
};

namespace detail
{

template <AddressModifier... Modifiers>
constexpr bool address_modifier_selections_are_unique()
{
    constexpr std::array<cfg::Sec, sizeof...(Modifiers)> selections {Modifiers.selection...};
    for (std::size_t i = 0; i < selections.size(); ++i)
    {
        for (std::size_t j = i + 1u; j < selections.size(); ++j)
        {
            if (selections[i] == selections[j])
            {
                return false;
            }
        }
    }
    return true;
}

} // namespace detail

/**
 * @brief Program distinct packer address-modifier selections through one grouped CFG write.
 *
 * Each modifier becomes one complete 16-bit SETC16 write. Passing the same selection more than
 * once is rejected at compile time instead of emitting multiple writes to the same configuration
 * word.
 *
 * @tparam Modifiers: Constant address modifiers to program; each selection may appear only once.
 */
template <AddressModifier... Modifiers>
inline __attribute__((always_inline)) void configure_address_modifiers()
{
    constexpr bool all_valid = (Modifiers.is_valid() && ...);
    constexpr bool unique    = detail::address_modifier_selections_are_unique<Modifiers...>();

    static_assert(sizeof...(Modifiers) > 0u, "at least one packer address modifier is required");
    static_assert(all_valid, "packer address modifier selection or counter step is out of range");
    static_assert(unique, "a packer address-modifier selection can be configured only once per grouped write");

    if constexpr (sizeof...(Modifiers) > 0u && all_valid && unique)
    {
        cfg::write<cfg::Access::TensixCfgUnit>(
            cfg::set<cfg::AddrMod[cfg::Src][cfg::Y].Incr, Modifiers.selection, Modifiers.source_row.step>()...,
            cfg::set<cfg::AddrMod[cfg::Src][cfg::Y].Clear, Modifiers.selection, Modifiers.source_row.clear ? 1u : 0u>()...,
            cfg::set<cfg::AddrMod[cfg::Dest][cfg::Y].Incr, Modifiers.selection, Modifiers.destination_row.step>()...,
            cfg::set<cfg::AddrMod[cfg::Dest][cfg::Y].Clear, Modifiers.selection, Modifiers.destination_row.clear ? 1u : 0u>()...,
            cfg::set<cfg::AddrMod[cfg::Src][cfg::Z].Incr, Modifiers.selection, Modifiers.source_face.step>()...,
            cfg::set<cfg::AddrMod[cfg::Src][cfg::Z].Clear, Modifiers.selection, Modifiers.source_face.clear ? 1u : 0u>()...,
            cfg::set<cfg::AddrMod[cfg::Dest][cfg::Z].Incr, Modifiers.selection, Modifiers.destination_face.step>()...,
            cfg::set<cfg::AddrMod[cfg::Dest][cfg::Z].Clear, Modifiers.selection, Modifiers.destination_face.clear ? 1u : 0u>()...);
    }
}

} // namespace hal::pack
