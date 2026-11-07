// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file
 * @brief Public entry point for the Blackhole counter HAL.
 *
 * @see hal::counters::adc::AddrCounter
 * @see hal::counters::rwc::MathCounter
 */

#pragma once

#include <type_traits>

#include "counters/adc.h"
#include "counters/rwc.h"

namespace hal::counters
{

/**
 * @brief Issue a compile-time address-counter or math-counter update.
 *
 * @tparam Descriptor: An adc::AddrCounter or rwc::MathCounter; forwards to adc::set or rwc::set.
 */
template <auto Descriptor>
inline __attribute__((always_inline)) void set()
{
    using Type = std::remove_cv_t<decltype(Descriptor)>;
    static_assert(std::is_same_v<Type, adc::AddrCounter> || std::is_same_v<Type, rwc::MathCounter>, "expected an AddrCounter or MathCounter descriptor");
    if constexpr (std::is_same_v<Type, adc::AddrCounter>)
    {
        adc::set<Descriptor>();
    }
    else
    {
        rwc::set<Descriptor>();
    }
}

/** @brief Issue a runtime address-counter update; see adc::set(). */
inline __attribute__((always_inline)) void set(const adc::AddrCounter descriptor)
{
    adc::set(descriptor);
}

/** @brief Issue a runtime math-counter update; see rwc::set(). */
inline __attribute__((always_inline)) void set(const rwc::MathCounter descriptor)
{
    rwc::set(descriptor);
}

} // namespace hal::counters
