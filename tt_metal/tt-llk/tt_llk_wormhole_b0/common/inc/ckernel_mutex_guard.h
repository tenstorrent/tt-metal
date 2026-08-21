// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"

namespace ckernel
{

/**
 * @brief Scope guard holding a Tensix thread mutex for the lifetime of the object.
 *
 * @tparam enable: When false the guard compiles away and the enclosed instructions issue unguarded.
 * @note A wait placed ahead of an acquire must also block p_stall::STALL_SYNC: otherwise the ATGETM
 *       slips past the unmet wait and the thread stalls with the mutex already held.
 */
template <bool enable = true>
class [[nodiscard]] T6MutexLockGuard final
{
public:
    explicit T6MutexLockGuard(const std::uint8_t index) noexcept : mutex_index(index)
    {
        if constexpr (enable)
        {
            t6_mutex_acquire(mutex_index);
        }
    }

    ~T6MutexLockGuard()
    {
        if constexpr (enable)
        {
            t6_mutex_release(mutex_index);
        }
    }

    // Non-copyable
    T6MutexLockGuard(const T6MutexLockGuard&)            = delete;
    T6MutexLockGuard& operator=(const T6MutexLockGuard&) = delete;

    // Non-movable
    T6MutexLockGuard(T6MutexLockGuard&&)            = delete;
    T6MutexLockGuard& operator=(T6MutexLockGuard&&) = delete;

private:
    const std::uint8_t mutex_index;
};

} // namespace ckernel
