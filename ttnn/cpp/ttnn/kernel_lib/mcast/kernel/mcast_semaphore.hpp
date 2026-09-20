// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <type_traits>
#include "ttnn/cpp/ttnn/kernel_lib/mcast/mcast_common.hpp"
#include "api/dataflow/noc_semaphore.h"

namespace dataflow_kernel_lib {

// A structural compile-time value accepts legacy IDs and native binding tokens
// without losing the host-selected scope. nullptr denotes an unused resource.
struct McastSemaphoreBinding {
    uint32_t id;
    SemScope scope;

    constexpr McastSemaphoreBinding(uint32_t id) : id(id), scope(SemScope::LOCAL_NONATOMIC) {}
    constexpr McastSemaphoreBinding(std::nullptr_t) : McastSemaphoreBinding(UNUSED_SEM_ID) {}
    template <uint32_t ID, SemScope SCOPE>
    constexpr McastSemaphoreBinding(SemaphoreBindingToken<ID, SCOPE>) : id(ID), scope(SCOPE) {}
};

namespace detail {
constexpr uint32_t mcast_semaphore_id(McastSemaphoreBinding binding) { return binding.id; }

// Normalize values to token types for out-of-line pipe definitions: SFPI's
// C++17 NTTP extension does not match class-valued parameters on those methods.
template <McastSemaphoreBinding BINDING>
using McastSemaphoreToken = SemaphoreBindingToken<BINDING.id, BINDING.scope>;

template <McastSemaphoreBinding BINDING>
using McastSemaphore = std::
    conditional_t<BINDING.id == UNUSED_SEM_ID, std::nullptr_t, Semaphore<ProgrammableCoreType::TENSIX, BINDING.scope>>;

template <McastSemaphoreBinding BINDING>
FORCE_INLINE McastSemaphore<BINDING> make_mcast_semaphore() {
    if constexpr (BINDING.id == UNUSED_SEM_ID) {
        return nullptr;
    } else {
        return McastSemaphore<BINDING>(SemaphoreBindingToken<BINDING.id, BINDING.scope>{});
    }
}

}  // namespace detail
}  // namespace dataflow_kernel_lib
