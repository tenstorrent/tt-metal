// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include "ttnn/cpp/ttnn/kernel_lib/mcast/mcast_common.hpp"
#include "api/dataflow/noc_semaphore.h"

namespace dataflow_kernel_lib::detail {

// Descriptor callers retain raw IDs. ProgramSpec callers pass generated binding tokens,
// whose scope must reach Semaphore unchanged. nullptr represents an unused resource role.
constexpr uint32_t mcast_semaphore_id(uint32_t id) { return id; }
constexpr uint32_t mcast_semaphore_id(std::nullptr_t) { return UNUSED_SEM_ID; }
template <uint32_t ID, SemScope SCOPE>
constexpr uint32_t mcast_semaphore_id(SemaphoreBindingToken<ID, SCOPE>) {
    return ID;
}

template <auto BINDING>
FORCE_INLINE auto make_mcast_semaphore() {
    if constexpr (mcast_semaphore_id(BINDING) == UNUSED_SEM_ID) {
        return nullptr;
    } else {
        return Semaphore(BINDING);
    }
}

}  // namespace dataflow_kernel_lib::detail
