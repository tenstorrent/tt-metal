// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Vendored, bring-up only: the kernel helper library's multicast helpers as of 73027b6e6ff1 (before the
// llk_helper_library port to the unified Mcast API), renamed into their own namespaces so they coexist with
// ttnn/cpp/ttnn/kernel_lib/mcast. Used by rms_norm_ttnn and mhc_pre_ttnn until they move to the new API.

#pragma once

#include <cstddef>
#include "../mcast_common.hpp"
#include "api/dataflow/noc_semaphore.h"

namespace dataflow_kernel_lib_mcast_v1::detail {

// Descriptor callers retain raw IDs. ProgramSpec callers pass generated binding tokens,
// whose scope must reach Semaphore unchanged. nullptr represents an unused resource role.
constexpr uint32_t mcast_semaphore_id(uint32_t id) { return id; }
constexpr uint32_t mcast_semaphore_id(std::nullptr_t) { return UNUSED_SEM_ID; }
constexpr uint32_t mcast_semaphore_id(SemaphoreBindingToken token) { return token.id; }

template <auto BINDING>
FORCE_INLINE auto make_mcast_semaphore() {
    if constexpr (mcast_semaphore_id(BINDING) == UNUSED_SEM_ID) {
        return nullptr;
    } else {
        return Semaphore(BINDING);
    }
}

}  // namespace dataflow_kernel_lib_mcast_v1::detail
