// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Deprecated: nlp_kv_cache_load_slice was renamed to kv_cache_load_slice. This header keeps the old include path and
// the old function name building for one release; include
// "ttnn/operations/experimental/transformer/kv_cache_load_slice/kv_cache_load_slice.hpp" and call
// ttnn::experimental::kv_cache_load_slice instead.

#pragma once

#include <utility>

#include "ttnn/operations/experimental/transformer/kv_cache_load_slice/kv_cache_load_slice.hpp"

namespace ttnn::experimental {

template <typename... Args>
auto nlp_kv_cache_load_slice(Args&&... args) {
    return kv_cache_load_slice(std::forward<Args>(args)...);
}
}  // namespace ttnn::experimental
