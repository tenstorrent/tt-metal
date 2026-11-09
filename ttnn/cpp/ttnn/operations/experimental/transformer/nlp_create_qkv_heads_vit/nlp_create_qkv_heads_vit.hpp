// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Deprecated: nlp_create_qkv_heads_vit was renamed to create_qkv_heads_vit. This header keeps the old include path and
// the old function name building for one release; include
// "ttnn/operations/experimental/transformer/create_qkv_heads_vit/create_qkv_heads_vit.hpp" and call
// ttnn::experimental::create_qkv_heads_vit instead.

#pragma once

#include <utility>

#include "ttnn/operations/experimental/transformer/create_qkv_heads_vit/create_qkv_heads_vit.hpp"

namespace ttnn::experimental {

template <typename... Args>
auto nlp_create_qkv_heads_vit(Args&&... args) {
    return create_qkv_heads_vit(std::forward<Args>(args)...);
}
}  // namespace ttnn::experimental
