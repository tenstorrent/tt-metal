// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Deprecated: nlp_create_qkv_heads, nlp_create_q_heads_split was renamed to create_qkv_heads, create_q_heads_split.
// This header keeps the old include path and the old function names building for one release; include
// "ttnn/operations/experimental/transformer/create_qkv_heads/create_qkv_heads.hpp" and call
// ttnn::experimental::create_qkv_heads instead.

#pragma once

#include <utility>

#include "ttnn/operations/experimental/transformer/create_qkv_heads/create_qkv_heads.hpp"

namespace ttnn::experimental {

template <typename... Args>
auto nlp_create_qkv_heads(Args&&... args) {
    return create_qkv_heads(std::forward<Args>(args)...);
}
template <typename... Args>
auto nlp_create_q_heads_split(Args&&... args) {
    return create_q_heads_split(std::forward<Args>(args)...);
}
}  // namespace ttnn::experimental
