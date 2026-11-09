// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Deprecated: nlp_concat_heads was renamed to concat_heads. This header keeps the old include path and
// the old function name building for one release; include
// "ttnn/operations/experimental/transformer/concat_heads/concat_heads.hpp" and call ttnn::experimental::concat_heads
// instead.

#pragma once

#include <utility>

#include "ttnn/operations/experimental/transformer/concat_heads/concat_heads.hpp"

namespace ttnn::experimental {

template <typename... Args>
auto nlp_concat_heads(Args&&... args) {
    return concat_heads(std::forward<Args>(args)...);
}
}  // namespace ttnn::experimental
