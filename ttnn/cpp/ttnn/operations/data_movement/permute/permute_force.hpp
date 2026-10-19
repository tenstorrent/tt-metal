// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement::detail {

// Pinned entry points for checking one implementation against the other. Bound only under
// `ttnn._ttnn.operations.data_movement`; `ttnn::permute` is the public entry.
ttnn::Tensor permute_force_native(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<int64_t>& dims,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

// Throws outside the support scope instead of falling back to native.
ttnn::Tensor permute_force_codegen(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<int64_t>& dims,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::operations::data_movement::detail
