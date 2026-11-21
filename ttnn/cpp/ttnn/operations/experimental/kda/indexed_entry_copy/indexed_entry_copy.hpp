// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::kda {

ttnn::Tensor indexed_entry_copy(
    const Tensor& source,
    const Tensor& destination,
    const Tensor& index,
    uint32_t index_scale,
    uint32_t index_offset,
    const std::optional<Tensor>& zero_fill_at_zero = std::nullopt,
    const std::optional<Tensor>& source_part = std::nullopt,
    uint32_t source_part_groups = 1);

}  // namespace ttnn::experimental::kda
