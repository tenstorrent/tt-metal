// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt_stl/small_vector.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::quasar {

// Quasar-native permute. Same contract as ttnn::permute for interleaved tensors: output axis k takes input
// axis dims[k]; layout, data type and (unless overridden) memory config follow the input. `pad_value` fills the
// tile padding of a TILE output whose tiles are rebuilt element by element.
Tensor permute(
    const Tensor& input_tensor,
    const ttsl::SmallVector<int64_t>& dims,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    float pad_value = 0.0f);

}  // namespace ttnn::operations::experimental::quasar
