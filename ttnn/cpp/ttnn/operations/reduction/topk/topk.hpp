// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

namespace ttnn {

std::vector<Tensor> topk(
    const Tensor& input_tensor,
    uint32_t k,
    int8_t dim,
    bool largest,
    bool sorted,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids = std::nullopt,
    const std::optional<Tensor>& indices_tensor = std::nullopt,
    const std::optional<std::tuple<Tensor, Tensor>>& preallocated_output_tensors = std::nullopt);

}  // namespace ttnn
