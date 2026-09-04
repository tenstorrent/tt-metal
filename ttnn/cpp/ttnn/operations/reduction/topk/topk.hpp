// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>
#include "ttnn/types.hpp"

namespace ttnn::operations::reduction::topk {
struct ExecuteTopK {
    static std::vector<Tensor> invoke(
        const Tensor& input_tensor,
        uint32_t k,
        int8_t dim,
        bool largest,
        bool sorted,
        const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids = std::nullopt,
        const std::optional<Tensor>& indices_tensor = std::nullopt,
        std::optional<std::tuple<Tensor&, Tensor&>> preallocated_output_tensors = std::nullopt,
        bool stable = false);
};

namespace detail {

// Reports whether the canonical sampling call shape (last-dimension,
// largest/sorted, unstable, with no custom tensors, grids, or memory config)
// will use the Blackhole large-indices route. This deliberately shares the
// implementation used by topk so Python sampling does not mirror policy.
bool sampling_topk_would_route_to_large_indices(const Tensor& input_tensor, uint32_t k);

}  // namespace detail
}  // namespace ttnn::operations::reduction::topk

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
    std::optional<std::tuple<Tensor&, Tensor&>> preallocated_output_tensors = std::nullopt,
    bool stable = false);

}  // namespace ttnn
