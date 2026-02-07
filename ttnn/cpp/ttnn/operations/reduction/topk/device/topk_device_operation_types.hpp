// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

#include <cstdint>
#include <optional>
#include <tuple>

namespace ttnn::prim {
struct TopkParams {
    uint32_t k{};
    int8_t dim{};
    bool largest{};
    bool sorted{};
    tt::tt_metal::MemoryConfig output_memory_config;
    tt::tt_metal::CoreRangeSet sub_core_grids;

    static constexpr auto attribute_names =
        std::forward_as_tuple("k", "dim", "largest", "sorted", "output_memory_config", "sub_core_grids");
    auto attribute_values() const {
        return std::forward_as_tuple(k, dim, largest, sorted, output_memory_config, sub_core_grids);
    }
};

struct TopkInputs {
    Tensor input;
    std::optional<Tensor> indices;
    std::optional<std::tuple<Tensor, Tensor>> preallocated_outputs;

    static constexpr auto attribute_names = std::forward_as_tuple("input", "indices", "preallocated_outputs");
    auto attribute_values() const { return std::forward_as_tuple(input, indices, preallocated_outputs); }
};
}  // namespace ttnn::prim
