// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include <tuple>

namespace ttnn::prim {

struct SamplingParams {
    std::optional<uint32_t> seed;
    std::optional<tt::tt_metal::CoreRangeSet> sub_core_grids;

    static constexpr auto attribute_names = std::forward_as_tuple("seed", "sub_core_grids");
    auto attribute_values() const { return std::forward_as_tuple(seed, sub_core_grids); }
};

struct SamplingInputs {
    Tensor input_values;
    Tensor input_indices;
    Tensor k;
    Tensor p;
    Tensor temp;
    std::optional<Tensor> preallocated_output;

    static constexpr auto attribute_names =
        std::forward_as_tuple("input_values", "input_indices", "k", "p", "temp", "preallocated_output");
    auto attribute_values() const {
        return std::forward_as_tuple(input_values, input_indices, k, p, temp, preallocated_output);
    }
};

}  // namespace ttnn::prim
