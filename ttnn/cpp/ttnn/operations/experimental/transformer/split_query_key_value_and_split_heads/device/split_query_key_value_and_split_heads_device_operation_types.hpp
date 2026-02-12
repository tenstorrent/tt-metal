// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include <tuple>

namespace ttnn::experimental::prim {

struct SplitQueryKeyValueAndSplitHeadsParams {
    CoreCoord compute_with_storage_grid_size;
    tt::tt_metal::MemoryConfig output_mem_config;
    uint32_t num_heads{};

    static constexpr auto attribute_names =
        std::forward_as_tuple("compute_with_storage_grid_size", "output_mem_config", "num_heads");
    auto attribute_values() const {
        return std::forward_as_tuple(compute_with_storage_grid_size, output_mem_config, num_heads);
    }
};

struct SplitQueryKeyValueAndSplitHeadsInputs {
    Tensor input_tensor;
    std::vector<std::optional<Tensor>> output_tensors;

    static constexpr auto attribute_names = std::forward_as_tuple("input_tensor", "output_tensors");
    auto attribute_values() const { return std::forward_as_tuple(input_tensor, output_tensors); }
};

}  // namespace ttnn::experimental::prim
