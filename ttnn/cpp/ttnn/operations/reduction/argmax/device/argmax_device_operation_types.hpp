// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

#include <optional>
#include <tuple>

namespace ttnn::prim {

struct ArgmaxParams {
    tt::tt_metal::DataType output_dtype{};
    std::optional<int> dim;
    bool keepdim{};
    std::optional<CoreRangeSet> sub_core_grids;
    bool use_multicore{};
    tt::tt_metal::MemoryConfig output_mem_config;

    static constexpr auto attribute_names =
        std::forward_as_tuple("output_dtype", "dim", "keepdim", "sub_core_grids", "use_multicore", "output_mem_config");
    auto attribute_values() const {
        return std::forward_as_tuple(output_dtype, dim, keepdim, sub_core_grids, use_multicore, output_mem_config);
    }
};

struct ArgmaxInputs {
    Tensor input;
    std::optional<Tensor> optional_output_tensor;

    static constexpr auto attribute_names = std::forward_as_tuple("input", "optional_output_tensor");
    auto attribute_values() const { return std::forward_as_tuple(input, optional_output_tensor); }
};

}  // namespace ttnn::prim
