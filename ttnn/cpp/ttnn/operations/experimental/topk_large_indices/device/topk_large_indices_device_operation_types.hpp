// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>
#include <optional>
#include <tuple>

#include <tt_stl/assert.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::topk_large_indices {

inline uint32_t flattened_rows_excluding_last_dim(const ttnn::Shape& shape) {
    uint64_t rows = 1;
    for (uint32_t i = 0; i + 1 < shape.rank(); ++i) {
        const auto dim = shape[i];
        TT_FATAL(
            dim == 0 || rows <= std::numeric_limits<uint32_t>::max() / dim,
            "topk_large_indices flattened leading dimensions must fit in uint32_t rows; got shape {}",
            shape);
        rows *= dim;
    }
    return static_cast<uint32_t>(rows);
}

struct operation_attributes_t {
    uint32_t k{};
    // Restrict the search to the first `valid_length` columns of each row instead of the full last
    // dimension. Lets top-k run over the real prefix of an over-allocated row (whose tail may be stale)
    // without physically slicing the input. nullopt = search the full width. Runtime-only (hash-excluded,
    // validated on cache hit) so a serving loop growing valid_length reuses one program.
    std::optional<uint32_t> valid_length{};

    static constexpr auto attribute_names = std::forward_as_tuple("k");
    auto attribute_values() const { return std::forward_as_tuple(k); }
};

struct tensor_args_t {
    const Tensor& input_tensor;

    tensor_args_t() = delete;
    explicit tensor_args_t(const Tensor& input_tensor_in) : input_tensor(input_tensor_in) {}

    static constexpr auto attribute_names = std::forward_as_tuple(
        "input_dtype",
        "input_layout",
        "input_memory_layout",
        "input_buffer_type",
        "compute_grid_x",
        "compute_grid_y",
        "input_tensor");
    auto attribute_values() const {
        const auto& input = input_tensor;
        const auto grid = input.device()->compute_with_storage_grid_size();
        const auto& mem_config = input.memory_config();
        return std::make_tuple(
            input.dtype(),
            input.layout(),
            mem_config.memory_layout(),
            mem_config.buffer_type(),
            grid.x,
            grid.y,
            std::cref(input));
    }
};

using tensor_return_value_t = Tensor;
using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::experimental::topk_large_indices
