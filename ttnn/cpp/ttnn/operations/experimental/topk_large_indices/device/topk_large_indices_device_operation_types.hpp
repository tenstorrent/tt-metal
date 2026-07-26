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
    bool return_values{false};
    // Logical index represented by input column zero. Runtime-only so successive
    // bounded paging stripes reuse one compiled program.
    uint32_t index_offset{0};
    // Restrict the search to the first `valid_length` columns of each row instead of the full last
    // dimension. Lets top-k run over the real prefix of an over-allocated row (whose tail may be stale)
    // without physically slicing the input. nullopt = search the full width. Runtime-only (hash-excluded,
    // validated on cache hit) so a serving loop growing valid_length reuses one program.
    std::optional<uint32_t> valid_length{};
};

struct tensor_args_t {
    Tensor input_tensor;
    // Optional IDs paired element-for-element with input values. The value-preserving
    // merge form uses these to carry global logical cache positions through TopK.
    std::optional<Tensor> input_indices;
};

using tensor_return_value_t = std::tuple<Tensor, Tensor>;
using spec_return_value_t = std::tuple<tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec>;

}  // namespace ttnn::operations::experimental::topk_large_indices
