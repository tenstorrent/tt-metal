// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operation.hpp"
#include "expand.hpp"
#include <functional>
#include <ttnn/operations/functions.hpp>
#include "ttnn/operations/data_movement/repeat/repeat.hpp"
#include "ttnn/operations/data_movement/view/view.hpp"

namespace {

struct ExpandPlan {
    ttsl::SmallVector<uint32_t> repetitions;
    ttsl::SmallVector<uint32_t> output_shape;
};

ExpandPlan build_expand_plan(const ttnn::Tensor& tensor, std::span<const int32_t> shape) {
    const auto& tensor_shape = tensor.logical_shape();
    const auto source_rank = tensor_shape.rank();
    const auto new_rank = shape.size();
    TT_FATAL(source_rank <= new_rank, "Output rank must be >= input rank for expand");

    ttsl::SmallVector<uint32_t> expansion_vector(new_rank);
    ttsl::SmallVector<uint32_t> output_shape(new_rank);

    // Right-align: walk output dims trailing-to-leading, pairing with input dims
    // from the end — mirrors repeat's match_input_rank (copy_backward).
    int32_t src_idx = static_cast<int32_t>(source_rank) - 1;
    for (int32_t out_idx = static_cast<int32_t>(new_rank) - 1; out_idx >= 0; --out_idx) {
        if (src_idx >= 0) {
            const auto src_dim = tensor_shape[src_idx];
            if ((shape[out_idx] == -1) || (shape[out_idx] == static_cast<int32_t>(src_dim))) {
                expansion_vector[out_idx] = 1;
                output_shape[out_idx] = src_dim;
            } else {
                TT_FATAL(
                    shape[out_idx] >= 0, "Expand dimension size must be -1 or non-negative (got {})", shape[out_idx]);
                TT_FATAL(src_dim == 1, "Only size 1 dimensions can be expanded in the output shape");
                expansion_vector[out_idx] = shape[out_idx];
                output_shape[out_idx] = shape[out_idx];
            }
            --src_idx;
        } else {
            TT_FATAL(
                shape[out_idx] >= 0,
                "Leading dimension must be non-negative (got {}); it has no corresponding input dimension",
                shape[out_idx]);
            expansion_vector[out_idx] = shape[out_idx];
            output_shape[out_idx] = shape[out_idx];
        }
    }
    return {std::move(expansion_vector), std::move(output_shape)};
}

}  // namespace

namespace ttnn {

Tensor expand(
    const ttnn::Tensor& tensor,
    const ttsl::Span<const int32_t> shape_vector,
    const std::optional<MemoryConfig>& memory_config) {
    auto [repetitions, expected_shape] = build_expand_plan(tensor, shape_vector);
    auto result = ttnn::repeat(tensor, repetitions, memory_config);

    // repeat's all-ones early return discards the rank padding from
    // match_input_rank, so reshape to the expected output rank if needed.
    if (result.logical_shape().rank() != expected_shape.size()) {
        result = ttnn::view(result, ttnn::Shape(expected_shape));
    }
    return result;
}

}  // namespace ttnn
