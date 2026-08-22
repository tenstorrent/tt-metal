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

    // Right-align: input dims map to the trailing output dims, matching
    // torch.expand semantics and repeat's own match_input_rank (copy_backward).
    const auto offset = new_rank - source_rank;
    for (size_t index = 0; index < new_rank; ++index) {
        if (index < offset) {
            TT_FATAL(
                shape[index] >= 0,
                "Leading dimension must be non-negative (got {}); it has no corresponding input dimension",
                shape[index]);
            expansion_vector[index] = shape[index];
            output_shape[index] = shape[index];
        } else {
            const auto src_dim = tensor_shape[static_cast<int32_t>(index - offset)];
            if ((shape[index] == -1) || (shape[index] == static_cast<int32_t>(src_dim))) {
                expansion_vector[index] = 1;
                output_shape[index] = src_dim;
            } else {
                TT_FATAL(shape[index] >= 0, "Expand dimension size must be -1 or non-negative (got {})", shape[index]);
                TT_FATAL(src_dim == 1, "Only size 1 dimensions can be expanded in the output shape");
                expansion_vector[index] = shape[index];
                output_shape[index] = shape[index];
            }
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
