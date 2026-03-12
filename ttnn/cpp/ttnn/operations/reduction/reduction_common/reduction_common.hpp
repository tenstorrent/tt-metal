// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/creation.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <variant>
#include <vector>

namespace reduction_common {

enum class ReduceType {
    Sum,
    Mean,
    Max,
    Min,
    Std,
    Var,
    Prod,
};

template <class Tuple, class T = std::decay_t<std::tuple_element_t<0, std::decay_t<Tuple>>>>
std::vector<std::optional<T>> tuple_to_vector_optional(Tuple&& tuple) {
    return std::apply(
        [](auto&&... elems) { return std::vector<std::optional<T>>{std::forward<decltype(elems)>(elems)...}; },
        std::forward<Tuple>(tuple));
}

ttnn::Tensor perform_transpose(
    const ttnn::Tensor& input_tensor, bool is_dim_last_idx, int8_t dim1 = -1, int8_t dim2 = -1);

ttnn::Tensor transform_to_4d_tensor(const ttnn::Tensor& input_tensor, bool is_rank_le_4d);

ttnn::SmallVector<int> generate_reduce_dim(
    const ttnn::Tensor& input_tensor_arg,
    const std::optional<std::variant<int, int64_t, ttnn::SmallVector<int>>>& dim_arg);

constexpr float get_zero_volume_fill_value(const ReduceType type) {
    switch (type) {
        case ReduceType::Sum: return 0.0f;
        case ReduceType::Mean:
        case ReduceType::Max:
        case ReduceType::Min:
        case ReduceType::Std:
        case ReduceType::Var: return std::numeric_limits<float>::quiet_NaN();
        case ReduceType::Prod: return 1.0f;
        default:
            // Don't just return NaN, since it may not be appropriate for all reduction types.
            TT_THROW("Unhandled reduction type");
    }
}

/* Creates appropriate output tensor for a given zero volume input tensor.
   The output tensor's shape is adjusted for keepdim:
   - if keepdim is true, the dimensions specified in dim are set to 1.
   - if keepdim is false, the dimensions specified in dim are removed.
   The output tensor is filled with NaN/0/1 based on the reduce_type.
*/
template <ReduceType reduce_type>
ttnn::Tensor zero_volume_reduce(
    const ttnn::Tensor& input_tensor,
    const ttnn::SmallVector<int>& dim,
    const bool keepdim,
    const ttnn::MemoryConfig& memory_config) {
    auto input_shape = input_tensor.logical_shape();

    // min/max is unsupported when reduction dim is zero
    if constexpr (reduce_type == ReduceType::Max || reduce_type == ReduceType::Min) {
        TT_FATAL(input_shape.rank() != 0, "Input tensor cannot be a scalar (rank 0)");

        // Check the shape of the reduction dims
        for (auto red_dim : dim) {
            if (input_shape[red_dim] == 0) {
                TT_THROW("Expected reduction dim {} to have non-zero size", red_dim);
            }
        }
    }

    ttnn::SmallVector<uint32_t> output_shape;

    const int rank = static_cast<int>(input_shape.rank());
    // Iterate over the input shape and adjust the output shape for keepdim
    for (int i = 0; i < rank; ++i) {
        // If this is in the reduction dims, keep it only if keepdim is true
        bool is_reduction_dim = std::find(dim.begin(), dim.end(), i) != dim.end();

        if (is_reduction_dim && keepdim) {
            output_shape.push_back(1);
        } else if (!is_reduction_dim) {
            output_shape.push_back(input_shape[i]);
        }
    }

    constexpr float fill_value = get_zero_volume_fill_value(reduce_type);

    return ttnn::full(
        ttnn::Shape(output_shape),
        fill_value,
        input_tensor.dtype(),
        input_tensor.layout(),
        *input_tensor.device(),
        memory_config);
}

}  // namespace reduction_common
