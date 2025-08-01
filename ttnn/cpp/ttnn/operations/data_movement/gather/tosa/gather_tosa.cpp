// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gather_tosa.hpp"
#include <cstdint>

#include "../gather.hpp"

#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include "ttnn/operations/data_movement/expand/expand.hpp"

namespace ttnn::operations::data_movement {
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
Tensor pre_tosa_gather_transform_input_index_tensor(const Tensor& input_tensor, const int8_t dim, const uint32_t C) {
    if (input_tensor.logical_shape().rank() == 1) {
        // Early exit for scalar tensors, return the same tensor
        return input_tensor;
    }

    // Unsqueeze the input tensor to add a new dimension
    const Tensor unsqueezed_tensor = ttnn::unsqueeze(input_tensor, -1);
    // Create a shape vector for the new tensor
    ttnn::SmallVector<int32_t> shape_vector = {input_tensor.logical_shape()[0], input_tensor.logical_shape()[1], C};
    Tensor expanded_tensor = ttnn::expand(unsqueezed_tensor, shape_vector, unsqueezed_tensor.memory_config());

    return expanded_tensor;
}
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

Tensor ExecuteTosaGather::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const Tensor& input_index_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config) {
    // TOSA Gather constraints
    constexpr int8_t dim = 1;
    constexpr bool sparse_grad = false;
    constexpr size_t input_tensor_rank_constraint = 3;
    constexpr size_t input_index_tensor_rank_constraint = 2;
    const std::optional<Tensor> optional_output_tensor_value = std::nullopt;
    const auto memory_config_value = memory_config.has_value() ? memory_config.value() : input_tensor.memory_config();

    // Input tensor
    const ttnn::Shape& original_input_tensor_lshape = input_tensor.logical_shape();  // [N, K, C]
    const auto input_tensor_rank = input_tensor.padded_shape().rank();
    TT_FATAL(
        input_tensor_rank == input_tensor_rank_constraint,
        "Input tensor rank must be {}, got: {}",
        input_tensor_rank_constraint,
        input_tensor_rank);
    const auto N = original_input_tensor_lshape[0];
    const auto C = original_input_tensor_lshape[-1];

    // Index tensor
    const auto& original_input_index_tensor_lshape = input_index_tensor.logical_shape();  // [N, W]
    const auto input_index_tensor_rank = input_index_tensor.padded_shape().rank();
    TT_FATAL(
        input_index_tensor_rank == input_index_tensor_rank_constraint,
        "Index tensor rank must be {}, got: {}",
        input_index_tensor_rank_constraint,
        input_index_tensor_rank);
    TT_FATAL(
        N == original_input_index_tensor_lshape[0],
        "Index tensor first dimension must be equal to input tensor first dimension");

    Tensor expanded_index_tensor =
        CMAKE_UNIQUE_NAMESPACE::pre_tosa_gather_transform_input_index_tensor(input_index_tensor, dim, C);

    return ttnn::gather(
        queue_id,
        input_tensor,
        dim,
        expanded_index_tensor,
        sparse_grad,
        memory_config_value,
        optional_output_tensor_value);
}

}  // namespace ttnn::operations::data_movement
