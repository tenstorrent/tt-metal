// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/embedding_backward/embedding_backward.hpp"

#include <utility>

#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/embedding_backward/device/embedding_backward_device_operation.hpp"
#include "ttnn/operation.hpp"

namespace ttnn {

Tensor embedding_bw(
    const Tensor& input_tensor_arg,
    const Tensor& weight_tensor_arg,
    const Tensor& output_gradient_tensor_arg,
    const std::optional<const DataType> dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    auto num_embeddings = weight_tensor_arg.logical_shape()[-2];

    const auto& input_shape = input_tensor_arg.logical_shape();
    auto batch_size = input_shape[0];
    auto sentence_size = input_shape[-1];
    auto input_tensor = ttnn::reshape(input_tensor_arg, ttnn::Shape({batch_size, 1, 1, sentence_size}));

    const auto output_mem_config = memory_config.value_or(output_gradient_tensor_arg.memory_config());
    const auto out_dtype = dtype.value_or(output_gradient_tensor_arg.dtype());

    auto weight_grad = ttnn::prim::embedding_backward(
        input_tensor, output_gradient_tensor_arg, output_mem_config, out_dtype, num_embeddings, optional_output_tensor);
    const auto& weight_shape = weight_tensor_arg.logical_shape();
    if (optional_output_tensor.has_value() || weight_shape.rank() == weight_grad.logical_shape().rank()) {
        return weight_grad;
    }
    TT_FATAL(
        weight_shape.volume() == weight_grad.logical_volume(),
        "embedding_bw: weight shape {} does not match the gradient shape {}",
        weight_shape,
        weight_grad.logical_shape());
    return ttnn::reshape(weight_grad, weight_shape);
}

}  // namespace ttnn
