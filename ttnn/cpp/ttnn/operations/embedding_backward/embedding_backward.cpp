// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/embedding_backward/embedding_backward.hpp"

#include <utility>

#include "ttnn/common/constants.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
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

    // The op adds each index's gradient rows into its partial output one chunk at a time. Kept in bfloat16 or
    // bfloat8_b, that running sum stops absorbing small terms once it grows, so a row that many indices hit loses
    // most of its gradient. Accumulate in FLOAT32 (the kernel reads the gradient in its own dtype) and round to the
    // requested dtype once at the end.
    const bool fp32_output = out_dtype == DataType::FLOAT32 || (optional_output_tensor.has_value() &&
                                                                optional_output_tensor->dtype() == DataType::FLOAT32);
    auto accumulated = ttnn::prim::embedding_backward(
        input_tensor,
        output_gradient_tensor_arg,
        output_mem_config,
        DataType::FLOAT32,
        num_embeddings,
        fp32_output ? optional_output_tensor : std::nullopt);
    if (fp32_output) {
        return accumulated;
    }
    auto output = ttnn::typecast(accumulated, out_dtype, output_mem_config, optional_output_tensor);
    accumulated.deallocate();
    return output;
}

}  // namespace ttnn
