// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/embedding_backward/embedding_backward.hpp"

#include <utility>

#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
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

    TT_FATAL(
        out_dtype == DataType::BFLOAT16 || out_dtype == DataType::BFLOAT8_B,
        "Embedding backward output must be BFLOAT16 or BFLOAT8_B");
    TT_FATAL(
        output_gradient_tensor_arg.dtype() == out_dtype,
        "Output and input gradient tensors must have the same dtype");

    // embedding_backward repeatedly reads and writes partial output tiles. Doing
    // that in the public low-precision dtype loses contributions when an index
    // occurs many times. Keep the public BF16/BFP8 input and output contract,
    // but use a private FP32 accumulation buffer and convert exactly once when
    // the complete gradient is available.
    auto fp32_accumulation = ttnn::prim::embedding_backward(
        input_tensor,
        output_gradient_tensor_arg,
        output_mem_config,
        DataType::FLOAT32,
        num_embeddings,
        std::nullopt);
    return ttnn::typecast(fp32_accumulation, out_dtype, output_mem_config, optional_output_tensor);
}

}  // namespace ttnn
