// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "view.hpp"
#include "ttnn/operations/core/set_tensor_spec/set_tensor_spec.hpp"

#include <tt-metalium/constants.hpp>

#include <tracy/Tracy.hpp>

namespace ttnn::operations::experimental::reshape {

static MemoryConfig infer_output_memory_config(
    const MemoryConfig& input_memory_config, const ttnn::Shape& output_padded_shape) {
    if (input_memory_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
        auto shard_spec = input_memory_config.shard_spec().value();
        shard_spec.shape[1] = output_padded_shape[-1];  // update output shard to match new shard width
        return MemoryConfig{input_memory_config.memory_layout(), input_memory_config.buffer_type(), shard_spec};
    } else {
        return input_memory_config;
    }
}

Tensor tensor_reshape(
    const Tensor& input_tensor, const ttnn::Shape& new_logical_shape, const ttnn::Shape& new_padded_shape) {
    tt::tt_metal::GraphTracker::instance().track_function_start(
        "Tensor::reshape", input_tensor, new_logical_shape, new_padded_shape);

    // Compute the new tensor spec based on the input tensor's properties
    const auto output_memory_config = infer_output_memory_config(input_tensor.memory_config(), new_padded_shape);
    auto new_spec = ttnn::TensorSpec(
        new_logical_shape,
        tt::tt_metal::TensorLayout::fromPaddedShape(
            input_tensor.dtype(),
            input_tensor.tensor_spec().page_config(),
            output_memory_config,
            new_logical_shape,
            new_padded_shape));

    // Delegate to set_tensor_spec operation which handles both eager and lazy modes
    // In eager mode: updates device buffer metadata (page size, shard spec) and returns new tensor
    // In lazy mode: creates a lazy operation that defers the update until evaluation
    auto output = ttnn::operations::core::set_tensor_spec(input_tensor, new_spec);

    tt::tt_metal::GraphTracker::instance().track_function_end(output);
    return output;
}

ttnn::Tensor ViewOperation::invoke(
    const ttnn::Tensor& tensor, const ttnn::Shape& logical_shape, const ttnn::Shape& padded_shape) {
    return tensor_reshape(tensor, logical_shape, padded_shape);
}

ttnn::Tensor ViewOperation::invoke(const ttnn::Tensor& tensor, const ttnn::Shape& shape) {
    return tensor_reshape(tensor, shape, shape);
}

}  // namespace ttnn::operations::experimental::reshape
