// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "ttnn/common/constants.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"
#include "device/copy_device_operation.hpp"

namespace ttnn::operations::data_movement {

ttnn::Tensor CopyOperation::invoke(
    uint8_t queue_id,
    const Tensor& src_tensor,
    const Tensor& dst_tensor) {
    operation::run(CopyDeviceOperation{dst_tensor.memory_config(), dst_tensor.get_dtype()}, {src_tensor, dst_tensor}, {}, {}, queue_id);
    return dst_tensor;
}

ttnn::Tensor CopyOperation::invoke(
    const Tensor& src_tensor,
    const Tensor& dst_tensor) {
    return invoke(ttnn::DefaultQueueId, src_tensor, dst_tensor);
}

ttnn::Tensor CloneOperation::invoke(
    uint8_t queue_id,
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<const DataType> output_dtype) {
    return operation::run(CopyDeviceOperation{output_mem_config.value_or(input_tensor.memory_config()), output_dtype.value_or(input_tensor.get_dtype())}, {input_tensor}, {}, {}, queue_id).at(0);
}

ttnn::Tensor CloneOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<const DataType> output_dtype) {
    return invoke(ttnn::DefaultQueueId, input_tensor, output_mem_config, output_dtype);
}

ttnn::Tensor AssignOperation::invoke(
    uint8_t queue_id,
    const Tensor& input,
    const MemoryConfig& output_mem_config,
    std::optional<const DataType> output_dtype,
    std::optional<Tensor> optional_output_tensor) {
    return operation::run(CopyDeviceOperation{output_mem_config, output_dtype.value_or(input.get_dtype())}, {input}, {}, {optional_output_tensor}, queue_id).at(0);
}

ttnn::Tensor AssignOperation::invoke(
    const Tensor& input,
    const MemoryConfig& output_mem_config,
    std::optional<const DataType> output_dtype) {
    return invoke(ttnn::DefaultQueueId, input, output_mem_config, output_dtype);
}

ttnn::Tensor AssignOperation::invoke(
    uint8_t queue_id,
    const Tensor& input_a,
    const Tensor& input_b) {
    operation::run(CopyDeviceOperation{input_b.memory_config(), input_b.get_dtype()}, {input_a, input_b}, {}, {}, queue_id);
    return input_b;
}

ttnn::Tensor AssignOperation::invoke(
    const Tensor& input_a,
    const Tensor& input_b) {
    return invoke(ttnn::DefaultQueueId, input_a, input_b);
}

} // ttnn::operations::data_movement namespace
