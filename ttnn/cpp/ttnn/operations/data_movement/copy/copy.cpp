// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/copy/copy.hpp"

#include <utility>

#include "device/copy_device_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/run_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

ttnn::Tensor CopyOperation::invoke(const Tensor& src_tensor, const Tensor& dst_tensor) {
    operation::run(
        CopyDeviceOperation{dst_tensor.memory_config(), dst_tensor.dtype()}, {src_tensor, dst_tensor}, {}, {});
    return dst_tensor;
}

ttnn::Tensor AssignOperation::invoke(
    const Tensor& input,
    const MemoryConfig& output_mem_config,
    std::optional<const DataType> output_dtype,
    std::optional<Tensor> optional_output_tensor) {
    return operation::run(
               CopyDeviceOperation{output_mem_config, output_dtype.value_or(input.dtype())},
               {input},
               {},
               {std::move(optional_output_tensor)})
        .at(0);
}

ttnn::Tensor AssignOperation::invoke(
    const Tensor& input, const MemoryConfig& output_mem_config, std::optional<const DataType> output_dtype) {
    return invoke(input, output_mem_config, output_dtype, std::nullopt);
}

ttnn::Tensor AssignOperation::invoke(const Tensor& input_a, const Tensor& input_b) {
    operation::run(CopyDeviceOperation{input_b.memory_config(), input_b.dtype()}, {input_a, input_b}, {}, {});
    return input_b;
}

}  // namespace ttnn::operations::data_movement
