// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/queue_id.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "typecast.hpp"
#include "cpp/ttnn/operations/data_movement/copy/device/copy_device_operation.hpp"

namespace ttnn::operations::experimental::copy {

ttnn::Tensor TypecastOperation::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const DataType& dtype,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return tt::tt_metal::operation::run(
               ttnn::operations::data_movement::CopyDeviceOperation{
                   output_mem_config.value_or(input_tensor.memory_config()), dtype},
               {input_tensor},
               {},
               {optional_output_tensor},
               queue_id)
        .at(0);
}

ttnn::Tensor TypecastOperation::invoke(
    const Tensor& input_tensor,
    const DataType& dtype,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return invoke(ttnn::DefaultQueueId, input_tensor, dtype, output_mem_config, optional_output_tensor);
}

}  // namespace ttnn::operations::experimental::copy
