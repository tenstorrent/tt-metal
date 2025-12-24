// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "bcast_device_operation.hpp"
#include "ttnn/device_operation.hpp"

#include <tt-logger/tt-logger.hpp>
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::prim {
ttnn::operations::data_movement::bcast::BcastDeviceOperation::tensor_return_value_t bcast(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    ttnn::BcastOpMath bcast_op,
    ttnn::BcastOpDim bcast_dim,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    bool in_place,
    const std::optional<Tensor>& preallocated_output) {
    using OperationType = ttnn::operations::data_movement::bcast::BcastDeviceOperation;
    return ttnn::device_operation::detail::launch_on_device<OperationType>(
        OperationType::operation_attributes_t{
            .math_op = bcast_op, .dim = bcast_dim, .output_mem_config = output_mem_config, .in_place = in_place},
        OperationType::tensor_args_t{
            .input_a = input_tensor_a, .input_b = input_tensor_b, .preallocated_output = preallocated_output});
}
}  // namespace ttnn::prim

namespace ttnn::operations::data_movement::bcast {

}  // namespace ttnn::operations::data_movement::bcast
