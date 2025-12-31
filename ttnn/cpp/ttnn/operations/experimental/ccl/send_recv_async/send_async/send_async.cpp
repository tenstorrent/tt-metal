// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "send_async.hpp"
#include "device/send_async_op_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental {

std::vector<ttnn::Tensor> send_async(
    const ttnn::Tensor& input_tensor, const tt::tt_metal::distributed::MeshSocket& mesh_socket) {
    using OperationType = operations::experimental::ccl::send_async::SendAsyncDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t(mesh_socket);
    auto tensor_args = OperationType::tensor_args_t{.input_tensor = input_tensor};

    return device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::experimental
