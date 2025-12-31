// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "recv_async.hpp"
#include "device/recv_async_op_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental {

std::vector<ttnn::Tensor> recv_async(
    const ttnn::Tensor& output_tensor, const tt::tt_metal::distributed::MeshSocket& mesh_socket) {
    using OperationType = operations::experimental::ccl::recv_async::RecvAsyncDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t(mesh_socket);
    auto tensor_args = OperationType::tensor_args_t{.output_tensor = output_tensor};

    return device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::experimental
