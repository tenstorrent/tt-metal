// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "socket_forward.hpp"

#include <vector>

#include "ttnn/operations/experimental/ccl/send_recv_async/socket_forward/device/socket_forward_device_operation.hpp"

namespace ttnn::operations::experimental::ccl {

std::vector<ttnn::Tensor> ExecuteSocketForward::invoke(
    const ttnn::Tensor& input_tensor,
    const tt::tt_metal::distributed::MeshSocket& recv_socket,
    const tt::tt_metal::distributed::MeshSocket& send_socket,
    std::size_t num_bytes) {
    return ttnn::prim::socket_forward(input_tensor, recv_socket, send_socket, num_bytes);
}

}  // namespace ttnn::operations::experimental::ccl
