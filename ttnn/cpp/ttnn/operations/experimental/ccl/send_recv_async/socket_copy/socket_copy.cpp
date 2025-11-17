// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "socket_copy.hpp"

#include <vector>

#include "ttnn/operations/experimental/ccl/send_recv_async/socket_copy/device/socket_copy_op.hpp"

namespace ttnn::operations::experimental::ccl {

std::vector<ttnn::Tensor> ExecuteSocketCopy::invoke(
    const ttnn::Tensor& input_tensor,
    const tt::tt_metal::distributed::MeshSocket& recv_socket,
    const tt::tt_metal::distributed::MeshSocket& send_socket,
    std::size_t num_bytes) {
    return socket_copy(input_tensor, recv_socket, send_socket, num_bytes);
}

}  // namespace ttnn::operations::experimental::ccl
