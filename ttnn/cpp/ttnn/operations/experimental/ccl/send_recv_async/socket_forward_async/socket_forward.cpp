// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "socket_forward.hpp"

#include <vector>

#include "ttnn/operations/experimental/ccl/send_recv_async/socket_forward_async/device/socket_forward_op.hpp"

namespace ttnn::operations::experimental::ccl {

std::vector<ttnn::Tensor> ExecuteSocketForward::invoke(
    const tt::tt_metal::distributed::MeshSocket& recv_socket,
    const tt::tt_metal::distributed::MeshSocket& send_socket,
    std::size_t num_bytes) {
    return socket_forward(recv_socket, send_socket, num_bytes);
}

}  // namespace ttnn::operations::experimental::ccl
