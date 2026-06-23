// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "buffered_send.hpp"

#include <vector>

#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include "ttnn/operations/experimental/ccl/send_recv_async/buffered_send/device/buffered_send_op_device_operation.hpp"

namespace ttnn::experimental {

std::vector<ttnn::Tensor> buffered_send(
    const ttnn::Tensor& input_tensor, const tt::tt_metal::distributed::MeshSocket& mesh_socket) {
    return ttnn::prim::buffered_send(input_tensor, mesh_socket);
}

}  // namespace ttnn::experimental
