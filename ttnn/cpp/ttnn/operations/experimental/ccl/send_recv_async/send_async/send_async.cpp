// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "send_async.hpp"

#include <vector>

#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include "ttnn/operations/experimental/ccl/send_recv_async/send_async/device/send_async_op_device_operation.hpp"

namespace ttnn::experimental {

std::vector<ttnn::Tensor> send_async(
    const ttnn::Tensor& input_tensor, const tt::tt_metal::distributed::MeshSocket& mesh_socket) {
    return ttnn::prim::send_async(input_tensor, mesh_socket);
}

}  // namespace ttnn::experimental
