// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "recv_direct_async.hpp"

#include <vector>

#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include "ttnn/operations/experimental/ccl/send_recv_async/recv_direct_async/device/recv_direct_async_op_device_operation.hpp"

namespace ttnn::experimental {

std::vector<ttnn::Tensor> recv_direct_async(
    const ttnn::Tensor& output_tensor, const tt::tt_metal::distributed::MeshSocket& mesh_socket) {
    return ttnn::prim::recv_direct_async(output_tensor, mesh_socket);
}

}  // namespace ttnn::experimental
