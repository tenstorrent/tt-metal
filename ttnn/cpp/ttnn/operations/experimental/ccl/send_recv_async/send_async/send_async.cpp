// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "send_async.hpp"

#include <vector>

#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include "ttnn/operations/experimental/ccl/send_recv_async/send_async/device/send_async_op_device_operation.hpp"
#include "ttnn/graph/composite_trace.hpp"

namespace ttnn::experimental {

std::vector<ttnn::Tensor> send_async(
    const ttnn::Tensor& input_tensor, const tt::tt_metal::distributed::MeshSocket& mesh_socket) {
    ttnn::graph::ScopedCompositeTrace _trace("ttnn::experimental::send_async");
    return ttnn::prim::send_async(input_tensor, mesh_socket);
}

}  // namespace ttnn::experimental
