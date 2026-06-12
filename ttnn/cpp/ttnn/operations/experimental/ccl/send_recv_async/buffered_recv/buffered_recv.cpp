// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "buffered_recv.hpp"

#include <vector>

#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include "ttnn/operations/experimental/ccl/send_recv_async/buffered_recv/device/buffered_recv_op_device_operation.hpp"

namespace ttnn::experimental {

std::vector<ttnn::Tensor> buffered_recv(
    const std::vector<ttnn::Tensor>& output_tensors,
    const tt::tt_metal::distributed::MeshSocket& mesh_socket,
    const tt::tt_metal::GlobalSemaphore& global_semaphore) {
    return ttnn::prim::buffered_recv(output_tensors, mesh_socket, global_semaphore);
}

}  // namespace ttnn::experimental
