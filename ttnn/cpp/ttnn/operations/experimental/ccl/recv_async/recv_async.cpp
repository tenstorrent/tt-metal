// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "recv_async.hpp"
#include <utility>
#include <tt-metalium/mesh_socket.hpp>
#include "ttnn/operations/experimental/ccl/recv_async/device/recv_async_op.hpp"

namespace ttnn::operations::experimental::ccl {

std::vector<ttnn::Tensor> ExecuteRecvAsync::invoke(
    const Tensor& output_tensor, const tt::tt_metal::distributed::MeshSocket& mesh_socket) {
    recv_async(output_tensor, mesh_socket);
    return {};
}

}  // namespace ttnn::operations::experimental::ccl
