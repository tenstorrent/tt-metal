// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "send_async.hpp"
#include <utility>
#include <tt-metalium/mesh_socket.hpp>
#include "ttnn/operations/experimental/ccl/send_async/device/send_async_op.hpp"

namespace ttnn::operations::experimental::ccl {

std::vector<ttnn::Tensor> ExecuteSendAsync::invoke(
    const ttnn::Tensor& input_tensor, const tt::tt_metal::distributed::MeshSocket& mesh_socket) {
    send_async(input_tensor, mesh_socket);
    return {};
}

}  // namespace ttnn::operations::experimental::ccl
