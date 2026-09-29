// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "send_direct_async.hpp"

#include <vector>

#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include "ttnn/operations/experimental/ccl/send_recv_async/send_direct_async/device/send_direct_async_op_device_operation.hpp"

namespace ttnn::experimental {

std::vector<ttnn::Tensor> send_direct_async(
    const ttnn::Tensor& input_tensor,
    const tt::tt_metal::distributed::MeshSocket& mesh_socket,
    std::optional<uint32_t> static_dst_address,
    std::optional<uint32_t> num_pages) {
    return ttnn::prim::send_direct_async(input_tensor, mesh_socket, static_dst_address, num_pages);
}

}  // namespace ttnn::experimental
