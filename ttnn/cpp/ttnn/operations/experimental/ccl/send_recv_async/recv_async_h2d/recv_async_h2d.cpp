// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "recv_async_h2d.hpp"

#include <vector>

#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include "ttnn/operations/experimental/ccl/send_recv_async/recv_async_h2d/device/recv_async_h2d_op_device_operation.hpp"

namespace ttnn::experimental {

std::vector<ttnn::Tensor> recv_async_h2d(
    const ttnn::Tensor& output_tensor, const tt::tt_metal::distributed::H2DSocket& h2d_socket) {
    return ttnn::prim::recv_async_h2d(output_tensor, h2d_socket);
}

}  // namespace ttnn::experimental
