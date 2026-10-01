// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "send_async_d2h.hpp"

#include <vector>

#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include "ttnn/operations/experimental/ccl/send_recv_async/send_async_d2h/device/send_async_d2h_op_device_operation.hpp"

namespace ttnn::experimental {

std::vector<ttnn::Tensor> send_async_d2h(
    const ttnn::Tensor& input_tensor, const tt::tt_metal::distributed::D2HSocket& d2h_socket) {
    return ttnn::prim::send_async_d2h(input_tensor, d2h_socket);
}

}  // namespace ttnn::experimental
