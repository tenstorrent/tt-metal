// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>

namespace ttnn::experimental {

// Sends `input_tensor` directly into the receiver's output tensor (bypassing the socket FIFO for
// payload data). The socket is only used to advertise the sender handshake-buffer address and to
// signal completion. See recv_direct_async for the matching receiver op.
std::vector<ttnn::Tensor> send_direct_async(
    const ttnn::Tensor& input_tensor, const tt::tt_metal::distributed::MeshSocket& mesh_socket);

}  // namespace ttnn::experimental
