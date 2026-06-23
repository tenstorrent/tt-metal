// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>

namespace ttnn::experimental {

// Sends `input_tensor` directly into the receiver's output tensor (bypassing the socket FIFO for
// payload data). Behaves the same as send_direct_async. See buffered_recv for the matching
// receiver op.
std::vector<ttnn::Tensor> buffered_send(
    const ttnn::Tensor& input_tensor, const tt::tt_metal::distributed::MeshSocket& mesh_socket);

}  // namespace ttnn::experimental
