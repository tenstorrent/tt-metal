// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>

namespace ttnn::experimental {

// Receives a tensor sent by send_direct_async. The payload is written directly into `output_tensor`
// by the sender; this op only performs the handshake (advertising the sender its output address)
// and waits for the completion token on the socket.
std::vector<ttnn::Tensor> recv_direct_async(
    const ttnn::Tensor& output_tensor, const tt::tt_metal::distributed::MeshSocket& mesh_socket);

}  // namespace ttnn::experimental
