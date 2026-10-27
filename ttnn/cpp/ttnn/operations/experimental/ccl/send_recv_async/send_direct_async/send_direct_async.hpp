// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>

namespace ttnn::experimental {

// Writes `input_tensor` straight into the receiver's output tensor, using the socket only to
// exchange addresses and signal completion. Matching receiver op: recv_direct_async.
std::vector<ttnn::Tensor> send_direct_async(
    const ttnn::Tensor& input_tensor, const tt::tt_metal::distributed::MeshSocket& mesh_socket);

}  // namespace ttnn::experimental
