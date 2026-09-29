// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>

namespace ttnn::experimental {

// Receives a tensor sent by send_direct_async. The sender writes `output_tensor` directly, so this op
// only advertises its address and then waits for the completion token. With `wait_only`, the address
// is not advertised (the sender was given it via static_dst_address); only the completion is awaited.
std::vector<ttnn::Tensor> recv_direct_async(
    const ttnn::Tensor& output_tensor,
    const tt::tt_metal::distributed::MeshSocket& mesh_socket,
    bool wait_only = false);

}  // namespace ttnn::experimental
