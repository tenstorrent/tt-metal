// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>

namespace ttnn {
namespace experimental {

std::vector<ttnn::Tensor> recv_async(
    const ttnn::Tensor& output_tensor, const tt::tt_metal::distributed::MeshSocket& mesh_socket);

}  // namespace experimental
}  // namespace ttnn
