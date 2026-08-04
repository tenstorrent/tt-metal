// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>

namespace ttnn::experimental {

// Receives data sent by buffered_send into a ring of N receive buffers. Which buffer a given send
// lands in is chosen by device-side ring state, not by the caller.
void buffered_recv(
    const std::vector<ttnn::Tensor>& output_tensors, const tt::tt_metal::distributed::MeshSocket& mesh_socket);

}  // namespace ttnn::experimental
