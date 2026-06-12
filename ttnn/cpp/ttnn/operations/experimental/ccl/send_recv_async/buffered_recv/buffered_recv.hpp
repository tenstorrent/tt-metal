// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/global_semaphore.hpp"
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>

namespace ttnn::experimental {

// Receives buffered data sent by buffered_send. Unlike recv_direct_async (which takes a single
// output tensor), buffered_recv takes N output tensors that act as a ring of receive buffers, the
// receive socket, and a global semaphore used to coordinate buffer availability.
//
// NOTE: This is currently a skeleton implementation; the full buffered receive logic is not yet
// wired up.
ttnn::Tensor buffered_recv(
    const std::vector<ttnn::Tensor>& output_tensors,
    const tt::tt_metal::distributed::MeshSocket& mesh_socket,
    const tt::tt_metal::GlobalSemaphore& global_semaphore);

}  // namespace ttnn::experimental
