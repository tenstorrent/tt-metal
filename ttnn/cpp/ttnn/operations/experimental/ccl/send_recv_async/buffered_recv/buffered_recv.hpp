// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>

namespace ttnn::experimental {

// Receives buffered data sent by buffered_send. Unlike recv_direct_async (which takes a single
// output tensor), buffered_recv takes N output tensors that act as a ring of receive buffers and the
// receive socket. Buffer availability is coordinated through an internally-allocated, zero-
// initialized persistent L1_SMALL buffer (no caller-provided global semaphore is required).
//
// The actual buffer written by each send is selected by device-side ring state.
void buffered_recv(
    const std::vector<ttnn::Tensor>& output_tensors, const tt::tt_metal::distributed::MeshSocket& mesh_socket);

}  // namespace ttnn::experimental
