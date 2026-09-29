// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>

namespace ttnn::experimental {

// Writes `input_tensor` straight into the receiver's output tensor, using the socket only to
// exchange addresses and signal completion. Matching receiver op: recv_direct_async.
// With `static_dst_address` set, the address exchange is skipped: pages go straight to that
// receiver buffer base address and only the completion page is pushed (pair with
// recv_direct_async(wait_only=true)).
std::vector<ttnn::Tensor> send_direct_async(
    const ttnn::Tensor& input_tensor,
    const tt::tt_metal::distributed::MeshSocket& mesh_socket,
    std::optional<uint32_t> static_dst_address = std::nullopt,
    std::optional<uint32_t> num_pages = std::nullopt);

}  // namespace ttnn::experimental
