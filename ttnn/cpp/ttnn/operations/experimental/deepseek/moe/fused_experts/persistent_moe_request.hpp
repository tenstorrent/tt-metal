// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::deepseek::moe {

// One request to the persistent fused_experts op from every chip of the input's mesh that sends on
// request_socket: the [64-byte header with `layer` in word 0 | input row] page, then -- unless `layer`
// is the stop id 0xFFFFFFFF -- the reduced [1, H] row from response_socket, returned in the input's spec.
Tensor persistent_moe_request(
    const Tensor& input_tensor,
    uint32_t layer,
    const tt::tt_metal::distributed::MeshSocket& request_socket,
    const tt::tt_metal::distributed::MeshSocket& response_socket);

}  // namespace ttnn::experimental::deepseek::moe
