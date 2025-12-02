// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/decorators.hpp"
#include <tt-metalium/mesh_socket.hpp>

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteSocketForward {
    static std::vector<ttnn::Tensor> invoke(
        const tt::tt_metal::distributed::MeshSocket& recv_socket,
        const tt::tt_metal::distributed::MeshSocket& send_socket,
        std::size_t num_bytes);
};

}  // namespace operations::experimental::ccl

namespace experimental {

constexpr auto socket_forward = ttnn::register_operation<
    "ttnn::experimental::socket_forward",
    ttnn::operations::experimental::ccl::ExecuteSocketForward>();

}  // namespace experimental
}  // namespace ttnn
