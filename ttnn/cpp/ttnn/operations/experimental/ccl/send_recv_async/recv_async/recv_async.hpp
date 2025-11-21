// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/decorators.hpp"
#include <tt-metalium/mesh_socket.hpp>

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteRecvAsync {
    static std::vector<ttnn::Tensor> invoke(
        const Tensor& output_tensor, const tt::tt_metal::distributed::MeshSocket& mesh_socket);
};

}  // namespace operations::experimental::ccl

namespace experimental {

constexpr auto recv_async =
    ttnn::register_operation<"ttnn::experimental::recv_async", ttnn::operations::experimental::ccl::ExecuteRecvAsync>();

}  // namespace experimental
}  // namespace ttnn
