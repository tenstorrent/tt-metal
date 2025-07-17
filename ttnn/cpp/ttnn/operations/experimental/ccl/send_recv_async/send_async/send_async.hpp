// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/decorators.hpp"
#include <tt-metalium/mesh_socket.hpp>

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteSendAsync {
    static std::vector<ttnn::Tensor> invoke(
        const ttnn::Tensor& input_tensor, const tt::tt_metal::distributed::MeshSocket& mesh_socket);
};

}  // namespace operations::experimental::ccl

namespace experimental {

constexpr auto send_async =
    ttnn::register_operation<"ttnn::experimental::send_async", ttnn::operations::experimental::ccl::ExecuteSendAsync>();

}  // namespace experimental
}  // namespace ttnn
