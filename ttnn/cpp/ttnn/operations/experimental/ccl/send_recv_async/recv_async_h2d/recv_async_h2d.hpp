// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>

namespace ttnn::experimental {

std::vector<ttnn::Tensor> recv_async_h2d(
    const ttnn::Tensor& output_tensor, const tt::tt_metal::distributed::H2DSocket& h2d_socket);

}  // namespace ttnn::experimental
