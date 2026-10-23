// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>

namespace ttnn::experimental {

std::vector<ttnn::Tensor> send_async_d2h(
    const ttnn::Tensor& input_tensor, const tt::tt_metal::distributed::D2HSocket& d2h_socket);

}  // namespace ttnn::experimental
