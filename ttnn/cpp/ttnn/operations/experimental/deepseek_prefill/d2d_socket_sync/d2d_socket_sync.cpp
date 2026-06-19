// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "d2d_socket_sync.hpp"

#include "device/d2d_socket_sync_device_operation.hpp"

namespace ttnn::experimental {

Tensor d2d_socket_sync(
    const tt::tt_metal::D2DStreamServiceSender& service, const Tensor& input, const std::optional<Tensor>& metadata) {
    return ttnn::prim::d2d_socket_sync(service, input, metadata);
}

}  // namespace ttnn::experimental
