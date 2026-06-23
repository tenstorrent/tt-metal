// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "h2d_socket_sync.hpp"

#include "device/h2d_socket_sync_device_operation.hpp"

namespace ttnn::experimental {

std::vector<Tensor> h2d_socket_sync(const tt::tt_metal::H2DStreamService& service, uint32_t metadata_size_bytes) {
    return ttnn::prim::h2d_socket_sync(service, metadata_size_bytes);
}

}  // namespace ttnn::experimental
