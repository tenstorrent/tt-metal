// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "outbound_socket_service_sync.hpp"

#include "device/outbound_socket_service_sync_device_operation.hpp"

namespace ttnn::experimental {

Tensor outbound_socket_service_sync(
    const ttnn::D2DStreamServiceSender& service, const Tensor& input, const std::optional<Tensor>& metadata) {
    return ttnn::prim::outbound_socket_service_sync(service, input, metadata);
}
Tensor outbound_socket_service_sync(
    const tt::tt_metal::D2HStreamService& service,
    const std::optional<Tensor>& input,
    const std::optional<Tensor>& metadata) {
    return ttnn::prim::outbound_socket_service_sync(service, input, metadata);
}

}  // namespace ttnn::experimental
