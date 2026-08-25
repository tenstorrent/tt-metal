// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "inbound_socket_service_sync.hpp"

#include "device/inbound_socket_service_sync_device_operation.hpp"

namespace ttnn::experimental {

std::vector<Tensor> inbound_socket_service_sync(
    const tt::tt_metal::H2DStreamService& service,
    uint32_t metadata_size_bytes,
    const std::optional<Tensor>& tokens_out,
    const std::optional<Tensor>& metadata_out) {
    return ttnn::prim::inbound_socket_service_sync(service, metadata_size_bytes, tokens_out, metadata_out);
}

std::vector<Tensor> inbound_socket_service_sync(
    const ttnn::D2DStreamServiceReceiver& service,
    uint32_t metadata_size_bytes,
    const std::optional<Tensor>& tokens_out,
    const std::optional<Tensor>& metadata_out) {
    return ttnn::prim::inbound_socket_service_sync(service, metadata_size_bytes, tokens_out, metadata_out);
}

}  // namespace ttnn::experimental
