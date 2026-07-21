// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn {
class D2DStreamServiceSender;
}  // namespace ttnn

namespace tt::tt_metal {
using D2DStreamServiceSender = ttnn::D2DStreamServiceSender;
class D2HStreamService;
}  // namespace tt::tt_metal

namespace ttnn::experimental {

// Copy `input` (a producing stage's output) into the D2D sender service's backing
// tensor and inc the service's data_ready_counter, signalling it to forward over
// fabric. The inverse of `inbound_socket_service_sync`.
//
// NON-BLOCKING: it does NOT wait for the forward. The sender forwards once it has
// num_workers acks AND the host grants the fabric lease (`service.release_fabric_links()`,
// called after this op). Back-pressure is the host's `service.wait_for_fabric_links()`
// before the next call -- waiting on-device here would deadlock that cadence.
//
// `input` must have the same per-shard spec as `service.get_backing_tensor()`. When
// `metadata` is provided (a [1,1,1,N] uint32 tensor), the designated worker forwards it
// to the sender service core (metadata mode; its size must match the service config).
// Returns the (now-filled) backing tensor.
Tensor outbound_socket_service_sync(
    const tt::tt_metal::D2DStreamServiceSender& service,
    const Tensor& input,
    const std::optional<Tensor>& metadata = std::nullopt);
Tensor outbound_socket_service_sync(
    const tt::tt_metal::D2HStreamService& service,
    const std::optional<Tensor>& input = std::nullopt,
    const std::optional<Tensor>& metadata = std::nullopt);

}  // namespace ttnn::experimental
