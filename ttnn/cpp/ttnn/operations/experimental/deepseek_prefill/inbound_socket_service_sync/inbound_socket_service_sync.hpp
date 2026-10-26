// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn {
class D2DStreamServiceReceiver;
}  // namespace ttnn

namespace tt::tt_metal {
class H2DStreamService;
using D2DStreamServiceReceiver = ttnn::D2DStreamServiceReceiver;
}  // namespace tt::tt_metal

namespace ttnn::experimental {

// Wait for the next H2DStreamService transfer to land in the service's backing
// tensor, copy it into a device tensor, and ack the service core.

// Returns, in order: tokens (the arriving row minus its trailing `overhang_size_bytes`), overhang (when
// `overhang_size_bytes > 0`), and metadata (when `metadata_size_bytes > 0`). Both splits ride the one copy.
//
// The optional trailing arguments are caller-owned destinations, one per output: when
// supplied, the op writes into them instead of allocating a fresh tensor.
std::vector<Tensor> inbound_socket_service_sync(
    const tt::tt_metal::H2DStreamService& service,
    uint32_t metadata_size_bytes = 0,
    uint32_t overhang_size_bytes = 0,
    const std::optional<Tensor>& tokens_out = std::nullopt,
    const std::optional<Tensor>& overhang_out = std::nullopt,
    const std::optional<Tensor>& metadata_out = std::nullopt);

// Same op, but draining a D2DStreamServiceReceiver (disaggregated-prefill
// device->device path). The receiver exposes the same getters as
// H2DStreamService. Same return contract and the same optional destinations.
std::vector<Tensor> inbound_socket_service_sync(
    const ttnn::D2DStreamServiceReceiver& service,
    uint32_t metadata_size_bytes = 0,
    uint32_t overhang_size_bytes = 0,
    const std::optional<Tensor>& tokens_out = std::nullopt,
    const std::optional<Tensor>& overhang_out = std::nullopt,
    const std::optional<Tensor>& metadata_out = std::nullopt);

}  // namespace ttnn::experimental
