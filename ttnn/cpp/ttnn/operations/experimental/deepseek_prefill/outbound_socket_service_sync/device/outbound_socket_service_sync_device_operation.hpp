// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "outbound_socket_service_sync_device_operation_types.hpp"
#include "outbound_socket_service_sync_program_factory.hpp"

namespace tt::tt_metal {
class D2DStreamServiceSender;
class D2HStreamService;
}

namespace ttnn::experimental::prim {

struct OutboundSocketServiceSyncOperation {
    using operation_attributes_t = OutboundSocketServiceSyncParams;
    using tensor_args_t = OutboundSocketServiceSyncInputs;
    // The op fills the service-owned backing tensor in place; the (single) output IS
    // that backing tensor (no new allocation).
    using spec_return_value_t = ttnn::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;
    using program_factory_t = std::variant<OutboundSocketServiceSyncProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

// Launch helpers (one templated body, see .cpp): snapshot per-coord service state, run the
// program-cached op, and inc the service's data_ready counter. NON-BLOCKING.
//   * D2D: copies `input` into the sender backing; forward waits on the host fabric lease
//     (release_fabric_links / wait_for_fabric_links).
//   * D2H: {backing+metadata, backing-only, metadata-only}; metadata-only has no payload or
//     lease. Returns the backing tensor (the record stands in for it in metadata-only).
ttnn::Tensor outbound_socket_service_sync(
    const tt::tt_metal::D2DStreamServiceSender& service,
    const ttnn::Tensor& input,
    const std::optional<ttnn::Tensor>& metadata = std::nullopt);

// D2H service: {backing+metadata, backing-only, metadata-only}. `input` is the payload to
// stream (omit for metadata-only); `metadata` is the record (omit for payload-only). At
// least one must be set.
ttnn::Tensor outbound_socket_service_sync(
    const tt::tt_metal::D2HStreamService& service,
    const std::optional<ttnn::Tensor>& input = std::nullopt,
    const std::optional<ttnn::Tensor>& metadata = std::nullopt);

}  // namespace ttnn::prim
