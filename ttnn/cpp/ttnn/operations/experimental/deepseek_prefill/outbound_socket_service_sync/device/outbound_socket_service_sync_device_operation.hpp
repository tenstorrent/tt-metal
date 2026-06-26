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
    // Keys the program on (service identity + config), NOT on the per-call input
    // buffer address (that is a BufferBinding, patched on cache hits).
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

// Launch helper. Snapshots the per-coord sender-service state out of `service`, then
// runs the device operation (build-once / program-cached). Copies `input` into the
// sender backing tensor and inc's the service's data_ready_counter; does NOT wait for
// the fabric forward (the host drives the lease via release_fabric_links /
// wait_for_fabric_links). Returns the (now-filled) backing tensor.
ttnn::Tensor outbound_socket_service_sync(
    const tt::tt_metal::D2DStreamServiceSender& service,
    const ttnn::Tensor& input,
    const std::optional<ttnn::Tensor>& metadata = std::nullopt);

}  // namespace ttnn::prim
