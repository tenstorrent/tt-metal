// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>

#include "inbound_socket_service_sync_device_operation_types.hpp"
#include "inbound_socket_service_sync_program_factory.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/services/h2d_socket_service.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct InboundSocketServiceSyncOperation {
    using operation_attributes_t = InboundSocketServiceSyncParams;
    using tensor_args_t = InboundSocketServiceSyncInputs;
    // tokens (always) + metadata (only when metadata_size_bytes > 0).
    using spec_return_value_t = std::vector<ttnn::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<InboundSocketServiceSyncProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn {
class D2DStreamServiceReceiver;
}  // namespace ttnn

namespace tt::tt_metal {
using D2DStreamServiceReceiver = ttnn::D2DStreamServiceReceiver;
}  // namespace tt::tt_metal

namespace ttnn::prim {

// Launch helper. Snapshots the per-coord service state out of `service`, then
// runs the device operation (which builds-once / caches the program).
// Returns [tokens] or [tokens, metadata] (when metadata_size_bytes > 0).
std::vector<ttnn::Tensor> inbound_socket_service_sync(
    const tt::tt_metal::H2DStreamService& service, uint32_t metadata_size_bytes);

// Same op, draining a D2DStreamServiceReceiver's backing tensor (disaggregated-
// prefill device->device path). The receiver exposes the same getters as
// H2DStreamService, so it runs the identical device operation.
std::vector<ttnn::Tensor> inbound_socket_service_sync(
    const tt::tt_metal::D2DStreamServiceReceiver& service, uint32_t metadata_size_bytes);

}  // namespace ttnn::prim
