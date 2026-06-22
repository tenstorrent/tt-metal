// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>

#include "h2d_socket_sync_device_operation_types.hpp"
#include "h2d_socket_sync_program_factory.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/services/h2d_socket_service.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct H2DSocketSyncOperation {
    using operation_attributes_t = H2DSocketSyncParams;
    using tensor_args_t = H2DSocketSyncInputs;
    // tokens (always) + metadata (only when metadata_size_bytes > 0).
    using spec_return_value_t = std::vector<ttnn::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<H2DSocketSyncProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    // Keys the program on (service identity + config), NOT on the per-call output
    // buffer address (that is a BufferBinding, patched on cache hits).
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

// Launch helper. Snapshots the per-coord service state out of `service`, then
// runs the device operation (which builds-once / caches the program).
// Returns [tokens] or [tokens, metadata] (when metadata_size_bytes > 0).
std::vector<ttnn::Tensor> h2d_socket_sync(const tt::tt_metal::H2DStreamService& service, uint32_t metadata_size_bytes);

}  // namespace ttnn::prim
