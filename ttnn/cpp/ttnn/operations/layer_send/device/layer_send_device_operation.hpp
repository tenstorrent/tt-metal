// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "layer_send_device_operation_types.hpp"
#include "layer_send_program_factory.hpp"

namespace ttnn::operations::layer_send {

// In place: the op writes into its io tensors and returns io[0]. The key is the plan's hash and, per io tensor, what
// its TensorAccessor compile-time args depend on (buffer type and layout; a sharded one's memory config and padded
// shape: FIX.32); never an address or a per-call value.
struct LayerSendDeviceOperation {
    using operation_attributes_t = LayerSendParams;
    using tensor_args_t = LayerSendInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<LayerSendMeshWorkloadFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::layer_send

namespace ttnn::prim {

Tensor layer_send(
    const std::vector<Tensor>& io,
    const std::shared_ptr<const ttnn::operations::layer_send::SendPlan>& plan,
    const std::vector<uint32_t>& values);

}  // namespace ttnn::prim
