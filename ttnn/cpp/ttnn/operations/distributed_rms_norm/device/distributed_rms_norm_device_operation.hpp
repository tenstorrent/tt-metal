// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>

#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/types.hpp"
#include "distributed_rms_norm_device_operation_types.hpp"
#include "distributed_rms_norm_program_factory.hpp"

namespace ttnn::operations::distributed_rms_norm {

// RMSNorm over a hidden dim sharded across one mesh axis, fused with the cross-device exchange of the Σx² partials.
//
// Call order (ttnn/api/ttnn/device_operation.hpp): create_output_tensors (the entry always passes the output and the
// exchange landing, so it returns the given output), compute_program_hash, then on a hit
// validate_on_program_cache_hit + override_runtime_arguments, on a miss validate_on_program_cache_miss +
// create_descriptor once per mesh coordinate.
struct DistributedRmsNormDeviceOperation {
    using operation_attributes_t = DistributedRmsNormParams;
    using tensor_args_t = DistributedRmsNormInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<DistributedRmsNormProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    // The plan (regime, L1 fit), the output and scratch specs, the devices and the aliasing rules.
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    // An equal key fixed the plan and every spec: only the devices and the aliasing of this call's buffers remain.
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// The caller errors of the Python _hard_validate (std::invalid_argument, raised as ValueError), in its order.
struct GroupResolution {
    uint32_t group_size = 0;
    uint32_t usable_links = 0;
    bool ring = false;
};
GroupResolution check_caller_inputs(
    const Tensor& input,
    int32_t cluster_axis,
    const std::optional<Tensor>& gamma,
    tt::tt_fabric::Topology topology,
    std::optional<int32_t> num_links);

// The support contract (SUPPORTED, the requested output memory_layout, EXCLUSIONS) and the regime refusals checked
// before dispatch (sharded output without a shard spec, mcast on a 2D fabric), in the Python's order.
void check_support(
    const DistributedRmsNormParams& attrs,
    const DistributedRmsNormInputs& t,
    const std::optional<tt::tt_metal::MemoryConfig>& requested_memory_config);

}  // namespace ttnn::operations::distributed_rms_norm

namespace ttnn::prim {

Tensor distributed_rms_norm(
    const Tensor& input,
    const std::optional<Tensor>& gamma,
    const std::optional<Tensor>& output,
    const std::optional<Tensor>& scratch,
    const ttnn::operations::distributed_rms_norm::DistributedRmsNormParams& params);

}  // namespace ttnn::prim
