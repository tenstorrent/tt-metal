// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/types.hpp"
#include "groupnorm_sc_N_1_HW_C_b_device_operation_types.hpp"
#include "groupnorm_sc_N_1_HW_C_b_program_factory.hpp"

namespace ttnn::operations::groupnorm_sc_N_1_HW_C_b {

// Call order per launch: create_output_tensors, compute_program_hash (the memoized scalar plan, no per-core work; the
// planner gate, PlanNotImplemented), then on a hit
// validate_on_program_cache_hit + the factory's override_runtime_arguments, on a miss
// validate_on_program_cache_miss + create_mesh_workload (create_descriptor per range).
struct GroupnormScN1HwCbDeviceOperation {
    using operation_attributes_t = GroupnormScN1HwCbParams;
    using tensor_args_t = GroupnormScN1HwCbInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<GroupnormScN1HwCbMeshWorkloadFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    // The support contract (structure, SUPPORTED / EXCLUSIONS) and the preallocated output.
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    // An equal key passed the miss validator: only the preallocated output (and its aliasing) is checked again.
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// The Python validate() up to its planner gate: structural ValueErrors (std::invalid_argument) and SUPPORTED /
// EXCLUSIONS (UnsupportedAxisValue / ExcludedCell), in the Python's order. The planner gate is the launch's.
void check_support(const GroupnormScN1HwCbParams& attrs, const GroupnormScN1HwCbInputs& t);

}  // namespace ttnn::operations::groupnorm_sc_N_1_HW_C_b

namespace ttnn::prim {

Tensor groupnorm_sc_N_1_HW_C_b(
    const Tensor& input,
    const std::optional<Tensor>& gamma,
    const std::optional<Tensor>& beta,
    const std::optional<Tensor>& output,
    const ttnn::operations::groupnorm_sc_N_1_HW_C_b::GroupnormScN1HwCbParams& params);

}  // namespace ttnn::prim
