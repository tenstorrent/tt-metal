// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/types.hpp"
#include "all_gather_rms_norm_device_operation_types.hpp"
#include "all_gather_rms_norm_planner.hpp"
#include "all_gather_rms_norm_program_factory.hpp"

namespace ttnn::operations::experimental::ccl::all_gather_rms_norm {

// Hooks on a miss: create_output_tensors, compute_program_hash, validate_on_program_cache_miss, create_descriptor.
// On a hit: create_output_tensors, compute_program_hash, validate_on_program_cache_hit, override_runtime_arguments.
struct AllGatherRmsNormDeviceOperation {
    using operation_attributes_t = AllGatherRmsNormParams;
    using tensor_args_t = AllGatherRmsNormInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<AllGatherRmsNormMeshWorkloadFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    // Support contract, the plan solve (L1 fit) and the scratch spec.
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    // An equal key already passed the miss validator; the entry checks the per-call scratch aliasing.
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// SUPPORTED / EXCLUSIONS / placement refusals of the Python validate(): throws UnsupportedAxisValue or ExcludedCell.
void check_support(const AllGatherRmsNormParams& attrs, const AllGatherRmsNormInputs& t);

// Logical cores of a sharded input's shard grid (the workers); empty for an interleaved input.
std::vector<CoreCoord> input_shard_cores(const Tensor& input);

// The memoized mesh plan of a call.
const planner::MeshPlan& mesh_plan_of(const AllGatherRmsNormParams& attrs, const Tensor& input);

}  // namespace ttnn::operations::experimental::ccl::all_gather_rms_norm

namespace ttnn::prim {

Tensor all_gather_rms_norm(
    const Tensor& input,
    const std::optional<Tensor>& gamma,
    const std::optional<Tensor>& beta,
    const std::optional<Tensor>& output,
    const std::optional<Tensor>& scratch,
    const ttnn::operations::experimental::ccl::all_gather_rms_norm::AllGatherRmsNormParams& params);

}  // namespace ttnn::prim
