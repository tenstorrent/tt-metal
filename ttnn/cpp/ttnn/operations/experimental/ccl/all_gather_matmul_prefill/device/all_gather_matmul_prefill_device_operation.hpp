// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include "ttnn/operation.hpp"
#include "all_gather_matmul_prefill_device_operation_types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

struct AllGatherMatmulPrefillProgramFactory {
    // One program per device: the gather's fabric connections and the K slice order are the device's own.
    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const AllGatherMatmulPrefillParams&,
        const AllGatherMatmulPrefillInputs&,
        std::vector<Tensor>&,
        const ttnn::MeshCoordinateRangeSet&);
};

struct AllGatherMatmulPrefillOperation {
    using operation_attributes_t = AllGatherMatmulPrefillParams;
    using tensor_args_t = AllGatherMatmulPrefillInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<AllGatherMatmulPrefillProgramFactory>;
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

Tensor all_gather_matmul_prefill(
    const AllGatherMatmulPrefillParams& params, const AllGatherMatmulPrefillInputs& inputs);

}  // namespace ttnn::experimental::prim
