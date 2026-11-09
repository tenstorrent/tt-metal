// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include "ttnn/operation.hpp"
#include "agmm_tuned_device_operation_types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

struct AgmmTunedProgramFactory {
    // One program per device: the gather's fabric connections and the K slice order are the device's own.
    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const AgmmTunedParams&,
        const AgmmTunedInputs&,
        std::vector<Tensor>&,
        const ttnn::MeshCoordinateRangeSet&);
};

struct AgmmTunedOperation {
    using operation_attributes_t = AgmmTunedParams;
    using tensor_args_t = AgmmTunedInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<AgmmTunedProgramFactory>;
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

Tensor agmm_tuned(
    const AgmmTunedParams& params, const AgmmTunedInputs& inputs);

// Each device's gather-table rows, devices in row-major mesh order (see kernels/agmm_tuned_table.hpp).
std::vector<std::vector<uint32_t>> agmm_tuned_gather_table_rows(
    const ttnn::MeshDevice& mesh,
    uint32_t cluster_axis,
    uint32_t num_links,
    uint32_t gather_column,
    const std::vector<uint32_t>& counter_addresses);

}  // namespace ttnn::experimental::prim
