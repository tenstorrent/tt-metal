// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <variant>
#include "ttnn/operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
namespace ttnn::experimental::prim {
struct ChronologicalSelectionsParams {
    uint32_t sequence_parallel_axis;
    uint32_t local_rows;
    uint32_t batch_heads;
    uint32_t key_dim;
    uint32_t value_dim;
};
struct ChronologicalSelectionsInputs {
    Tensor actual_start;
};
struct ChronologicalSelectionsFactory {
    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const ChronologicalSelectionsParams&,
        const ChronologicalSelectionsInputs&,
        std::vector<Tensor>&,
        const ttnn::MeshCoordinateRangeSet&);
};
struct ChronologicalSelectionsOperation {
    using operation_attributes_t = ChronologicalSelectionsParams;
    using tensor_args_t = ChronologicalSelectionsInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<ChronologicalSelectionsFactory>;
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};
}  // namespace ttnn::experimental::prim
