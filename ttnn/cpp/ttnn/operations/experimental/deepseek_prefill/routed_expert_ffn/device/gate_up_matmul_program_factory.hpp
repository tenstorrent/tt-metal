// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "gate_up_matmul_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/mesh_coord.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn {

struct GateUpMatmulSharedVariables {
    tt::tt_metal::KernelHandle reader_x_id = 0;
    tt::tt_metal::KernelHandle reader_weights_id = 0;
    tt::tt_metal::KernelHandle compute_id = 0;
    CoreCoord core;
};

struct GateUpMatmulProgramFactory {
    using shared_variables_t = GateUpMatmulSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;
    using tensor_return_value_t = std::array<Tensor, 2>;

    static cached_mesh_workload_t create_mesh_workload(
        const GateUpMatmulParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const GateUpMatmulInputs& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const GateUpMatmulParams& operation_attributes,
        const GateUpMatmulInputs& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn
