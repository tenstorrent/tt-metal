// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"

namespace ttnn::prim {

struct MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory {
    struct shared_variables_t {
        std::vector<tt::tt_metal::KernelHandle> writer_kernel_ids;
        std::vector<CoreCoord> all_worker_cores_ordered;
        tt::tt_metal::CBHandle cb_src2{};
        tt::tt_metal::CBHandle cb_output_reshard{};
    };

    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const ttnn::prim::MatmulParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const ttnn::prim::MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const ttnn::prim::MatmulParams& operation_attributes,
        const ttnn::prim::MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);
};

}  // namespace ttnn::prim
