// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "recv_direct_async_op_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct RecvDirectAsyncSharedVariables {
    std::vector<tt::tt_metal::CoreCoord> receiver_core_coords;
    tt::tt_metal::KernelHandle handshake_kernel_id{};
};

struct RecvDirectAsyncMeshWorkloadFactory {
    using shared_variables_t = RecvDirectAsyncSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const RecvDirectAsyncParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const Tensor& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
        const RecvDirectAsyncParams& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const Tensor& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const RecvDirectAsyncParams& operation_attributes,
        const Tensor& tensor_args,
        std::vector<Tensor>& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
