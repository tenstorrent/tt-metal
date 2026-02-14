// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "recv_async_op_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct RecvAsyncSharedVariables {
    std::vector<tt::tt_metal::CoreCoord> receiver_core_coords;
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    bool socket_storage_in_dram = false;
};

struct RecvAsyncMeshWorkloadFactory {
    using shared_variables_t = RecvAsyncSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const RecvAsyncParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const Tensor& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
        const RecvAsyncParams& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const Tensor& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const RecvAsyncParams& operation_attributes,
        const Tensor& tensor_args,
        std::vector<Tensor>& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
