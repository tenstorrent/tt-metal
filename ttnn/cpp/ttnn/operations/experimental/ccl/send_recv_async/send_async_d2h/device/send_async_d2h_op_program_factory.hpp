// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "send_async_d2h_op_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct SendAsyncD2HSharedVariables {
    tt::tt_metal::CoreCoord sender_core_coord{};
    tt::tt_metal::KernelHandle reader_kernel_id{};
};

struct SendAsyncD2HMeshWorkloadFactory {
    using shared_variables_t = SendAsyncD2HSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const SendAsyncD2HParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const Tensor& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
        const SendAsyncD2HParams& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const Tensor& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const SendAsyncD2HParams& operation_attributes,
        const Tensor& tensor_args,
        std::vector<Tensor>& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
