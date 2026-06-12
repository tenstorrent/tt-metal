// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/mesh_buffer.hpp>

#include "buffered_recv_op_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct BufferedRecvSharedVariables {
    std::vector<tt::tt_metal::CoreCoord> receiver_core_coords;
    tt::tt_metal::KernelHandle handshake_kernel_id{};
    // Internally-allocated, zero-initialized persistent L1_SMALL buffer used to coordinate receive-
    // buffer availability (replaces the previously caller-provided global semaphore). Kept alive
    // here so the device-side allocation outlives the cached workload and its address stays stable.
    tt::tt_metal::distributed::AnyBuffer coordination_buffer;
};

struct BufferedRecvMeshWorkloadFactory {
    using shared_variables_t = BufferedRecvSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const BufferedRecvParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const std::vector<Tensor>& tensor_args,
        Tensor& tensor_return_value);

    static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
        const BufferedRecvParams& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const std::vector<Tensor>& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const BufferedRecvParams& operation_attributes,
        const std::vector<Tensor>& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
