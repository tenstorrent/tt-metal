// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/mesh_buffer.hpp>

#include "buffered_send_op_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct BufferedSendSharedVariables {
    std::vector<tt::tt_metal::CoreCoord> sender_core_coords;
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    // Persistent L1_SMALL buffer that backs the handshake landing zone (where the receiver writes
    // the OutputTensorInfo struct) and the advertise stage. Kept alive here so the device-side
    // allocation outlives the cached workload and its address stays stable across reuse.
    tt::tt_metal::distributed::AnyBuffer info_buffer;
};

struct BufferedSendMeshWorkloadFactory {
    using shared_variables_t = BufferedSendSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const BufferedSendParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const Tensor& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
        const BufferedSendParams& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const Tensor& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const BufferedSendParams& operation_attributes,
        const Tensor& tensor_args,
        std::vector<Tensor>& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
