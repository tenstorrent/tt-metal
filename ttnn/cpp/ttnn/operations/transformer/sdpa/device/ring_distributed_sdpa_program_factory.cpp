// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/recipe/sdpa_recipe_ring_distributed.hpp"
#include "ring_distributed_sdpa_device_operation.hpp"
#include <tt-logger/tt-logger.hpp>

using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

// Resolve ring_id either from explicit override or by inferring from the mesh coordinate.
uint32_t resolve_ring_id(
    const RingDistributedSDPAParams& operation_attributes,
    const RingDistributedSDPAInputs& tensor_args,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    if (operation_attributes.ring_id.has_value()) {
        // Use explicitly provided ring_id
        return operation_attributes.ring_id.value();
    }

    // Infer ring_id directly from the mesh coordinate. ring_size constrains the ring to one
    // of the two mesh axes, so the index along that axis IS the ring_id - no need to look up
    // the device by coord (which uses the deprecated MeshDevice::get_device(MeshCoordinate)
    // and is multi-host-unsafe).
    TT_FATAL(
        mesh_dispatch_coordinate.has_value(),
        "mesh_dispatch_coordinate must be provided when ring_id is not explicitly set");
    auto* mesh_device = tensor_args.q.device();
    TT_FATAL(mesh_device != nullptr, "Mesh device must not be null when inferring ring_id");

    const auto& mesh_coord = mesh_dispatch_coordinate.value();
    const auto& mesh_view = mesh_device->get_view();
    if (mesh_view.shape()[0] == operation_attributes.ring_size) {
        // Ring is along axis 0 (rows of the mesh). ring_id = row index.
        const uint32_t curr_ring_id = mesh_coord[0];
        log_debug(tt::LogOp, "Inferred ring_id (axis 0): {}", curr_ring_id);
        return curr_ring_id;
    }
    if (mesh_view.shape()[1] == operation_attributes.ring_size) {
        // Ring is along axis 1 (columns of the mesh). ring_id = column index.
        const uint32_t curr_ring_id = mesh_coord[1];
        log_debug(tt::LogOp, "Inferred ring_id (axis 1): {}", curr_ring_id);
        return curr_ring_id;
    }
    TT_FATAL(
        false,
        "Ring size {} doesn't match mesh dimensions [{}, {}]",
        operation_attributes.ring_size,
        mesh_view.shape()[0],
        mesh_view.shape()[1]);
    return 0;  // unreachable; satisfies non-void return.
}

}  // namespace

// One recipe program per mesh coordinate: that device's two Q slabs (device/recipe/sdpa_recipe_ring_distributed).
WorkloadDescriptor RingDistributedSdpaDeviceOperation::RingDistributedSdpaProgramFactory::create_workload_descriptor(
    const RingDistributedSDPAParams& operation_attributes,
    const RingDistributedSDPAInputs& tensor_args,
    Tensor& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    WorkloadDescriptor wd;
    const auto coords = tensor_coords.coords();
    wd.programs.reserve(coords.size());
    for (const auto& coord : coords) {
        auto desc = ttnn::operations::transformer::sdpa::detail::ring_distributed_recipe_descriptor(
            operation_attributes,
            tensor_args,
            tensor_return_value,
            resolve_ring_id(operation_attributes, tensor_args, coord));
        wd.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }
    return wd;
}

}  // namespace ttnn::prim
