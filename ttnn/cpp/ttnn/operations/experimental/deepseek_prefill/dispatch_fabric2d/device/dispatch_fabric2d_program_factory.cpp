// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_fabric2d_program_factory.hpp"

#include <tt_stl/assert.hpp>

#include "dispatch_fabric2d_assignments.hpp"
#include "dispatch_fabric2d_placement.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d {

tt::tt_metal::WorkloadDescriptor DispatchFabric2dProgramFactory::create_workload_descriptor(
    const DispatchFabric2dParams& operation_attributes,
    const DispatchFabric2dInputs& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/,
    const ttnn::MeshCoordinateRangeSet& /*tensor_coords*/) {
    auto* mesh = operation_attributes.device;
    const uint32_t extent = static_cast<uint32_t>(mesh->shape()[operation_attributes.axis]);

    // Placement and the schedule are exercised here so a build that reaches this point has proven both
    // against real fabric geometry, even before the kernels exist.
    const auto placement = decide_placement(mesh, operation_attributes.axis, operation_attributes.num_links);

    std::vector<uint32_t> ring_chip_ids(extent);
    for (const auto& coord : ttnn::MeshCoordinateRange(mesh->shape())) {
        if (coord[operation_attributes.axis == 0 ? 1 : 0] != 0) {
            continue;
        }
        ring_chip_ids[coord[operation_attributes.axis]] = mesh->get_fabric_node_id(coord).chip_id;
    }
    for (const auto& [coord, streams] : placement) {
        TT_FATAL(
            streams.size() == stream_count(operation_attributes.num_links),
            "dispatch_fabric2d: chip {} got {} streams, expected {}",
            coord,
            streams.size(),
            stream_count(operation_attributes.num_links));
        (void)generate_assignments(ring_chip_ids, coord[operation_attributes.axis], operation_attributes.num_links);
    }

    TT_THROW("dispatch_fabric2d: kernels are not implemented yet");
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d
