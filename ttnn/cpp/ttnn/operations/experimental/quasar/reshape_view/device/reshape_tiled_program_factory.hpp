// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/quasar/reshape_view/device/reshape_device_operation_types.hpp"

namespace ttnn::prim::qsr {

namespace detail {

// Host-compute the input->output tile page-mapping tensor (segments of contiguous data described
// as {input_page_index, input_page_offset, output_page_offset, num_elements}). Defined in
// reshape_tiled_program_factory.cpp and reused by the Metal-2 tiled factory.
Tensor compute_reshape_mapping_host_tensor(
    uint32_t num_input_pages,
    uint32_t num_output_pages,
    const Shape& input_shape,
    const Shape& output_shape,
    const std::array<uint32_t, 2>& tile_shape,
    const std::array<uint32_t, 2>& face_shape);

}  // namespace detail

struct ReshapeViewTiledProgramFactory {
    // create_workload_descriptor() materializes the host-computed
    // input-to-output page-mapping tensor onto the device and parks the
    // owning Tensor on the WorkloadDescriptor so its backing buffer
    // outlives the cached programs.  The mapping is fully determined by
    // the hashed input/output shapes.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const ReshapeViewParams& operation_attributes,
        const ReshapeViewInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::prim::qsr
