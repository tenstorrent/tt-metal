// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/reshape_view/device/reshape_device_operation_types.hpp"

namespace ttnn::prim {

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

}  // namespace ttnn::prim
