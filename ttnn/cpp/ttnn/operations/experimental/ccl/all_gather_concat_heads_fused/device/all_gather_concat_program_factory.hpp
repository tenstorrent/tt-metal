// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/all_gather_concat_device_operation_types.hpp"

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::experimental::prim {

struct AllGatherConcatMeshWorkloadFactory {
    // Contract (2): declarative WorkloadDescriptor.  Builds one
    // ProgramDescriptor per coord (ring_index / forward & backward neighbours
    // vary across the mesh).
    //
    // The single GlobalSemaphore lives on AllGatherConcatParams (caller
    // allocated) so this factory needs no workload-scoped resources.  The
    // dynamic CB that points at the output tensor's buffer is wired up via
    // CBDescriptor::buffer so the framework patches its address on every
    // dispatch.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const AllGatherConcatParams& operation_attributes,
        const AllGatherConcatInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::experimental::prim
