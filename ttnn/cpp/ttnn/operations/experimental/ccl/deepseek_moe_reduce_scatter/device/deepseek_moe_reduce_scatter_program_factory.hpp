// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "deepseek_moe_reduce_scatter_device_operation_types.hpp"

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::experimental::prim {

struct DeepseekMoEReduceScatterMeshWorkloadFactory {
    // Contract (2): declarative WorkloadDescriptor.  Builds one
    // ProgramDescriptor per coord.
    //
    // The two workload-scoped GlobalSemaphores (op + pre_op_barrier) are
    // allocated up front and parked in wd.semaphores so they outlive the
    // cached MeshWorkload.  Dynamic CBs that point at input/intermediate
    // tensor buffers are wired up via CBDescriptor::buffer so the framework
    // patches their addresses on every dispatch.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const DeepseekMoEReduceScatterParams& operation_attributes,
        const DeepseekMoEReduceScatterInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::experimental::prim
