// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "moe_gpt_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::operations::experimental::moe_gpt::program {

struct MoEGPTMeshWorkloadFactory {
    // Contract (2): declarative WorkloadDescriptor.  Builds one
    // ProgramDescriptor per coord.  All semaphores are program-scoped
    // (reserved via SemaphoreDescriptor slots).  Dynamic CBs that point at
    // input/intermediate/output tensor buffers are wired up via
    // CBDescriptor::buffer so the framework patches their addresses on every
    // dispatch.  Per-core runtime args use Buffer* binding for tensor
    // addresses so the framework patches them on every dispatch.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::operations::experimental::moe_gpt::program
