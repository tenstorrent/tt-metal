// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "recv_async_op_device_operation_types.hpp"

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include "ttnn/distributed/types.hpp"

namespace ttnn::experimental::prim {

struct RecvAsyncMeshWorkloadFactory {
    // Contract (2): declarative WorkloadDescriptor.  Symmetric to
    // SendAsyncMeshWorkloadFactory but on the receiver side.
    //
    // Only receiver-participating coords get a program; other coords are
    // simply not added.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const RecvAsyncParams& operation_attributes,
        const Tensor& tensor_args,
        std::vector<Tensor>& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::experimental::prim
