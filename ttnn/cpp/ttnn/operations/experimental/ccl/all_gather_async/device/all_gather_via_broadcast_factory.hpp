// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "all_gather_async_device_operation_types.hpp"

#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct AllGatherViaBroadcastFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const AllGatherAsyncParams& operation_attributes,
        const AllGatherAsyncInputs& tensor_args,
        Tensor& output_tensor,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::experimental::prim
