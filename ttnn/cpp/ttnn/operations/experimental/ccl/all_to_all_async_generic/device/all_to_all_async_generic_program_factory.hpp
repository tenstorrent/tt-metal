// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "all_to_all_async_generic_device_operation_types.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::experimental::prim {

struct AllToAllAsyncGenericProgram {
    struct ResolvedRouting {
        uint32_t num_links;
        ttnn::ccl::Topology topology;
        tt::tt_fabric::Topology axis_topology;
        bool axis_is_straight;
    };

    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const AllToAllAsyncGenericParams& operation_attributes,
        const AllToAllAsyncGenericInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::experimental::prim
