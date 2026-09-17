// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "moe_fanout_reach_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::moe_fanout_reach {

struct MoeFanoutReachProgramFactory {
    using tensor_return_value_t = ttnn::Tensor;

    // One ProgramDescriptor per mesh coordinate: a chip's reach row is measured from its own position
    // on the ring, so `my_row` is a compile-time arg and the program cannot be replicated.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const MoeFanoutReachParams& operation_attributes,
        const MoeFanoutReachInputs& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_fanout_reach
