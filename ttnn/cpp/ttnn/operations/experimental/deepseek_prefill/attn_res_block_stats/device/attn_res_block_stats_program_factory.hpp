// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include <tt-metalium/workload_descriptor.hpp>

#include "attn_res_block_stats_device_operation_types.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::experimental::prim {

struct AttnResBlockStatsProgramFactory {
    // One program per chip: where the op gathers across an axis, each chip's rank and fabric
    // routes differ.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const AttnResBlockStatsParams& operation_attributes,
        const AttnResBlockStatsInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::experimental::prim
