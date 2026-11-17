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

    // The arrival semaphore is the caller's and left out of the program hash: a cache hit
    // writes its address into the gather kernel's common args.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const AttnResBlockStatsParams& operation_attributes,
        const AttnResBlockStatsInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_coordinate);
};

}  // namespace ttnn::experimental::prim
