// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "all_gather_async_device_operation_types.hpp"

#include <tt-metalium/workload_descriptor.hpp>

#include <optional>

namespace ttnn::experimental::prim {

struct LlamaShardedMeshWorkloadFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const AllGatherAsyncParams& operation_attributes,
        const AllGatherAsyncInputs& tensor_args,
        Tensor& output_tensor,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const AllGatherAsyncParams& operation_attributes,
        const AllGatherAsyncInputs& tensor_args,
        Tensor& output_tensor,
        const std::optional<ttnn::MeshCoordinate>& mesh_coordinate = std::nullopt);
};

}  // namespace ttnn::experimental::prim
