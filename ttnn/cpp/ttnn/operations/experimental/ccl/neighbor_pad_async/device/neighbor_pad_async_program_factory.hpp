// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "neighbor_pad_async_device_operation_types.hpp"
#include "ttnn/distributed/types.hpp"

#include <tt-metalium/program.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include <optional>
#include <variant>

namespace ttnn::experimental::prim {

// Whether one program serves every size of the padded dims: runtime_shapes is set, the output is interleaved, and
// the fabric cores split the dims before the padded one, which the program hash keeps, rather than the sticks of a
// padded dim 0.
bool neighbor_pad_async_is_shape_free(const NeighborPadAsyncParams& args, const NeighborPadAsyncInputs& tensor_args);

struct NeighborPadAsyncMeshWorkloadFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const NeighborPadAsyncParams& operation_attributes,
        const NeighborPadAsyncInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const NeighborPadAsyncParams& operation_attributes,
        const NeighborPadAsyncInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& coord = std::nullopt);
};

}  // namespace ttnn::experimental::prim
