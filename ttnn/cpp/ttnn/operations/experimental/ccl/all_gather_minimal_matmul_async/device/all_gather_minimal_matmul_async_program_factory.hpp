// SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

#include <tt-metalium/program.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include <optional>
#include <vector>

namespace ttnn::experimental::prim {

struct AllGatherMinimalMatmulAsyncProgramFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const AllGatherMinimalMatmulAsyncParams& operation_attributes,
        const AllGatherMinimalMatmulAsyncInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const AllGatherMinimalMatmulAsyncParams& operation_attributes,
        const AllGatherMinimalMatmulAsyncInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value,
        std::optional<ttnn::MeshCoordinate> mesh_coordinate = std::nullopt);
};

}  // namespace ttnn::experimental::prim
