// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>

#include "offset_cumsum_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

struct OffsetCumsumProgramFactory {
    using tensor_return_value_t = std::array<Tensor, 3>;

    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const OffsetCumsumParams& operation_attributes,
        const Tensor& input,
        tensor_return_value_t& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    // Only row_idx varies with the mesh coordinate. Chips at the same position
    // along cluster_axis share a program while reading their own local histograms.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const OffsetCumsumParams& operation_attributes,
        const Tensor& input,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::experimental::prim
