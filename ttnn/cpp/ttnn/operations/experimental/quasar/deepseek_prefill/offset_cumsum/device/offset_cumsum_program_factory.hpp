// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "offset_cumsum_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/mesh_coord.hpp>
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim::qsr {

struct OffsetCumsumProgramFactory {
    using tensor_return_value_t = std::array<Tensor, 4>;

    // Only row_idx varies with the mesh coordinate. Chips at the same position
    // along cluster_axis share a program while reading their own local histograms.
    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const OffsetCumsumParams& operation_attributes,
        const Tensor& input,
        tensor_return_value_t& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::experimental::prim::qsr
