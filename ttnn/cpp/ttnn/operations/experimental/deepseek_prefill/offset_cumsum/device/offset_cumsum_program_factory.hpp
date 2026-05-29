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

    // Per-coord program build.  `mesh_dispatch_coordinate` is required: this op
    // bakes a per-device `row_idx` (derived from cluster_axis) into the reader's
    // runtime args, so each device gets a different program.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const OffsetCumsumParams& operation_attributes,
        const Tensor& input,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::experimental::prim
