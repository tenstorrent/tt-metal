// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <optional>

#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/distributed/types.hpp"
#include "ckv_bfp4_rows_device_operation_types.hpp"

namespace ttnn::operations::ckv_bfp4_rows {

// One program on every core of the grid_x x grid_y rectangle (cores past the band count idle); the same program
// on every chip of the mesh. create_descriptor plans on a miss; override_runtime_arguments writes the two buffer
// addresses and the three per-call values (src_row0, num_bands, row0) into the common args on every hit.
struct CkvBfp4RowsProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const CkvBfp4RowsParams& attrs, const CkvBfp4RowsInputs& tensor_args, Tensor& output);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const CkvBfp4RowsParams& attrs,
        const CkvBfp4RowsInputs& tensor_args,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

// Programs created so far (cache misses across the process), read by the package's _ROWS_PREPARED view.
std::size_t programs_built();

}  // namespace ttnn::operations::ckv_bfp4_rows
