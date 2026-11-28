// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/distributed/types.hpp"
#include "unpack_fp8_rows_device_operation_types.hpp"

namespace ttnn::operations::unpack_fp8_rows {

// One program for every call of a (rows spec, out spec, grid, ncores, common_values): cores y * grid_x + x <
// ncores, each taking its share of the 32-row bands by its index. create_descriptor runs on a miss;
// override_runtime_arguments on a hit writes the two addresses and the per-call values (src_page0, out_page0,
// nbands): per core on every core, or once in the common args when common_values.
struct UnpackFp8RowsProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const UnpackFp8RowsParams& attrs, const UnpackFp8RowsInputs& tensors, Tensor& output);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const UnpackFp8RowsParams& attrs,
        const UnpackFp8RowsInputs& tensors,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::operations::unpack_fp8_rows
