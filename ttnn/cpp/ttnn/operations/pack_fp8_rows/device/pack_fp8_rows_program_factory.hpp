// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/distributed/types.hpp"
#include "pack_fp8_rows_device_operation_types.hpp"

namespace ttnn::operations::pack_fp8_rows {

// Python plan(): rows = product of all dims but the last; cores = max(1, min(grid_x * grid_y,
// div_up(rows, ROWS_BLOCK))); cores are (i % grid_x, i / grid_x) for i < cores.
struct PackFp8RowsPlan {
    uint32_t rows = 0;
    uint32_t cores = 1;
    uint32_t grid_x = 1;
};
PackFp8RowsPlan plan_pack_fp8_rows(const PackFp8RowsParams& attrs, const Tensor& e4m3);

// One kernel (kernel_index::PACK), one staging CB (cb::STAGE), only common runtime args.
struct PackFp8RowsProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const PackFp8RowsParams& attrs, const PackFp8RowsInputs& t, Tensor& output);

    // Rewrites the three address common args (E4M3_ADDR, SCALE_ADDR, OUT_ADDR); the key pins the rest.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const PackFp8RowsParams& attrs,
        const PackFp8RowsInputs& t,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::operations::pack_fp8_rows
