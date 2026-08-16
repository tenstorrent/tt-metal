// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "topk_large_indices_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

#include <vector>

namespace ttnn::operations::experimental::topk_large_indices::program {

struct TopkLargeIndicesSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle compute_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    std::vector<CoreCoord> cores{};
};

struct TopkLargeIndicesProgramFactory {
    using shared_variables_t = TopkLargeIndicesSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

// Column-parallel (intra-row multi-core) configuration. Derived purely from
// (k, physical last dim, num_rows, worker grid) so that:
//   - select_program_factory / compute_program_hash / create all agree, and
//   - valid_length stays runtime-only: it shrinks per-core active chunk
//     counts (down to empty slices) without changing the program structure,
//     so a serving loop growing valid_length reuses one cached program.
struct ColumnSplitConfig {
    bool enabled = false;
    // Number of row slices == number of local cores (each reduces >= 1 chunk).
    uint32_t num_slices = 0;
    // Local cores form the rectangle (0, 0)..(local_grid_x-1, local_grid_y-1)
    // with num_slices == local_grid_x * local_grid_y; the final (merge) core
    // sits at (0, local_grid_y).
    uint32_t local_grid_x = 0;
    uint32_t local_grid_y = 0;
};

ColumnSplitConfig compute_column_split_config(uint32_t k, uint32_t n, uint32_t num_rows, const CoreCoord& grid);

struct TopkLargeIndicesMultiCoreSharedVariables {
    tt::tt_metal::KernelHandle reader_local_kernel_id{};
    tt::tt_metal::KernelHandle compute_local_kernel_id{};
    tt::tt_metal::KernelHandle writer_local_kernel_id{};
    tt::tt_metal::KernelHandle reader_final_kernel_id{};
    tt::tt_metal::KernelHandle compute_final_kernel_id{};
    tt::tt_metal::KernelHandle writer_final_kernel_id{};
    std::vector<CoreCoord> local_cores{};
    CoreCoord final_core{};
};

struct TopkLargeIndicesMultiCoreProgramFactory {
    using shared_variables_t = TopkLargeIndicesMultiCoreSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::topk_large_indices::program
