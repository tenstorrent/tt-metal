// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "topk_large_indices_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

#include <vector>

namespace ttnn::operations::experimental::topk_large_indices::program {

// The resolved worker core set as one dense rectangle, when it is one. The
// multi-engine paths (column-parallel trees, hybrid row split) require a
// rectangle: the cost model sees its dimensions and every placement is
// translated by its origin, so it may sit anywhere in the device grid. A
// non-rectangular resolved set returns nullopt and the op runs the
// row-parallel engine over the enumerated cores.
std::optional<CoreRange> topk_li_worker_rect(const CoreRangeSet& resolved_worker_core_grid);

struct CoreRowAssignment {
    CoreCoord core{};
    uint32_t start_row{};
    uint32_t num_rows{};
};

// This is the canonical mapping used to populate reader/writer runtime arguments. Keeping it visible
// allows a host-only unit test to pin ordering across discontiguous CoreRangeSets.
std::vector<CoreRowAssignment> derive_core_row_assignments(const CoreRangeSet& core_grid, uint32_t num_rows);

struct TopkLargeIndicesSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle compute_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    CoreRangeSet core_grid{};
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
    // Number of row slices == number of tree cores PER RECTANGLE (each reduces >= 1 chunk).
    uint32_t num_slices = 0;
    // Each tree is a local_grid_x x local_grid_y rectangle with
    // num_slices == local_grid_x * local_grid_y. Within a rectangle, slice i
    // lives at rectangle-local (i % local_grid_x, i / local_grid_x); slice 0
    // is that rectangle's merge-tree root (the reduction is in place: at tree
    // level L, core i with i % 2^(L+1) == 0 merges core i + 2^L's survivor).
    uint32_t local_grid_x = 0;
    uint32_t local_grid_y = 0;
    // Multi-rectangle (rows > 1 with an explicit num_slices): the grid is
    // tiled with num_rects disjoint rectangles running concurrently, rows
    // split contiguously across them (runtime args — one cached program
    // serves any row count with the same rectangle layout). 1 on the
    // single-row model-selected path; 0 when disabled.
    uint32_t num_rects = 0;
};

// num_slices_override: internal explicit P (operation_attributes_t::num_slices). Validated loudly
// against [2, 128] and the row's chunk count, clamped only against the physical grid; setting it
// when the column-parallel path is not selected is an error.
// FUSED_E2E compute-kernel gate: true when every possible chunk count of the
// row fits the 5-bit chunk-id stamp (<= 32 chunks of the LLK window over the
// full logical last dim). Shared by the row-parallel factory (kernel define)
// and compute_program_hash (the derived bit) so they can never skew.
bool fused_e2e_gate(uint32_t k, uint32_t input_last_dim);

// Row-parallel compute-body selection, single-sourced into the kernel defines
// AND compute_program_hash so they can never skew:
//   FusedSegmented -- k >= 1024: ONE codepath at every width (a <= 32-chunk
//     row runs the single-segment path, which is the FUSED_E2E sequence), so
//     the width term leaves program selection entirely for this class.
//   FusedE2E -- smaller k with <= 32 chunks (fused end-to-end).
//   Classic -- smaller k, wider rows (keeps the chunk-skip, which pays there).
enum class ComputeBodyMode : uint32_t { Classic = 0, FusedE2E = 1, FusedSegmented = 2 };
ComputeBodyMode compute_body_mode(uint32_t k, uint32_t input_last_dim);

ColumnSplitConfig compute_column_split_config(
    uint32_t k,
    uint32_t n,
    uint32_t num_rows,
    const CoreCoord& grid,
    std::optional<uint32_t> num_slices_override = std::nullopt,
    bool allow_multi_row = true);

struct TopkLargeIndicesMultiCoreSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle compute_node_kernel_id{};
    tt::tt_metal::KernelHandle compute_root_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    CoreRangeSet core_grid{};
    // Per-rectangle slice-major (row-major within the rectangle) core lists;
    // rect_cores[r][0] is rectangle r's tree root. Single-rect programs have
    // exactly one entry.
    std::vector<std::vector<CoreCoord>> rect_cores{};
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
