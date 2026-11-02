// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernels/topk_large_indices_compute_body_mode.hpp"
#include "topk_large_indices_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

#include <optional>
#include <vector>

namespace ttnn::operations::experimental::topk_large_indices::program {

// One core's work: num_rows rows, whole rows from start_row and, with has_split, last one half of split_row's
// K chunks. The two halves sit on a core pair; the first half receives the partner's survivor and writes the row.
struct CoreRowAssignment {
    CoreCoord core;
    uint32_t start_row{};
    uint32_t num_rows{};
    uint32_t partner_index{};
    uint32_t split_row{};
    bool has_split{};
    // Which half of split_row this core owns. Where the halves meet is the kernels' business, not the host's.
    bool second_half{};
};

// The canonical mapping used to populate reader/writer runtime arguments. Visible so a host-only test can pin
// its ordering across discontiguous CoreRangeSets.
// num_chunks counts the K chunks of the searched row prefix. row_split must be the compiled program's
// capability (see row_split_enabled): when it is false the caller reads only start_row and num_rows, so the
// halved layout must not be produced at all.
std::vector<CoreRowAssignment> derive_core_row_assignments(
    const CoreRangeSet& core_grid, uint32_t num_rows, uint32_t num_chunks, bool row_split);

struct TopkLargeIndicesSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle compute_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    CoreRangeSet core_grid;
    std::vector<CoreCoord> cores;
    ttnn::Shape input_shape;
    std::optional<uint32_t> valid_length;
    uint32_t llk_k = 0;
    bool row_split = false;
    uint32_t num_rows = 0;
    uint32_t num_chunks = 0;
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

ComputeBodyMode compute_body_mode(uint32_t k, uint32_t input_last_dim);

// Rows split across cores only when two cores per split row are free (all rows, or the rows left after whole
// rounds), so other programs carry no transfer code.
inline bool row_split_possible(uint32_t num_rows, uint32_t num_cores) {
    const uint32_t split_rows = num_rows % num_cores;
    return split_rows != 0 && 2ull * split_rows <= num_cores;
}

// Whether the input's rows span two or more K' chunks, a compile-time argument of the compute kernel.
bool multi_chunk_rows(uint32_t k, uint32_t input_last_dim);

// Whether the leftover rows are actually halved across core pairs: the geometry must allow it and the body must be
// able to reduce a chunk range that does not start at zero. The Classic body carries its own per-chunk index
// epilogue and has no such form.
bool row_split_enabled(uint32_t k, uint32_t num_rows, uint32_t input_last_dim, uint32_t num_cores);

// Whether a row spans more than one fused segment, so the K' 2048 split body has to round-trip its accumulator
// through L1.
bool long_rows(uint32_t k, uint32_t input_last_dim);

// Whether a core can hold a whole row of several chunks, which the compute kernel splits across MATH and PACK.
// A program whose rows all go to core pairs has no whole rows, so it builds no split code.
bool split_whole_rows(uint32_t k, uint32_t num_rows, uint32_t input_last_dim, uint32_t num_cores);

}  // namespace ttnn::operations::experimental::topk_large_indices::program
