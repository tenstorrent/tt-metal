// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernels/topk_large_indices_compute_body_mode.hpp"
#include "topk_large_indices_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

#include <vector>

namespace ttnn::operations::experimental::topk_large_indices::program {

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

ComputeBodyMode compute_body_mode(uint32_t k, uint32_t input_last_dim);

}  // namespace ttnn::operations::experimental::topk_large_indices::program
