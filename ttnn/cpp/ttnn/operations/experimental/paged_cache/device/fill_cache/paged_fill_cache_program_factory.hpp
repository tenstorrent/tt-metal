// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "paged_fill_cache_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::paged_cache::program {

struct PagedFillCacheSharedVariables {
    tt::tt_metal::KernelHandle unary_reader_kernel_id = 0;
    tt::tt_metal::KernelHandle unary_writer_kernel_id = 0;
    std::vector<tt::tt_metal::CoreCoord> cores;
    uint32_t g1_numcores = 0;
    uint32_t g2_numcores = 0;
    uint32_t num_blocks_per_core_group_1 = 0;
    uint32_t num_blocks_per_core_group_2 = 0;
    uint32_t Wt = 0;
    bool use_batch_idx_tensor = false;
};

struct PagedFillCacheProgramFactory {
    using shared_variables_t = PagedFillCacheSharedVariables;
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

}  // namespace ttnn::operations::experimental::paged_cache::program
