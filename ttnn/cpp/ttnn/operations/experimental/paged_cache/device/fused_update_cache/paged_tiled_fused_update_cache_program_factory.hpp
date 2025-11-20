// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "paged_fused_update_cache_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <vector>
#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::experimental::paged_cache::program {

struct PagedTiledFusedUpdateCacheSharedVariables {
    tt::tt_metal::KernelHandle unary_reader_kernel_id = 0;
    tt::tt_metal::KernelHandle unary_writer_kernel_id = 0;
    std::vector<tt::tt_metal::CoreCoord> cores1;
    std::vector<tt::tt_metal::CoreCoord> cores2;
    uint32_t Wbytes = 0;
    uint32_t Wt = 0;
    tt::tt_metal::CBHandle cb_src1 = 0;
    tt::tt_metal::CBHandle cb_src3 = 0;
    tt::tt_metal::CBHandle cb_cur_pos_id = 0;
    tt::tt_metal::CBHandle cb_page_table_id = 0;
    uint32_t cache_batch_num_tiles = 0;
    bool use_index_tensor = false;
    bool is_paged_cache = false;
};

struct PagedTiledFusedUpdateCacheProgramFactory {
    using shared_variables_t = PagedTiledFusedUpdateCacheSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const fused_update_cache::operation_attributes_t& operation_attributes,
        const fused_update_cache::tensor_args_t& tensor_args,
        fused_update_cache::tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const fused_update_cache::operation_attributes_t& operation_attributes,
        const fused_update_cache::tensor_args_t& tensor_args,
        fused_update_cache::tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::paged_cache::program
