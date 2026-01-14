// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "paged_update_cache_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <vector>

namespace ttnn::operations::experimental::paged_cache::update::program {

struct PagedUpdateCacheSharedVariables {
    tt::tt_metal::KernelHandle unary_reader_kernel_id = 0;
    tt::tt_metal::KernelHandle unary_writer_kernel_id = 0;
    std::vector<tt::tt_metal::CoreCoord> cores;
    uint32_t Wbytes = 0;
    uint32_t Wt = 0;
    tt::tt_metal::CBHandle cb_src1 = 0;
    uint32_t cache_batch_num_tiles = 0;
    bool use_index_tensor = false;
    bool is_paged_cache = false;
};

struct PagedUpdateCacheProgramFactory {
    using shared_variables_t = PagedUpdateCacheSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        Tensor& tensor_return_value);
};

struct PagedUpdateCacheMeshWorkloadFactory {
    using shared_variables_t = PagedUpdateCacheSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const operation_attributes_t& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const tensor_args_t& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::paged_cache::update::program
