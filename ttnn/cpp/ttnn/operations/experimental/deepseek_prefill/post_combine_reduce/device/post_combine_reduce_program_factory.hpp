// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "post_combine_reduce_types.hpp"

#include "ttnn/device_operation.hpp"
#include <tt-metalium/mesh_coord.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::post_combine_reduce {

struct PostCombineReduceProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id;
        tt::tt_metal::KernelHandle compute_kernel_id;
        tt::tt_metal::KernelHandle writer_kernel_id;
        tt::tt_metal::CBHandle output_cb_handle;
        std::vector<tt::tt_metal::CoreCoord> cores;
        // Tracks whether this program was built with the DeepSeek (dispatch
        // table) skip path so override_runtime_arguments can thread the
        // dispatch_table / indices addresses only when they exist.
        bool use_dispatch_table_skip;
    };

    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const PostCombineReduceParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const PostCombineReduceInputs& tensor_args,
        ttnn::Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const PostCombineReduceParams& operation_attributes,
        const PostCombineReduceInputs& tensor_args,
        ttnn::Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::post_combine_reduce
