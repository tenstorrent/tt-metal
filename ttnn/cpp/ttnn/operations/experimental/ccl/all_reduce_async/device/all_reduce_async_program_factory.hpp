// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "all_reduce_async_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct AllReduceAsyncSharedVariables {
    tt::tt_metal::KernelHandle worker_sender_reader_kernel_id{};
    tt::tt_metal::KernelHandle worker_sender_writer_kernel_id{};
    tt::tt_metal::KernelHandle reduction_reader_kernel_id{};
    std::vector<tt::tt_metal::CoreCoord> sender_worker_cores;
    CoreRangeSet output_tensor_cores;
    tt::tt_metal::CBHandle cb_out{};
    tt::tt_metal::CBHandle cb_reduction{};
};

struct AllReduceAsyncMeshWorkloadFactory {
    using shared_variables_t = AllReduceAsyncSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const AllReduceAsyncParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const AllReduceAsyncInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const AllReduceAsyncParams& operation_attributes,
        const AllReduceAsyncInputs& tensor_args,
        Tensor& output_tensor);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const AllReduceAsyncParams& operation_attributes,
        const ttnn::MeshCoordinate& coord,
        const AllReduceAsyncInputs& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::experimental::prim

namespace ttnn {

std::tuple<CoreRangeSet, std::vector<CoreCoord>> ar_choose_worker_cores(
    size_t num_links, size_t num_workers_per_link, const CoreRangeSet& available_cores);

}  // namespace ttnn
