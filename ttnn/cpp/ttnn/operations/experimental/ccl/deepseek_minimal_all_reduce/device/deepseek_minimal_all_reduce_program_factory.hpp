// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/operations/experimental/ccl/deepseek_minimal_all_reduce/device/deepseek_minimal_all_reduce_device_operation_types.hpp"

namespace ttnn::operations::experimental::ccl::deepseek_minimal_all_reduce::program {
struct DeepseekMinimalAllReduceProgramFactory {
    struct shared_variables_t {
        std::vector<tt::tt_metal::CoreCoord> sender_worker_cores;
        std::vector<tt::tt_metal::CoreCoord> receiver_worker_cores;
        tt::tt_metal::KernelHandle worker_sender_reader_kernel_id{};
        tt::tt_metal::KernelHandle worker_sender_writer_kernel_id{};
        tt::tt_metal::KernelHandle worker_receiver_reader_kernel_id{};
        tt::tt_metal::CBHandle compute_cb_in1_handle{};
        tt::tt_metal::CBHandle compute_cb_in2_handle{};
        tt::tt_metal::CBHandle compute_cb_out_handle{};
        tt::tt_metal::CBHandle compute_cb_residual_handle{};
        bool has_residual = false;
        tt::tt_metal::GlobalSemaphore semaphore1;
        tt::tt_metal::GlobalSemaphore semaphore2;
        uint32_t ring_index = 0;
    };

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

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
        const operation_attributes_t& operation_attributes,
        const ttnn::MeshCoordinate& coord,
        const tensor_args_t& tensor_args,
        Tensor& output_tensor,
        const Tensor& intermediate_tensor,
        const tt::tt_metal::GlobalSemaphore& semaphore1,
        const tt::tt_metal::GlobalSemaphore& semaphore2);
};

}  // namespace ttnn::operations::experimental::ccl::deepseek_minimal_all_reduce::program
