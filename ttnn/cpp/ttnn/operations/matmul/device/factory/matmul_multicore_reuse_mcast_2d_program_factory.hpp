// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

namespace ttnn::prim {

struct MatmulMultiCoreReuseMcast2DProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle mm_kernel_in0_sender_id{};
        std::vector<CoreCoord> in0_sender_interleaved_cores;
        tt::tt_metal::KernelHandle mm_kernel_in1_sender_writer_id{};
        std::vector<CoreCoord> in1_sender_cores;
        tt::tt_metal::KernelHandle mm_kernel_in1_receiver_writer_id{};
        std::vector<CoreCoord> in1_receiver_cores;
        tt::tt_metal::KernelHandle mm_kernel_in1_receiver_writer_other_noc_setup_id{};
        std::vector<CoreCoord> in1_receiver_other_cores;
        tt::tt_metal::CBHandle cb_src2{};
        tt::tt_metal::CBHandle cb_output{};
        uint32_t num_cores_with_work_r{};
        uint32_t num_cores_with_work_c{};
        uint32_t start_core_x{};
        uint32_t start_core_y{};
        bool transpose_mcast{};
        std::vector<CoreCoord> cores;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const ttnn::prim::MatmulParams& operation_attributes,
        const ttnn::prim::MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const ttnn::prim::MatmulParams& operation_attributes,
        const ttnn::prim::MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const shared_variables_t& shared_variables,
        const ttnn::prim::MatmulParams& operation_attributes,
        const ttnn::prim::MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);
};

struct MatmulMeshWorkloadMultiCoreReuseMcast2DProgramFactory {
    using shared_variables_t = MatmulMultiCoreReuseMcast2DProgramFactory::shared_variables_t;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const ttnn::prim::MatmulParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const ttnn::prim::MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const ttnn::prim::MatmulParams& operation_attributes,
        const ttnn::prim::MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);
};

MatmulMultiCoreReuseMcast2DProgramFactory::cached_program_t matmul_multi_core_reuse_mcast_2d_optimized_helper(
    tt::tt_metal::Program& program, /* Take programa as input by reference */
    const Tensor& a,
    const Tensor& b,
    const std::optional<const Tensor>& bias,
    Tensor& output_tensor,
    bool broadcast_batch,
    DeviceComputeKernelConfig compute_kernel_config,
    const operations::matmul::MatmulProgramConfig& program_config,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler);

}  // namespace ttnn::prim
