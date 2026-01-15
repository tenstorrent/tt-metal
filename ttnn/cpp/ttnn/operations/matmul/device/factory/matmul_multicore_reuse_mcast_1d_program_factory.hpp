// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/matmul/device/matmul_1d_type.hpp"

namespace ttnn::prim {

struct matmul_mcast_1d_common_override_variables_t {
    std::vector<tt::tt_metal::KernelHandle> kernels;
    std::vector<tt::tt_metal::CBHandle> cbs;
    bool extract_shard_sub_blocks{};
    CoreCoord start_core;
    std::vector<CoreCoord> cores;
    uint32_t num_cores_with_work{};
    ttnn::prim::Matmul1DType type{};
};

struct MatmulMultiCoreReuseMcast1DProgramFactory {
    using shared_variables_t = matmul_mcast_1d_common_override_variables_t;

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

struct MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory {
    using shared_variables_t = MatmulMultiCoreReuseMcast1DProgramFactory::shared_variables_t;
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

MatmulMultiCoreReuseMcast1DProgramFactory::shared_variables_t matmul_multi_core_reuse_mcast_1d_optimized_helper(
    tt::tt_metal::Program& program,
    const Tensor& a,
    const std::vector<Tensor>& b_tensors,
    const std::optional<const Tensor>& bias,
    const std::vector<Tensor>& output_tensors,
    bool broadcast_batch,
    DeviceComputeKernelConfig compute_kernel_config,
    const operations::matmul::MatmulProgramConfig& program_config,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    uint32_t start_cb_index,
    std::optional<CoreRangeSet> restricted_cores);

MatmulMultiCoreReuseMcast1DProgramFactory::cached_program_t matmul_multi_core_reuse_mcast_1d_optimized_helper(
    tt::tt_metal::Program& program,
    const Tensor& a,
    const std::vector<Tensor>& b_tensors,
    const std::optional<const Tensor>& bias,
    const std::vector<Tensor>& output_tensors,
    bool broadcast_batch,
    DeviceComputeKernelConfig compute_kernel_config,
    const operations::matmul::MatmulProgramConfig& program_config,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id);

namespace reuse_mcast_1d_optimized_helpers {
void override_program_parameters(
    const MatmulMultiCoreReuseMcast1DProgramFactory::shared_variables_t& override_variables,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    Program& program,
    const ttnn::prim::MatmulInputs& tensor_args,
    const std::vector<ttnn::Tensor>& tensor_return_value);
}  // namespace reuse_mcast_1d_optimized_helpers
}  // namespace ttnn::prim
