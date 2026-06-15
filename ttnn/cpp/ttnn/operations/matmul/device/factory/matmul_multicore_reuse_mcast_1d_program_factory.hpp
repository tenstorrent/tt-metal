// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/matmul/device/matmul_1d_type.hpp"
#include <tt-metalium/program_descriptors.hpp>

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

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const shared_variables_t& shared_variables,
        const ttnn::prim::MatmulParams& operation_attributes,
        const ttnn::prim::MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);

    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ttnn::prim::MatmulParams& operation_attributes,
        const ttnn::prim::MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value,
        const std::optional<CoreRangeSet>& core_range_set = std::nullopt);
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

ttnn::device_operation::CachedProgram<MatmulMultiCoreReuseMcast1DProgramFactory::shared_variables_t>
matmul_multi_core_reuse_mcast_1d_optimized_helper(
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

// ProgramDescriptor-flavored variant of matmul_multi_core_reuse_mcast_1d_optimized_helper.
//
// Supports all three 1D paths:
//   * `mcast_in0`        — broadcast in0 across cores (single-B, single-output, c_0 base).
//   * `!mcast_in0 && !gather_in0` — broadcast in1; same single-B/output/c_0 constraints.
//   * `gather_in0`       — ring topology used by CCL+matmul fused ops
//     (llama_reduce_scatter_matmul, rs_matmul_op, all_gather_matmul_async). Supports
//     multi-B / multi-output, a non-zero `start_cb_index` (to leave low CB slots free for
//     the caller's CCL kernels), `restricted_cores`, and an optional GlobalCircularBuffer.
//
// The mcast (non-gather) paths TT_FATAL when callers pass gather_in0-only options
// (multi-B, multi-output, non-zero start_cb_index, restricted_cores, global_cb).
void matmul_multi_core_reuse_mcast_1d_optimized_helper_descriptor(
    tt::tt_metal::ProgramDescriptor& desc,
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
    // Default 0 mirrors tt::CBIndex::c_0; kept as a plain uint32_t to avoid pulling
    // <hostdevcommon/kernel_structs.h> into this widely-included public header. A
    // static_assert in the .cpp confirms the equivalence.
    uint32_t start_cb_index = 0,
    std::optional<CoreRangeSet> restricted_cores = std::nullopt);

namespace reuse_mcast_1d_optimized_helpers {
void override_program_parameters(
    const MatmulMultiCoreReuseMcast1DProgramFactory::shared_variables_t& override_variables,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    Program& program,
    const ttnn::prim::MatmulInputs& tensor_args,
    const std::vector<ttnn::Tensor>& tensor_return_value);
}  // namespace reuse_mcast_1d_optimized_helpers
}  // namespace ttnn::prim
