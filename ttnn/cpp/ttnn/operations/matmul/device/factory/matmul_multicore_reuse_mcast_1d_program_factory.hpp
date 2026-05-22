// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/matmul/device/matmul_1d_type.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include <hostdevcommon/kernel_structs.h>

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
// Mirrors the legacy helper's argument list. Currently supports the `mcast_in0` and
// `!mcast_in0 && !gather_in0` paths (where `start_cb_index` must be tt::CBIndex::c_0 and
// `restricted_cores`/`global_cb`/multi-`b_tensors` must be at their default/single-element
// values, matching what the existing descriptor builders accept).
//
// The `gather_in0` path is the actually-used path for the current CCL fused matmul callers
// (llama_reduce_scatter_matmul, rs_matmul_op, all_gather_matmul_async) but does not yet
// have a descriptor builder; this variant TT_FATALs on it until that work lands. Tracked as
// follow-up to issue #42193 alongside the 1d-mcast helper migration.
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
    uint32_t start_cb_index = tt::CBIndex::c_0,
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
