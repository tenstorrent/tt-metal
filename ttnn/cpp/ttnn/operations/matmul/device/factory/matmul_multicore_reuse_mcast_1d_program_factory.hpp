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

    // This method is the cache-hit hook for the MeshWorkload sibling factory below and for
    // all_gather_matmul_async, which call it directly.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const shared_variables_t& shared_variables,
        const ttnn::prim::MatmulParams& operation_attributes,
        const ttnn::prim::MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);

    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
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

// Mask-mode sparsity for the mcast_in0 body, as the sparse matmul drives it: batch bB of the weight's
// batchB groups is computed only where the one-page `mask` holds a non-zero entry, which is the in0
// sender's and the in1 sender/writer's SPARSITY path. Absent, the body builds the dense program
// unchanged.
struct McastIn0Sparsity {
    const tt::tt_metal::MeshTensor& mask;
    // Groups (experts) the mask covers per outer batch.
    uint32_t batchB = 0;
    // Batches the in0 receivers and compute loop over: nnz when the caller supplied it, else batchB.
    uint32_t num_batch_compute = 0;
    // True when nnz is unknown, so the in0 sender broadcasts each batch's validity to the receivers
    // and compute instead of them looping num_batch_compute times.
    bool get_batch_from_reader = false;
    // False when in0 holds one [M, K] slice per group (is_input_a_sparse) rather than one for all.
    bool bcast_A = true;
    // Output holds only the active groups' results, packed in mask order.
    bool compact_output = false;
};

namespace reuse_mcast_1d_optimized_helpers {
// The mcast_in0 body over prefetcher_pipes with mask-mode sparsity: the sparse matmul's spec factory
// translates its parameters into these and builds the same program the dense matmul over pipes does,
// plus the SPARSITY paths. in0 and output must be interleaved; `batchA` is the outer batch count.
ttnn::device_operation::ProgramArtifacts create_sparse_mcast_in0_artifacts(
    const Tensor& a,
    const Tensor& b,
    const Tensor& output,
    operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig program_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    uint32_t batchA,
    const ttnn::PrefetcherPipeList& prefetcher_pipes,
    const McastIn0Sparsity& sparsity);

void override_program_parameters(
    const MatmulMultiCoreReuseMcast1DProgramFactory::shared_variables_t& override_variables,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    Program& program,
    const ttnn::prim::MatmulInputs& tensor_args,
    const std::vector<ttnn::Tensor>& tensor_return_value);
}  // namespace reuse_mcast_1d_optimized_helpers
}  // namespace ttnn::prim
