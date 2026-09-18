// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include <tt-metalium/program_descriptors.hpp>

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

    // Not the cache-hit hook for this factory: it is void-returning, so the factory satisfies
    // ProgramSpecFactoryConcept and the framework refreshes the tensor bindings itself. The
    // ported-from override wrote nothing but addresses -- the in0 sender's address slot (or, when in0
    // is sharded, the backing address of its borrowed buffer), the in1 sender's in1 / output / bias
    // slots, and both receiver groups' output slots -- all of which are bindings now, and a borrowed
    // buffer draws its backing address from the same tensor argument. So there is no non-tensor
    // refresh to re-apply, and CustomProgramSpecFactoryConcept would buy nothing.
    //
    // This method survives with its pre-Metal-2.0 name and signature because the CCL fused ops
    // (all_gather_matmul_async, matmul_reduce_scatter_async) call it directly, supplying the
    // shared_variables_t that create_program_artifacts does not produce.
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

ttnn::device_operation::CachedProgram<MatmulMultiCoreReuseMcast2DProgramFactory::shared_variables_t>
matmul_multi_core_reuse_mcast_2d_optimized_helper(
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
