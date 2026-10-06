// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Legacy (pre-Metal 2.0) 2D multicast matmul builder, used only by the CCL-fused ops
// all_gather_matmul_async and matmul_reduce_scatter_async, which build the matmul and the CCL
// kernels into one tt::tt_metal::Program and exchange fused-op signals through positional
// runtime args (ttnn/operations/ccl/ccl_op_fusion.hpp).
//
// Moved verbatim out of ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp
// when that factory was ported to Metal 2.0, so that matmul itself carries no legacy builder. Delete
// this file once those CCL ops are on Metal 2.0 and use matmul's spec-based builder instead.

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim::ccl_fusion {

struct Mcast2DSharedVariables {
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

// Cache-hit refresh of the tensor addresses written by matmul_multi_core_reuse_mcast_2d_optimized_helper.
void override_mcast_2d_runtime_arguments(
    tt::tt_metal::Program& program,
    const Mcast2DSharedVariables& shared_variables,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value);

ttnn::device_operation::CachedProgram<Mcast2DSharedVariables> matmul_multi_core_reuse_mcast_2d_optimized_helper(
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

// ProgramDescriptor form of the helper above: appends the matmul's kernels, CBs and semaphores to `desc`. Tensor
// addresses are recorded as buffer bindings, so a WorkloadDescriptor op built from it needs no matmul-specific
// cache-hit refresh.
void matmul_multi_core_reuse_mcast_2d_optimized_helper(
    tt::tt_metal::ProgramDescriptor& desc,
    const Tensor& a,
    const Tensor& b,
    const std::optional<const Tensor>& bias,
    Tensor& output_tensor,
    bool broadcast_batch,
    DeviceComputeKernelConfig compute_kernel_config,
    const operations::matmul::MatmulProgramConfig& program_config,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler);

}  // namespace ttnn::prim::ccl_fusion
