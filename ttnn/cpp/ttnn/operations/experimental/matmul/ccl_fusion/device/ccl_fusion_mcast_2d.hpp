// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Pre-Metal 2.0 2D multicast matmul builder, in ProgramDescriptor form, used only by the CCL-fused ops
// all_gather_matmul_async and matmul_reduce_scatter_async, which build the matmul and the CCL kernels into one
// program per device and exchange fused-op signals through positional runtime args
// (ttnn/operations/ccl/ccl_op_fusion.hpp). Delete this file once those CCL ops are on Metal 2.0 and use matmul's
// spec-based builder instead.
//
// It binds matmul's legacy kernels in ttnn/operations/matmul/device/kernels/ (not their _metal2 forks),
// which matmul's legacy 1D builder binds too. Copy them here before matmul deletes them.

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim::ccl_fusion {

// Appends the matmul's kernels, CBs and semaphores to `desc`. Tensor addresses are recorded as buffer bindings, so a
// WorkloadDescriptor op built from it needs no matmul-specific cache-hit refresh.
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
