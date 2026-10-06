// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ProgramDescriptor form of the 2D multicast matmul builder, used only by the CCL-fused ops
// all_gather_matmul_async and matmul_reduce_scatter_async, which build the matmul next to their CCL
// kernels in one per-device program and exchange fused-op signals through positional runtime args
// (ttnn/operations/ccl/ccl_op_fusion.hpp), which matmul's Metal 2.0 path does not express yet.
//
// Kept out of ttnn/operations/matmul so that matmul itself carries no pre-Metal 2.0 builder. Delete
// this file once those CCL ops are on Metal 2.0 and use matmul's spec-based builder instead.

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim::ccl_fusion {

// Appends the matmul's kernels, CBs and semaphores to `desc`. Tensor
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
