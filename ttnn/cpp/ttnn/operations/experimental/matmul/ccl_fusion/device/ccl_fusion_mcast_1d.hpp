// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ProgramDescriptor form of the 1D mcast_in0 matmul builder, for CCL-fused ops built as a ProgramDescriptor /
// WorkloadDescriptor (all_gather_matmul_async). matmul's own 1D factory is on Metal 2.0 (#56836); the fused ops keep
// building the matmul next to their CCL kernels and exchange fused-op signals through positional runtime args
// (ttnn/operations/ccl/ccl_op_fusion.hpp), which the Metal 2.0 path does not express yet.
//
// The builder is matmul's own create_program_mcast_in0_descriptor as of 917da3f9949^ (removed by #56836), which
// already pushed the fused-op runtime args but never initialised the signaler. Updated for what changed in matmul's
// Program& builder since (Quasar mcast-rectangle normalisation), pointed at the ccl_fusion kernel copies, with the
// mcast semaphore ids allocated in the caller's descriptor and the signaler initialised. mcast_in1 / gather_in0 are
// rejected: neither works fused today (#50167). Delete this file once the CCL ops are on Metal 2.0.

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

namespace ttnn::prim::ccl_fusion {

// Appends the 1D mcast_in0 matmul's kernels, CBs and semaphores to `desc`. Tensor addresses are buffer bindings.
void matmul_multi_core_reuse_mcast_1d_optimized_helper(
    tt::tt_metal::ProgramDescriptor& desc,
    const Tensor& a,
    const std::vector<Tensor>& b_tensors,
    const std::optional<const Tensor>& bias,
    const std::vector<Tensor>& output_tensors,
    bool broadcast_batch,
    DeviceComputeKernelConfig compute_kernel_config,
    const operations::matmul::MatmulProgramConfig& program_config,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler);

}  // namespace ttnn::prim::ccl_fusion
