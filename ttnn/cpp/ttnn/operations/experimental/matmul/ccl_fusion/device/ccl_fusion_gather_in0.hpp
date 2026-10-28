// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ProgramDescriptor form of matmul's gather_in0 (ring) 1D builder, for llama_reduce_scatter_matmul, which builds the
// matmul next to its reduce-scatter kernels in one per-device program and signals the reduce-scatter through the
// llama MatmulFusedOpSignaler (positional runtime args, ttnn/operations/ccl/ccl_op_fusion.hpp). in1 may be fed by a
// GlobalCircularBuffer (the prefetcher), which a ProgramDescriptor can attach but Metal 2.0 cannot yet.
//
// Translated from matmul's legacy process_gather_in0_program_and_create_override_variables (still used by matmul's
// own MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory): same CBs (the GCB-backed remote CB included), kernels,
// compile-time and runtime args; tensor-backed CBs carry their tensors and the in1 address is a buffer binding, so a
// WorkloadDescriptor op built from it needs no matmul-specific cache-hit refresh. matmul's Metal 2.0 gather_in0
// (create_program_gather_in0_artifacts) takes in1 from PrefetcherPipes, not from a GlobalCircularBuffer, so it cannot
// serve this op yet. Delete this file once llama_reduce_scatter_matmul is on Metal 2.0.
//
// It binds matmul's legacy gather kernels in ttnn/operations/matmul/device/kernels/ (not their _metal2 forks), which
// matmul's legacy 1D builder binds too. Copy them here before matmul deletes them.

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/global_circular_buffer.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

namespace ttnn::prim::ccl_fusion {

// Appends the gather_in0 ring matmul's kernels, CBs and semaphores to `desc` (CB indices from start_cb_index;
// restricted_cores excluded from the matmul placement). Tensor addresses are buffer bindings.
void matmul_multi_core_reuse_mcast_1d_gather_in0_helper(
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
    // Kept by the descriptor until a Program is built from it: pass the op attributes' object, not a temporary.
    const tt::tt_metal::experimental::GlobalCircularBuffer* global_cb,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    uint32_t start_cb_index,
    std::optional<CoreRangeSet> restricted_cores);

}  // namespace ttnn::prim::ccl_fusion
