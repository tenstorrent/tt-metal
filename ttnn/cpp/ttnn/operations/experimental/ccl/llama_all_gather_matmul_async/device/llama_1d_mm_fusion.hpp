// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.hpp"

#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::llama_matmul {

ttnn::prim::matmul_mcast_1d_common_override_variables_t matmul_multi_core_agmm_fusion_helper(
    tt::tt_metal::Program& program,
    const Tensor& a,
    const std::vector<Tensor>& b_tensors,
    const std::optional<const Tensor>& bias,
    const std::vector<Tensor>& output_tensors,
    bool broadcast_batch,
    DeviceComputeKernelConfig compute_kernel_config,
    const matmul::MatmulProgramConfig& program_config,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    uint32_t start_cb_index,
    std::optional<CoreRangeSet> restricted_cores);

// ProgramDescriptor-flavored variant of matmul_multi_core_agmm_fusion_helper.
//
// Kernels / CBs / semaphores are appended onto the caller-supplied
// ProgramDescriptor (same shape as matmul_multi_core_reuse_mcast_1d_optimized_helper_descriptor).
// Only the gather_in0 path of MatmulMultiCoreReuseMultiCast1DProgramConfig is supported:
// TT_FATAL fires if mcast_in0 is set or gather_in0 is false.
//
// This is a llama-specific fork of the gather_in0 builder: the in0 reader does a 4-step
// multicast (chunks indexed via the CCL fused-op ring) instead of the ring-hop topology used
// by the generic gather_in0 builder, and the worker grid comes from the in1 (weight) shard
// rather than from the in0 (aggregated activations) shard.
void matmul_multi_core_agmm_fusion_helper_descriptor(
    tt::tt_metal::ProgramDescriptor& desc,
    const Tensor& a,
    const std::vector<Tensor>& b_tensors,
    const std::optional<const Tensor>& bias,
    const std::vector<Tensor>& output_tensors,
    bool broadcast_batch,
    DeviceComputeKernelConfig compute_kernel_config,
    const matmul::MatmulProgramConfig& program_config,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    uint32_t start_cb_index,
    std::optional<CoreRangeSet> restricted_cores);

void override_agmm_fusion_program_parameters(
    const ttnn::prim::matmul_mcast_1d_common_override_variables_t& override_variables,
    const ttnn::prim::MatmulParams& operation,
    tt::tt_metal::Program& program,
    const std::vector<tt::tt_metal::Tensor>& input_tensors,
    const std::vector<std::optional<const tt::tt_metal::Tensor>>& optional_input_tensors,
    const std::vector<tt::tt_metal::Tensor>& output_tensors);
}  // namespace ttnn::operations::llama_matmul
