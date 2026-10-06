// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::llama_matmul {

// Appends the fused all-gather + gather_in0 ring matmul's kernels, CBs and semaphores to `desc`. Tensor addresses are
// buffer bindings and the tensor-backed CBs carry their buffers, so a WorkloadDescriptor op built from it needs no
// matmul-specific cache-hit refresh.
void matmul_multi_core_agmm_fusion_helper(
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
}  // namespace ttnn::operations::llama_matmul
