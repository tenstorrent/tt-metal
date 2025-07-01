// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ring_fusion.hpp"

namespace ttnn::operations::transformer::detail {

tt::tt_metal::operation::ProgramWithCallbacks ring_joint_sdpa(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor_q,
    const Tensor& input_tensor_k,
    const Tensor& input_tensor_v,
    const Tensor& joint_tensor_q,
    const Tensor& joint_tensor_k,
    const Tensor& joint_tensor_v,
    const Tensor& output_tensor,
    const Tensor& joint_output_tensor,
    const Tensor& lse_output_tensor,
    std::size_t logical_n,
    std::optional<float> scale,
    std::size_t q_chunk_size,
    std::size_t k_chunk_size,
    std::size_t ring_size,
    DeviceComputeKernelConfig compute_kernel_config,
    std::optional<SDPAProgramConfig> program_config,
    std::optional<RingSDPAFusedOpSignaler>& sdpa_fused_op_signaler);

}  // namespace ttnn::operations::transformer::detail
