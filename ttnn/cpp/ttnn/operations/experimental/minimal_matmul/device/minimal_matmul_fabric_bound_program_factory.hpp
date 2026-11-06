// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>

#include "minimal_matmul_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

namespace ttnn::experimental::prim {

namespace minimal_matmul_fabric_bound_layout {
// Number of kernels minimal_matmul_fabric_bound_factory_helper_common appends; a parent op that pushes its own
// kernels after the matmul's starts them at the matmul's first kernel index plus this.
inline constexpr uint32_t kNumKernels = 5;
}  // namespace minimal_matmul_fabric_bound_layout

// Appends the fabric-bound matmul kernels, CBs and semaphores to `desc` (exposed for the fused all-gather matmul).
// The first appended kernel lands at desc.kernels.size() on entry.
void minimal_matmul_fabric_bound_factory_helper_common(
    tt::tt_metal::ProgramDescriptor& desc,
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const Tensor>& bias_tensor,
    const std::optional<operations::unary::UnaryWithParam>& fused_activation,
    const std::optional<const MinimalMatmulConfig>& config,
    const std::vector<Tensor>& output_tensors,
    const DeviceComputeKernelConfig& compute_kernel_config,
    std::optional<ttnn::experimental::ccl::MinimalMatmulFusedOpSignaler>& fused_op_signaler,
    uint32_t N_chunks,
    std::optional<float> fused_ternary_scalar = std::nullopt,
    const std::optional<const Tensor>& fused_ternary_input_a = std::nullopt,
    const std::optional<const Tensor>& fused_ternary_input_b = std::nullopt,
    std::optional<ttnn::experimental::ccl::StridedReduceScatterFusedOpSignaler> srs_fused_op_signaler = std::nullopt,
    bool fuse_swiglu = false);

// Cache-hit patch for a program built by minimal_matmul_fabric_bound_factory_helper_common starting at
// `first_kernel_index`. Rewrites every tensor address by role; `ag_input_tensor` is the all-gather input read by
// the in0 injectors (empty unless the local slice is read from the input), and `output_tensors` must be the
// tensors the program was built with, in the same order.
void minimal_matmul_fabric_bound_patch_runtime_args(
    tt::tt_metal::Program& program,
    uint32_t first_kernel_index,
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const Tensor>& bias_tensor,
    const std::optional<const Tensor>& ag_input_tensor,
    const std::optional<const Tensor>& fused_ternary_input_a,
    const std::optional<const Tensor>& fused_ternary_input_b,
    const std::vector<Tensor>& output_tensors);

}  // namespace ttnn::experimental::prim
