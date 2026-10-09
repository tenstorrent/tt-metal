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

// Kernel push order and runtime-arg layout of minimal_matmul_fabric_bound_factory_helper_common.
// minimal_matmul_fabric_bound_patch_runtime_args writes at these positions, so they must track the helper.
namespace minimal_matmul_fabric_bound_layout {
// Kernel offsets from the helper's first kernel index.
inline constexpr uint32_t kIn0SenderKernel = 0;
inline constexpr uint32_t kIn0ReceiverKernel = 1;
inline constexpr uint32_t kIn1SenderKernel = 2;
inline constexpr uint32_t kIn1ReceiverKernel = 3;
inline constexpr uint32_t kComputeKernel = 4;
inline constexpr uint32_t kNumKernels = 5;

// in0 runtime args: [in0 address, bias address, AG input address, is_sink, next/prev noc (4), M/N tile ranges (4),
// defer_write_k_block, max_defer_write_k_block, num_local_k_blocks].
inline constexpr uint32_t kIn0InputAddrArg = 0;
inline constexpr uint32_t kIn0BiasAddrArg = 1;
inline constexpr uint32_t kIn0AgInputAddrArg = 2;
inline constexpr uint32_t kIn0FixedArgs = 15;
// in1 runtime args: [weight address, bias address, is_sink, next/prev noc (4), M/N tile ranges (4),
// defer_write_k_block, max_defer_write_k_block, num_local_k_blocks].
inline constexpr uint32_t kIn1WeightAddrArg = 0;
inline constexpr uint32_t kIn1BiasAddrArg = 1;
inline constexpr uint32_t kIn1FixedArgs = 14;
// With a fused ternary, [ternary_a address, ternary_b address, broadcast_ternary_b] follow the fixed args of both
// in0 and in1; the output addresses come next, one per output tensor.
inline constexpr uint32_t kTernaryAAddrOffset = 0;
inline constexpr uint32_t kTernaryBAddrOffset = 1;
inline constexpr uint32_t kTernaryArgs = 3;
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
