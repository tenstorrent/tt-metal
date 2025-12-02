// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"

namespace ttnn::operations {

tt::tt_metal::operation::ProgramWithCallbacks matmul_multi_core_agmm_fusion_helper(
    tt::tt_metal::Program& program,
    const Tensor& input0,
    const std::vector<Tensor>& input_tensors_b,
    const std::optional<const Tensor>& bias,
    const std::vector<Tensor>& output_tensors,
    bool bcast_batch,
    DeviceComputeKernelConfig compute_kernel_config,
    const operations::matmul::MatmulProgramConfig& program_config,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id);

namespace llama_matmul {

tt::tt_metal::operation::ProgramWithCallbacks matmul_multi_core_agmm_fusion_helper(
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
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id);

}  // namespace llama_matmul

namespace llama_agmm_fusion_helpers {

void override_program_parameters(
    const ttnn::operations::matmul::matmul_mcast_1d_common_override_variables_t& override_variables,
    const void* operation,
    tt::tt_metal::Program& program,
    const std::vector<tt::tt_metal::Tensor>& input_tensors,
    const std::vector<std::optional<const tt::tt_metal::Tensor>>& optional_input_tensors,
    const std::vector<tt::tt_metal::Tensor>& output_tensors);

}  // namespace llama_agmm_fusion_helpers

}  // namespace ttnn::operations
