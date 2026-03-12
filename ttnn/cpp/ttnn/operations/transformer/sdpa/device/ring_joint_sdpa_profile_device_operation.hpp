// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"

#include "ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_profile_device_operation_types.hpp"
#include "ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_profile_program_factory.hpp"

namespace ttnn::prim {

/**
 * Device operation for profiling ring_joint_sdpa on a single device.
 *
 * This operation simulates what one device in a ring would compute,
 * using pre-staged KV data instead of actual ring communication.
 * Enables accurate compute time measurement without synchronization overhead.
 */
struct RingJointSDPAProfileDeviceOperation {
    using operation_attributes_t = RingJointSDPAProfileParams;
    using tensor_args_t = RingJointSDPAProfileInputs;
    using spec_return_value_t = RingJointSDPAProfileResultSpec;
    using tensor_return_value_t = RingJointSDPAProfileResult;
    using program_factory_t = std::variant<RingJointSDPAProfileProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::operation::OpPerformanceModelGeneral<Tensors> create_op_performance_model(
        const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensors);
};

/**
 * Entry point for ring_joint_sdpa profiling.
 *
 * @param input_tensor_q Local Q tensor for this device
 * @param input_tensor_k Local K tensor for this device
 * @param input_tensor_v Local V tensor for this device
 * @param gathered_k Pre-staged K from all devices in arrival order
 * @param gathered_v Pre-staged V from all devices in arrival order
 * @param ring_size Number of devices in the ring
 * @param ring_index Which device position we're simulating (0 to ring_size-1)
 * @param logical_n Logical sequence length (unpadded)
 * @param program_config SDPA program configuration
 * @param is_causal Whether to use causal attention
 * @param is_balanced Whether to use balanced workload distribution
 * @param scale Optional attention scale factor
 * @param compute_kernel_config Optional compute kernel configuration
 * @param joint_tensor_q Optional joint Q tensor
 * @param joint_tensor_k Optional joint K tensor
 * @param joint_tensor_v Optional joint V tensor
 * @param joint_strategy Joint attention strategy (default: "rear")
 */
RingJointSDPAProfileResult ring_joint_scaled_dot_product_attention_profile(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& gathered_k,
    const ttnn::Tensor& gathered_v,
    std::size_t ring_size,
    std::size_t ring_index,
    std::size_t logical_n,
    ttnn::operations::transformer::SDPAProgramConfig program_config,
    bool is_causal = false,
    bool is_balanced = false,
    std::optional<float> scale = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    const std::optional<ttnn::Tensor>& joint_tensor_q = std::nullopt,
    const std::optional<ttnn::Tensor>& joint_tensor_k = std::nullopt,
    const std::optional<ttnn::Tensor>& joint_tensor_v = std::nullopt,
    const std::optional<std::string>& joint_strategy = std::nullopt);

}  // namespace ttnn::prim
