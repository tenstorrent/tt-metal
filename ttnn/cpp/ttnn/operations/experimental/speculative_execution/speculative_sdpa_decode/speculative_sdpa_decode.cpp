// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "speculative_sdpa_decode.hpp"

#include "device/speculative_sdpa_decode_op.hpp"
#include <tt-metalium/constants.hpp>
#include "ttnn/run_operation.hpp"
#include "cpp/ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"

using namespace tt::tt_metal;

namespace {
inline uint32_t get_chunk_size(uint32_t s) {
    /*
    # find maximum power of 2 divisor of s
    for i in range(1, s):
        if s % (2**(i+1)) != 0:
            break
    */
    uint32_t i = 1;
    for (; i < s; i++) {
        if (s % (1 << (i + 1)) != 0) {
            break;
        }
    }
    return std::min(512, 1 << i);
}
}  // namespace

namespace ttnn::operations::experimental::speculative_execution {

using SDPAProgramConfig = ttnn::operations::transformer::SDPAProgramConfig;

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor>
ExecuteSpeculativeScaledDotProductAttentionDecode::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    std::optional<float> lambda_,
    const bool is_causal,
    const std::optional<const Tensor>& attn_mask,
    const std::vector<uint32_t>& cur_pos,
    const std::optional<const Tensor>& cur_pos_tensor,
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<Tensor>& priority_tensor,
    const std::optional<Tensor>& other_priority_tensor,
    const bool ccl_enabled,
    const std::optional<global_semaphore::MultiDeviceGlobalSemaphore>& multi_device_global_semaphore) {
    auto arch = input_tensor_q.device()->arch();
    uint32_t s = input_tensor_k.get_logical_shape()[-2];
    uint32_t k_chunk_size = get_chunk_size(s);
    if (program_config.has_value() && program_config.value().k_chunk_size > 0) {
        k_chunk_size = program_config.value().k_chunk_size;
        // assert chunk size must be power of 2 and multiple of 32
        TT_FATAL(
            (k_chunk_size & (k_chunk_size - 1)) == 0,
            "User provided k_chunk_size must be power of 2, got: {}",
            k_chunk_size);
        TT_FATAL(k_chunk_size % 32 == 0, "User provided k_chunk_size must be multiple of 32, got: {}", k_chunk_size);
    } else {
        TT_FATAL(
            k_chunk_size % 32 == 0,
            "Chunk size must be multiple of 32, but the maximum calculated k_chunk_size is: {}",
            k_chunk_size);
    }

    // get chunk size and then pass to sdpa decode as an attribute for prgm cache
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, true, false, false);

    // ccl related
    auto devices = input_tensor_q.get_workers();
    uint32_t num_devices = devices.size();
    ttnn::ccl::Topology ccl_topology = ttnn::ccl::Topology::Ring;
    if (num_devices == 2) {
        ccl_topology = ttnn::ccl::Topology::Linear;
    }
    tt::log_info("devices: {}", devices);
    tt::log_info("num_devices: {}", num_devices);
    tt::log_info("ccl_enabled: {}", ccl_enabled);

    if (ccl_enabled) {
        TT_FATAL(num_devices > 1, "Must have at least 2 devices for ccl enabled, got: {}", num_devices);
        TT_FATAL(
            multi_device_global_semaphore.has_value(), "Must provide multi_device_global_semaphore for ccl enabled");
    }
    std::vector<Tensor> output_tensors = {// 4 output tensors
                                          Tensor(operation::get_workers_for_op_output({input_tensor_q})),
                                          Tensor(operation::get_workers_for_op_output({input_tensor_q})),
                                          Tensor(operation::get_workers_for_op_output({input_tensor_q})),
                                          Tensor(operation::get_workers_for_op_output({input_tensor_q}))};

    operation::launch_op(
        [lambda_,
         is_causal,
         cur_pos,
         scale,
         memory_config,
         program_config,
         kernel_config_val,
         k_chunk_size,
         ccl_enabled,
         devices,
         ccl_topology,
         multi_device_global_semaphore,
         queue_id](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            // ccl related
            auto this_device = input_tensors.at(0).device();
            uint32_t num_devices = devices.size();
            uint32_t device_index = 0;
            std::optional<IDevice*> forward_device = std::nullopt;
            std::optional<IDevice*> backward_device = std::nullopt;
            std::optional<GlobalSemaphore> semaphore = std::nullopt;
            if (ccl_enabled) {
                auto semaphores = multi_device_global_semaphore.value().global_semaphores;
                for (uint32_t i = 0; i < num_devices; ++i) {
                    if (devices.at(i) == this_device) {
                        device_index = i;
                        semaphore = semaphores.at(i);  // Get raw pointer
                        if (i != 0) {
                            backward_device = devices.at(i - 1);
                        }
                        if (i != num_devices - 1) {
                            forward_device = devices.at(i + 1);
                        }
                    }
                }
                tt::log_info("num_devices: {}", num_devices);
                tt::log_info("device_index: {}", device_index);
                tt::log_info("backward_device: {}", backward_device);
                tt::log_info("forward_device: {}", forward_device);
                tt::log_info("semaphore: {}", semaphore->address());
            }

            return operation::run(
                SpeculativeScaledDotProductAttentionDecode{
                    .lambda_ = lambda_,
                    .is_causal = is_causal,
                    .cur_pos = cur_pos,
                    .scale = scale,
                    .output_mem_config = memory_config.value_or(operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
                    .program_config = program_config,
                    .compute_kernel_config = kernel_config_val,
                    .k_chunk_size = k_chunk_size,
                    .paged_attention = false,
                    .ccl_enabled = ccl_enabled,
                    .num_devices = num_devices,
                    .device_index = device_index,
                    .topology = ccl_topology,
                    .semaphore = semaphore,
                    .forward_device = forward_device,
                    .backward_device = backward_device},
                input_tensors,
                optional_input_tensors,
                {},
                queue_id);
        },
        {input_tensor_q, input_tensor_k, input_tensor_v},
        output_tensors,
        {cur_pos_tensor, std::nullopt, attn_mask, priority_tensor, other_priority_tensor});

    TT_FATAL(output_tensors.size() == 4, "Expected 4 output tensors, got: {}", output_tensors.size());
    return std::make_tuple(
        std::move(output_tensors[0]),
        std::move(output_tensors[1]),
        std::move(output_tensors[2]),
        std::move(output_tensors[3]));
}

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor>
ExecuteSpeculativeScaledDotProductAttentionDecode::invoke(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    std::optional<float> lambda_,
    const bool is_causal,
    const std::optional<const Tensor>& attn_mask,
    const std::vector<uint32_t>& cur_pos,
    const std::optional<const Tensor>& cur_pos_tensor,
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<Tensor>& priority_tensor,
    const std::optional<Tensor>& other_priority_tensor,
    const bool ccl_enabled,
    const std::optional<global_semaphore::MultiDeviceGlobalSemaphore>& multi_device_global_semaphore) {
    return invoke(
        DefaultQueueId,
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        lambda_,
        is_causal,
        attn_mask,
        cur_pos,
        cur_pos_tensor,
        scale,
        memory_config,
        program_config,
        compute_kernel_config,
        priority_tensor,
        other_priority_tensor,
        ccl_enabled,
        multi_device_global_semaphore);
}

}  // namespace ttnn::operations::experimental::speculative_execution
