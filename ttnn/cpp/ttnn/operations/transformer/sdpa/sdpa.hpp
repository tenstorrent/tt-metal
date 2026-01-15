// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"

namespace ttnn {
namespace operations::transformer {

struct ExecuteScaledDotProductAttention {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor_q,
        const ttnn::Tensor& input_tensor_k,
        const ttnn::Tensor& input_tensor_v,
        const std::optional<ttnn::Tensor>& attn_mask = std::nullopt,
        bool is_causal = true,
        std::optional<float> scale = std::nullopt,
        std::optional<uint32_t> sliding_window_size = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<SDPAProgramConfig> program_config = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        const std::optional<ttnn::Tensor>& attention_sink = std::nullopt);
};

struct ExecuteChunkedScaledDotProductAttention {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor_q,
        const ttnn::Tensor& input_tensor_k,
        const ttnn::Tensor& input_tensor_v,
        const ttnn::Tensor& page_table_tensor,
        int64_t chunk_start_idx,  // Must be a multiple of program_config.q_chunk_size
        std::optional<float> scale = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<SDPAProgramConfig> program_config = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

struct ExecuteJointAttention {
    static std::tuple<ttnn::Tensor, ttnn::Tensor> invoke(
        const ttnn::Tensor& input_tensor_q,
        const ttnn::Tensor& input_tensor_k,
        const ttnn::Tensor& input_tensor_v,
        const ttnn::Tensor& joint_tensor_q,
        const ttnn::Tensor& joint_tensor_k,
        const ttnn::Tensor& joint_tensor_v,
        const std::string& joint_strategy,
        SDPAProgramConfig program_config,
        std::optional<float> scale = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

struct ExecuteRingJointAttention {
    static std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> invoke(
        const ttnn::Tensor& input_tensor_q,
        const ttnn::Tensor& input_tensor_k,
        const ttnn::Tensor& input_tensor_v,
        const ttnn::Tensor& joint_tensor_q,
        const ttnn::Tensor& joint_tensor_k,
        const ttnn::Tensor& joint_tensor_v,
        ttnn::Tensor& persistent_output_buffer_k,
        ttnn::Tensor& persistent_output_buffer_v,
        const std::string& joint_strategy,
        std::size_t logical_n,
        SDPAProgramConfig program_config,
        int32_t dim,
        const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
        uint32_t num_links,
        uint32_t cluster_axis,
        const MeshDevice& mesh_device,
        ttnn::ccl::Topology topology,
        std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
        CoreCoord ccl_core_grid_offset,
        std::optional<float> scale = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

struct ExecuteFlashMLAPrefill {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor_q,
        const ttnn::Tensor& input_tensor_k,
        uint32_t head_dim_v,
        const std::optional<ttnn::Tensor>& attn_mask = std::nullopt,
        bool is_causal = true,
        std::optional<float> scale = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<SDPAProgramConfig> program_config = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

struct ExecuteChunkedFlashMLAPrefill {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor_q,
        const ttnn::Tensor& input_tensor_k,
        uint32_t head_dim_v,
        const ttnn::Tensor& page_table_tensor,
        int64_t chunk_start_idx,
        std::optional<float> scale = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<SDPAProgramConfig> program_config = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

struct ExecuteRingDistributedScaledDotProductAttention {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor_q,
        const ttnn::Tensor& input_tensor_k,
        const ttnn::Tensor& input_tensor_v,
        uint32_t ring_size,
        std::optional<uint32_t> ring_id =
            std::nullopt,  // Optional: if provided, uses this value; if nullopt, infers from device coordinate
        std::optional<float> scale = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<SDPAProgramConfig>& program_config = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        const std::optional<ttnn::Tensor>& page_table = std::nullopt,
        std::optional<int64_t> chunk_start_idx = std::nullopt);
};

}  // namespace operations::transformer

namespace transformer {

constexpr auto scaled_dot_product_attention = ttnn::register_operation<
    "ttnn::transformer::scaled_dot_product_attention",
    ttnn::operations::transformer::ExecuteScaledDotProductAttention>();

constexpr auto chunked_scaled_dot_product_attention = ttnn::register_operation<
    "ttnn::transformer::chunked_scaled_dot_product_attention",
    ttnn::operations::transformer::ExecuteChunkedScaledDotProductAttention>();

constexpr auto joint_scaled_dot_product_attention = ttnn::register_operation<
    "ttnn::transformer::joint_scaled_dot_product_attention",
    ttnn::operations::transformer::ExecuteJointAttention>();

constexpr auto ring_joint_scaled_dot_product_attention = ttnn::register_operation<
    "ttnn::transformer::ring_joint_scaled_dot_product_attention",
    ttnn::operations::transformer::ExecuteRingJointAttention>();

constexpr auto flash_mla_prefill = ttnn::
    register_operation<"ttnn::transformer::flash_mla_prefill", ttnn::operations::transformer::ExecuteFlashMLAPrefill>();

constexpr auto chunked_flash_mla_prefill = ttnn::register_operation<
    "ttnn::transformer::chunked_flash_mla_prefill",
    ttnn::operations::transformer::ExecuteChunkedFlashMLAPrefill>();

constexpr auto ring_distributed_scaled_dot_product_attention = ttnn::register_operation<
    "ttnn::transformer::ring_distributed_scaled_dot_product_attention",
    ttnn::operations::transformer::ExecuteRingDistributedScaledDotProductAttention>();

}  // namespace transformer

}  // namespace ttnn
