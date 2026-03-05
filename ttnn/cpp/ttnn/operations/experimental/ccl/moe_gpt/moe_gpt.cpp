// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_gpt.hpp"
#include "device/moe_gpt_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/shape/shape.hpp"

namespace ttnn::operations::experimental::moe_gpt {

ttnn::Tensor ExecuteMoEGPT::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& w0_w1_tensor,
    const ttnn::Tensor& w2_tensor,
    const ttnn::Tensor& output_tensor,
    const uint32_t num_experts,
    bool enable_dram_output,
    std::optional<ttnn::Tensor> dram_output_tensor,
    std::optional<ttnn::Tensor> sparse_buffer,
    std::optional<ttnn::Tensor> expert_indices,
    std::optional<ttnn::Tensor> expert_scores,
    std::optional<ttnn::Tensor> expert_mapping,
    std::optional<ttnn::Tensor> tilize_output,
    std::optional<uint32_t> cluster_axis) {
    if (enable_dram_output && !dram_output_tensor.has_value()) {
        auto shape = input_tensor.logical_shape();
        uint32_t M = shape[-2];
        uint32_t K = shape[-1];

        auto dram_mem_config =
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
        tt::tt_metal::TensorLayout tensor_layout(
            input_tensor.dtype(), tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), dram_mem_config);
        auto dram_spec = tt::tt_metal::TensorSpec(ttnn::Shape({num_experts, 1, M, K}), tensor_layout);
        dram_output_tensor = tt::tt_metal::create_device_tensor(dram_spec, input_tensor.device());
    }

    return ttnn::prim::moe_gpt(
        input_tensor,
        w0_w1_tensor,
        w2_tensor,
        output_tensor,
        num_experts,
        enable_dram_output,
        std::move(dram_output_tensor),
        std::move(sparse_buffer),
        std::move(expert_indices),
        std::move(expert_scores),
        std::move(expert_mapping),
        std::move(tilize_output),
        cluster_axis);
}

}  // namespace ttnn::operations::experimental::moe_gpt
