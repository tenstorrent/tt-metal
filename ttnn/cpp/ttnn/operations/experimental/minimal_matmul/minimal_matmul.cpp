// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "minimal_matmul.hpp"
#include "device/minimal_matmul_device_operation.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental {

ttnn::Tensor minimal_matmul(
    const std::variant<ttnn::Tensor, std::vector<ttnn::Tensor>>& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const std::optional<ttnn::Tensor>& bias_tensor,
    std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation,
    const std::optional<const ttnn::experimental::prim::MinimalMatmulConfig>& config,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    bool fuse_swiglu,
    const std::optional<ttnn::Tensor>& valid_rows_tensor,
    uint32_t valid_rows_addend,
    const std::optional<ttnn::Tensor>& slot_tensor,
    uint32_t kv_num_layers,
    uint32_t kv_layer_idx,
    std::optional<uint32_t> out_head_dim,
    bool in0_k_prefix) {
    // Unpack: single Tensor, or [prefix, suffix] virtually concatenated over K (concat-free).
    ttnn::Tensor main_input;
    std::optional<ttnn::Tensor> second_input;
    if (const auto* inputs = std::get_if<std::vector<ttnn::Tensor>>(&input_tensor)) {
        TT_FATAL(
            inputs->size() == 2,
            "minimal_matmul: list input must be exactly 2 tensors [prefix, suffix] for fused concat over K, got {}",
            inputs->size());
        main_input = (*inputs)[0];
        second_input = (*inputs)[1];
    } else {
        main_input = std::get<ttnn::Tensor>(input_tensor);
    }

    // Call device operation with chunks=1 (default), which returns a vector with 1 element
    auto outputs = ttnn::prim::minimal_matmul(
        main_input,
        weight_tensor,
        bias_tensor,
        std::move(fused_activation),
        config,
        memory_config,
        dtype,
        compute_kernel_config,
        /*chunks=*/1,
        /*dim=*/-1,
        /*fused_ternary_scalar=*/std::nullopt,
        /*fused_ternary_input_a=*/std::nullopt,
        /*fused_ternary_input_b=*/std::nullopt,
        fuse_swiglu,
        second_input,
        valid_rows_tensor,
        valid_rows_addend,
        slot_tensor,
        kv_num_layers,
        kv_layer_idx,
        out_head_dim,
        in0_k_prefix);

    // Extract and return the single output
    TT_FATAL(outputs.size() == 1, "Expected single output from minimal_matmul, got {}", outputs.size());
    return outputs[0];
}

}  // namespace ttnn::experimental
