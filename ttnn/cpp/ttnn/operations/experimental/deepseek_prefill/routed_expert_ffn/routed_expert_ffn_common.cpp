// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_expert_ffn.hpp"
#include "routed_expert_ffn_common.hpp"

#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn {

namespace detail {

uint32_t best_in0_block_w(
    uint32_t K_tiles,
    uint32_t per_core_M,
    uint32_t per_core_N,
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    tt::tt_metal::DataType output_dtype,
    float l1_safety_margin) {
    using namespace ttnn::operations::matmul::utilities;

    // get_estimated_size_of_cbs is an estimate that doesn't account for CB alignment
    // overhead and other internal allocations. Apply a 10% safety margin (same approach
    // as softmax op) so the chosen block width fits on all platforms.
    uint32_t max_l1_space = static_cast<uint32_t>(get_max_l1_space(input_tensor_a) * l1_safety_margin);
    uint32_t interm_tile_size = estimate_interm_tile_size(compute_kernel_config, output_dtype);

    // best should be larger than 0 if there's enough L1.
    uint32_t best = 0;
    for (uint32_t d = 1; d <= K_tiles; ++d) {
        if (K_tiles % d != 0) {
            continue;
        }
        uint32_t estimated_size = get_estimated_size_of_cbs(
            /*per_core_M=*/per_core_M,
            /*per_core_N=*/per_core_N,
            /*in0_block_w=*/d,
            /*input_tensor_a=*/input_tensor_a,
            /*input_tensor_b=*/input_tensor_b,
            /*transpose_a=*/false,
            /*transpose_b=*/false,
            /*interm_single_tile_size=*/interm_tile_size,
            /*bias_single_tile_size=*/0);
        if (estimated_size < max_l1_space) {
            best = d;
        }
    }

    if (best == 0) {
        // CB size is monotonic in in0_block_w (in0/in1 CBs scale linearly with it),
        // so the smallest possible config is at d=1. Recompute it for the error so
        // the caller can see exactly how much over budget we are.
        uint32_t min_cb_size = get_estimated_size_of_cbs(
            /*per_core_M=*/per_core_M,
            /*per_core_N=*/per_core_N,
            /*in0_block_w=*/1,
            /*input_tensor_a=*/input_tensor_a,
            /*input_tensor_b=*/input_tensor_b,
            /*transpose_a=*/false,
            /*transpose_b=*/false,
            /*interm_single_tile_size=*/interm_tile_size,
            /*bias_single_tile_size=*/0);
        TT_FATAL(
            false,
            "best_in0_block_w: no in0_block_w divides upper bound {} and fits L1. "
            "per_core_M={}, per_core_N={}, in0 dtype={}, in1 dtype={}, output dtype={}. "
            "L1 budget {}%: {} B; minimum estimated CB size (at in0_block_w=1): {} B. "
            "Consider reducing per_core_M/per_core_N, using a smaller weight dtype (bfp8/bfp4), "
            "or using a larger compute grid to shrink per-core work.",
            K_tiles,
            per_core_M,
            per_core_N,
            input_tensor_a.dtype(),
            input_tensor_b.dtype(),
            output_dtype,
            static_cast<int>(100 * l1_safety_margin),
            max_l1_space,
            min_cb_size);
    }

    return best;
}

ttnn::Tensor routed_expert_ffn_default(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    std::optional<ttnn::Tensor> output) {
    auto gate_result = ttnn::matmul(
        /*input_tensor_a=*/x,
        /*input_tensor_b=*/gate_proj,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        /*memory_config=*/std::nullopt,
        /*dtype=*/std::nullopt,
        /*program_config=*/std::nullopt,
        /*activation=*/std::string("silu"),
        /*compute_kernel_config=*/compute_kernel_config);

    auto up_result = ttnn::matmul(
        /*input_tensor_a=*/x,
        /*input_tensor_b=*/up_proj,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        /*memory_config=*/std::nullopt,
        /*dtype=*/std::nullopt,
        /*program_config=*/std::nullopt,
        /*activation=*/std::nullopt,
        /*compute_kernel_config=*/compute_kernel_config);

    ttnn::multiply_(/*lhs=*/gate_result, /*rhs=*/up_result);
    up_result.deallocate();

    return ttnn::matmul(
        /*input_tensor_a=*/gate_result,
        /*input_tensor_b=*/down_proj,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        /*memory_config=*/std::nullopt,
        /*dtype=*/std::nullopt,
        /*program_config=*/std::nullopt,
        /*activation=*/std::nullopt,
        /*compute_kernel_config=*/compute_kernel_config,
        /*core_grid=*/std::nullopt,
        /*output_tile=*/std::nullopt,
        /*optional_output_tensor=*/std::move(output));
}

}  // namespace detail

ttnn::Tensor routed_expert_ffn(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    std::optional<ttnn::Tensor> output) {
    const uint32_t M_tiles = x.padded_shape()[-2] / ttnn::TILE_SIZE;
    const bool is_wormhole = x.device()->arch() == tt::ARCH::WORMHOLE_B0;

    // Fall back to default (auto-configured) matmuls for very large M where the
    // manually configured program configs would need too many grid rows.
    if (M_tiles > 64) {
        return detail::routed_expert_ffn_default(
            /*x=*/x,
            /*gate_proj=*/gate_proj,
            /*up_proj=*/up_proj,
            /*down_proj=*/down_proj,
            /*compute_kernel_config=*/compute_kernel_config,
            /*output=*/std::move(output));
    }

    TT_FATAL(
        x.dtype() == DataType::BFLOAT16 || x.dtype() == DataType::BFLOAT8_B,
        "routed_expert_ffn: x must be BFLOAT16 or BFLOAT8_B, got {}",
        x.dtype());

    if (is_wormhole) {
        return detail::routed_expert_ffn_wh(x, gate_proj, up_proj, down_proj, compute_kernel_config, output);
    }
    return detail::routed_expert_ffn_bh(x, gate_proj, up_proj, down_proj, compute_kernel_config, std::move(output));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn
