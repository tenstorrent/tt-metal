// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/experimental/regime_a_matmul/device/regime_a_matmul_config.hpp"

namespace ttnn::experimental::prim {

struct RegimeAMatmulParams {
    // The whole config drives compile-time kernel args, so it lives in operation_attributes and is
    // keyed by the framework's default reflection-based program hash.
    std::optional<RegimeAMatmulConfig> config;

    // ---- Production single-chip fusions (all optional; nullopt/1 => byte-identical no-fusion path). ----
    // Applied at the output/compute stage (post split-K reduction for Pk>1) so no extra output DRAM
    // round-trip. All fusion presence flags participate in the reflection-based program-cache hash.
    std::optional<operations::unary::UnaryWithParam> fused_activation;  // Y = act(A@B + bias)
    // addcmul: Y = residual + scalar*(A@B + bias)*gate. Scalar present <=> addcmul active (residual/gate
    // tensors live in RegimeAMatmulInputs). Rejected together with fused_activation.
    std::optional<float> fused_ternary_scalar;
    int32_t chunks = 1;  // output column-split count (regime_a_matmul_split); 1 => single output tensor
    int32_t dim = -1;    // split dim (only -1 supported)

    std::optional<tt::tt_metal::MemoryConfig> output_mem_config;
    std::optional<tt::tt_metal::DataType> output_dtype;

    DeviceComputeKernelConfig compute_kernel_config;

    // Test-only ablation bitmask (RegimeADiag). 0 for the public path. Part of the reflection-based
    // program-cache hash, so a diagnostic program never aliases a normal one. Set only via the internal
    // ttnn::prim::regime_a_matmul_diag entry; never through Python/nanobind.
    uint32_t diag_mask = 0;
};

struct RegimeAMatmulInputs {
    Tensor input_tensor;   // in0 : [.., M, K], DRAM interleaved, bf16, TILE
    Tensor weight_tensor;  // in1 : [.., K, N], DRAM width-sharded (8 banks), bf16, TILE

    // ---- Optional fusion operands (DRAM interleaved, TILE). ----
    std::optional<Tensor> bias_tensor;            // [.., 1, N] / [.., N] row-broadcast bias
    std::optional<Tensor> fused_ternary_input_a;  // addcmul residual/base, full [M, N]
    std::optional<Tensor> fused_ternary_input_b;  // addcmul gate/multiplier, [1, N] bcast or [M, N] full
};

}  // namespace ttnn::experimental::prim
