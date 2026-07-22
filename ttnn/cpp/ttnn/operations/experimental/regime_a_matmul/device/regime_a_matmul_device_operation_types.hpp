// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
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

    // NOTE: numerics are FIXED production behavior, not options — BF16 in/out, HiFi2, FP32 dest-accumulation,
    // DRAM-interleaved output. There is deliberately no output dtype / memory_config / compute_kernel_config
    // here: they were previously accepted but ignored (an API-correctness hazard), so they are not part of the
    // op's attributes or program-cache identity. The split `dim` is always -1 (validated in the wrapper) and is
    // likewise not stored/hashed — only `chunks` reaches the device op.
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
