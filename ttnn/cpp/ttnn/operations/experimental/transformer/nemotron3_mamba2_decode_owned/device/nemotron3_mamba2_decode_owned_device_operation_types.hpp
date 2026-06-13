// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::prim {

// Mamba2 SSD decode-step op — see
// experiments/owned_ops/nemotron3_mamba2_decode_owned/README.md and
// research/mm7_g1_mamba2_kernel_design.md.
struct Nemotron3Mamba2DecodeOwnedParams {
    std::optional<MemoryConfig> output_memory_config;
    bool debug_fill;      // shortcut for debug_mode=1 (fill_one smoke)
    uint32_t debug_mode;  // 0=production, 1=fill_one, 2..5=incremental
    // Softplus / clamp constants (passed as float32 bits per LLK calling
    // convention). Defaults match Nemotron-3 config.json.
    uint32_t softplus_beta_bits;        // 0x3f800000u = 1.0f
    uint32_t softplus_beta_recip_bits;  // 0x3f800000u = 1.0f
    uint32_t softplus_threshold_bits;   // 0x41a00000u = 20.0f
    uint32_t time_step_floor_bits;      // 0x38d1b717u = 1e-4f
    uint32_t time_step_max_bits;        // 0x3dcccccdu = 0.1f
};

// Tensor inputs to the Mamba2 SSD decode step. Kernel boundary matches
// NemotronHMamba2Mixer.forward POST-conv1d / silu / split and
// PRE-MambaRMSNormGated / out_proj. See architecture brief §4.3.
struct Nemotron3Mamba2DecodeOwnedInputs {
    const Tensor& x;                              // [B, num_heads, head_dim]                    bf16
    const Tensor& z;                              // [B, num_heads, head_dim]                    bf16 (pass-through)
    const Tensor& dt;                             // [B, num_heads]                              bf16
    const Tensor& dt_bias;                        // [num_heads]                                 bf16 (weight)
    const Tensor& A_log;                          // [num_heads]                                 bf16 (weight)
    const Tensor& D;                              // [num_heads]                                 bf16 (weight)
    const Tensor& B_in;                           // [B, n_groups, ssm_state]                    bf16
    const Tensor& C_in;                           // [B, n_groups, ssm_state]                    bf16
    const Tensor& ssm_state;                      // [B, num_heads, head_dim, ssm_state]         fp32 (mutated)
    const std::optional<Tensor>& preallocated_y;  // [B, num_heads, head_dim] bf16
};

}  // namespace ttnn::experimental::prim
