// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <variant>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

// Gate factor: SIGMOID -> sigmoid(gate); SILU -> gate * sigmoid(gate).
enum class SigmoidGatedRmsNormGateActivation : uint8_t { SIGMOID, SILU };

struct SigmoidGatedRmsNormParams {
    uint32_t batch;
    uint32_t num_heads;
    uint32_t sequence;
    uint32_t value_dim;
    float epsilon;
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype;
    DeviceComputeKernelConfig compute_kernel_config;
    SigmoidGatedRmsNormGateActivation gate_activation = SigmoidGatedRmsNormGateActivation::SIGMOID;
    // The op reads gate tile columns [gate_col_offset_tiles, gate_col_offset_tiles + H*V/32) of each tile row;
    // the tile-row stride is the gate's padded width in tiles, so the gate may be wider than H*V.
    uint32_t gate_col_offset_tiles = 0;
    // Compute-kernel variant: 0 = legacy kernel, 1..4 = fused kernel (see kSigmoidGatedRmsNormKernelVariant*).
    uint32_t kernel_variant = 0;
};

// Kernel variants (see sigmoid_gated_rms_norm_fused.cpp). 1-3 give the same result; 4 changes the sigmoid's exp.
inline constexpr uint32_t kSigmoidGatedRmsNormKernelVariantLegacy = 0;       // seven passes, one pack per pass
inline constexpr uint32_t kSigmoidGatedRmsNormKernelVariantFused = 1;        // fused + pipelined, library SFPU gate
inline constexpr uint32_t kSigmoidGatedRmsNormKernelVariantFusedGate = 2;    // + one fused SFPU pass for the gate
inline constexpr uint32_t kSigmoidGatedRmsNormKernelVariantPackSfpu = 3;     // + SFPU work on the PACK thread
inline constexpr uint32_t kSigmoidGatedRmsNormKernelVariantFastSigmoid = 4;  // + exp_21f sigmoid (not bit-exact)
inline constexpr uint32_t kSigmoidGatedRmsNormKernelVariantMax = 4;
inline constexpr uint32_t kSigmoidGatedRmsNormDefaultKernelVariant = kSigmoidGatedRmsNormKernelVariantFastSigmoid;

struct SigmoidGatedRmsNormInputs {
    Tensor input;
    Tensor gate;
    Tensor weight;
};

}  // namespace ttnn::experimental::prim
