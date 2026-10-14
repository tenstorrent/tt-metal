// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/constants.hpp>

#include "qkv_causal_conv1d_silu_device_operation_types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

// Kimi-K3 supplies four learned causal-convolution taps; one channel block of weights is queued per tap.
inline constexpr uint32_t qkv_causal_conv1d_silu_tap_count = 4;

// Per-core DFB depths in channel blocks. Row-major activation, partial and output are double-buffered; partial
// must hold the previous tap's block, still being read, while compute writes the next tap's block.
inline constexpr uint32_t qkv_causal_conv1d_silu_act_rm_blocks = 2;
inline constexpr uint32_t qkv_causal_conv1d_silu_act_tile_blocks = 1;
inline constexpr uint32_t qkv_causal_conv1d_silu_partial_blocks = 2;
inline constexpr uint32_t qkv_causal_conv1d_silu_output_blocks = 2;

// The reader stages one channel block's tile rows plus the three history rows in a private window.
inline uint64_t qkv_causal_conv1d_silu_window_bytes(uint64_t block_ct, uint64_t element_bytes) {
    return (tt::constants::TILE_HEIGHT + qkv_causal_conv1d_silu_tap_count - 1) * block_ct * tt::constants::TILE_WIDTH *
           element_bytes;
}

inline uint64_t qkv_causal_conv1d_silu_l1_bytes(uint64_t block_ct, uint64_t tile_bytes, uint64_t element_bytes) {
    constexpr uint64_t dfb_blocks = qkv_causal_conv1d_silu_act_rm_blocks + qkv_causal_conv1d_silu_act_tile_blocks +
                                    qkv_causal_conv1d_silu_tap_count + qkv_causal_conv1d_silu_partial_blocks +
                                    qkv_causal_conv1d_silu_output_blocks;
    return (dfb_blocks * block_ct * tile_bytes) + qkv_causal_conv1d_silu_window_bytes(block_ct, element_bytes);
}

struct QkvCausalConv1dSiluProgramFactory {
    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const QkvCausalConv1dSiluParams&,
        const QkvCausalConv1dSiluInputs&,
        std::vector<Tensor>&,
        const ttnn::MeshCoordinateRangeSet&);
};

}  // namespace ttnn::experimental::prim
