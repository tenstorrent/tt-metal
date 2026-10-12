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

// Tiled input: the reader keeps two parities of [left, right] 35-row strips per channel tile, one staging tile per
// channel tile and three rows of the channel block, plus alignment slack.
inline uint64_t qkv_causal_conv1d_silu_tiled_scratch_bytes(uint64_t block_ct, uint64_t tile_bytes) {
    constexpr uint64_t strip_stride = 1152;
    return (2 * block_ct * 2 * strip_stride) + 64 + (block_ct * tile_bytes) +
           ((qkv_causal_conv1d_silu_tap_count - 1) * block_ct * 2 * tt::constants::TILE_WIDTH) + 64;
}

// Channel tiles per compute group: the destination registers hold one group's outputs.
inline uint32_t qkv_causal_conv1d_silu_tiled_group(uint32_t block_ct) {
    for (uint32_t group = 8; group > 1; --group) {
        if (block_ct % group == 0) {
            return group;
        }
    }
    return 1;
}

inline uint64_t qkv_causal_conv1d_silu_tiled_l1_bytes(uint64_t block_ct, uint64_t tile_bytes) {
    const uint64_t group = qkv_causal_conv1d_silu_tiled_group(block_ct);
    // Tap tiles, diagonal weights and the identity; double-buffered shifted taps and output groups.
    const uint64_t dfb_tiles = (2 * qkv_causal_conv1d_silu_tap_count * block_ct) + 1 +
                               (2 * qkv_causal_conv1d_silu_tap_count * group) + (2 * group);
    return (dfb_tiles * tile_bytes) + qkv_causal_conv1d_silu_tiled_scratch_bytes(block_ct, tile_bytes);
}

struct QkvCausalConv1dSiluTiledProgramFactory {
    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const QkvCausalConv1dSiluParams&,
        const QkvCausalConv1dSiluInputs&,
        std::vector<Tensor>&,
        const ttnn::MeshCoordinateRangeSet&);
};

struct QkvCausalConv1dSiluProgramFactory {
    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const QkvCausalConv1dSiluParams&,
        const QkvCausalConv1dSiluInputs&,
        std::vector<Tensor>&,
        const ttnn::MeshCoordinateRangeSet&);
};

}  // namespace ttnn::experimental::prim
