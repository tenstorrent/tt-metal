// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/constants.hpp>

#include "qkv_causal_conv1d_silu_device_operation_types.hpp"
#include "qkv_causal_conv1d_silu_layout.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

// Kimi-K3 supplies four learned causal-convolution taps; one channel block of weights is queued per tap.
inline constexpr uint32_t qkv_causal_conv1d_silu_tap_count = 4;

// The reader keeps two parities of [left, right] strips per channel tile, one staging tile per channel tile and three
// rows of the channel block, plus alignment slack.
inline uint64_t qkv_causal_conv1d_silu_scratch_bytes(uint64_t block_ct, uint64_t tile_bytes) {
    return (2 * block_ct * 2 * qkv_conv::strip_stride) + 64 + (block_ct * tile_bytes) +
           (qkv_conv::history_rows * block_ct * 2 * tt::constants::TILE_WIDTH) + 64;
}

// Channel tiles per compute group: the destination registers hold one group's outputs.
inline uint32_t qkv_causal_conv1d_silu_group(uint32_t block_ct) {
    for (uint32_t group = 8; group > 1; --group) {
        if (block_ct % group == 0) {
            return group;
        }
    }
    return 1;
}

inline uint64_t qkv_causal_conv1d_silu_l1_bytes(uint64_t block_ct, uint64_t tile_bytes, uint32_t max_staged_ct) {
    const uint64_t group = qkv_causal_conv1d_silu_group(block_ct);
    // Tap tiles, diagonal weights and the identity; double-buffered shifted taps and output groups.
    const uint64_t dfb_tiles = (2 * qkv_causal_conv1d_silu_tap_count * block_ct) + 1 +
                               (2 * qkv_causal_conv1d_silu_tap_count * group) + (2 * group);
    return (dfb_tiles * tile_bytes) + qkv_causal_conv1d_silu_scratch_bytes(block_ct, tile_bytes) + 128 +
           qkv_conv::staged_rows_scratch_bytes(2 * qkv_conv::history_rows, max_staged_ct);
}

struct QkvCausalConv1dSiluProgramFactory {
    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const QkvCausalConv1dSiluParams&,
        const QkvCausalConv1dSiluInputs&,
        std::vector<Tensor>&,
        const ttnn::MeshCoordinateRangeSet&);
};

}  // namespace ttnn::experimental::prim
