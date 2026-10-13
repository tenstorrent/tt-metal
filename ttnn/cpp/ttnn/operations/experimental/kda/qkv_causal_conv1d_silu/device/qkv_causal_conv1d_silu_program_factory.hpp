// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/core_coord.hpp>

#include "qkv_causal_conv1d_silu_device_operation_types.hpp"
#include "qkv_causal_conv1d_silu_layout.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

// The fabric history exchange runs on the compute grid's last core, which the convolution leaves free.
inline CoreCoord qkv_causal_conv1d_silu_exchange_core(const CoreCoord& grid) { return {grid.x - 1, grid.y - 1}; }

// Every convolution core stages an equal share of the channel tiles of the rows the exchange core sends.
inline uint32_t qkv_causal_conv1d_silu_staged_ct(uint32_t channel_tiles, uint32_t convolution_cores) {
    return (channel_tiles + convolution_cores - 1) / convolution_cores;
}

inline uint64_t qkv_causal_conv1d_silu_l1_bytes(uint32_t block_ct, uint64_t tile_bytes, uint32_t max_staged_ct) {
    using namespace qkv_conv;
    const uint64_t group = dest_group(block_ct);
    // Tap tiles, diagonal weights and the identity; double-buffered shifted taps and output groups.
    const uint64_t dfb_tiles = (2 * tap_count * block_ct) + 1 + (2 * tap_count * group) + (2 * group);
    return (dfb_tiles * tile_bytes) + ReaderScratch{block_ct}.bytes() + StagingScratch{max_staged_ct}.bytes();
}

struct QkvCausalConv1dSiluProgramFactory {
    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const QkvCausalConv1dSiluParams&,
        const QkvCausalConv1dSiluInputs&,
        std::vector<Tensor>&,
        const ttnn::MeshCoordinateRangeSet&);
};

}  // namespace ttnn::experimental::prim
