// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>
#include <optional>

#include <tt-metalium/core_coord.hpp>

#include "qkv_causal_conv1d_silu_device_operation_types.hpp"
#include "hostdevcommon/qkv_causal_conv1d_silu_layout.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

namespace ttnn::experimental::prim {

// The fabric history exchange runs on the first Tensix row's last core, which the convolution leaves free. That row
// sits next to the ETH cores, so its fabric packets do not cross the chip and the convolution cores' traffic.
inline tt::tt_metal::CoreCoord qkv_causal_conv1d_silu_exchange_core(const tt::tt_metal::CoreCoord& grid) {
    return {grid.x - 1, 0};
}

// The convolution's work items, row-major over the grid; with the exchange, every core but the exchange core.
inline kda_factory_detail::KdaPrepWorkDist qkv_causal_conv1d_silu_distribution(
    const tt::tt_metal::CoreCoord& grid, uint32_t work_items, bool exchange) {
    return kda_factory_detail::distribute_prep(
        grid,
        work_items,
        std::numeric_limits<uint32_t>::max(),
        exchange ? std::optional(qkv_causal_conv1d_silu_exchange_core(grid)) : std::nullopt);
}

// Every convolution core stages an equal share of the channel tiles of the rows the exchange core sends.
inline uint32_t qkv_causal_conv1d_silu_staged_ct(uint32_t channel_tiles, uint32_t convolution_cores) {
    return (channel_tiles + convolution_cores - 1) / convolution_cores;
}

// Dataflow buffer depths in tiles: double-buffered shifted taps and output groups, and per channel block the tap
// tiles, the identity and the diagonal weights.
struct QkvCausalConv1dSiluDfbTiles {
    uint32_t act_tile;
    uint32_t taps;
    uint32_t eye;
    uint32_t weights;
    uint32_t output;
    uint32_t total() const { return act_tile + taps + eye + weights + output; }
};

inline QkvCausalConv1dSiluDfbTiles qkv_causal_conv1d_silu_dfb_tiles(uint32_t block_ct, uint32_t dest_limit) {
    using qkv_conv::tap_count;
    const uint32_t group = qkv_conv::dest_group(block_ct, dest_limit);
    return {
        .act_tile = 2 * tap_count * group,
        .taps = tap_count * block_ct,
        .eye = 1,
        .weights = tap_count * block_ct,
        .output = 2 * group};
}

inline uint64_t qkv_causal_conv1d_silu_l1_bytes(
    uint32_t block_ct, uint32_t dest_limit, uint64_t tile_bytes, uint32_t max_staged_ct) {
    return (qkv_causal_conv1d_silu_dfb_tiles(block_ct, dest_limit).total() * tile_bytes) +
           qkv_conv::ReaderScratch{block_ct}.bytes() + qkv_conv::StagingScratch{max_staged_ct}.bytes();
}

struct QkvCausalConv1dSiluProgramFactory {
    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const QkvCausalConv1dSiluParams&,
        const QkvCausalConv1dSiluInputs&,
        std::vector<Tensor>&,
        const ttnn::MeshCoordinateRangeSet&);
};

}  // namespace ttnn::experimental::prim
