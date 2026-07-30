// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/core_coord.hpp>

namespace ttnn::experimental {

// Direct (one-shot) reduce-scatter: latency-optimal sibling of reduce_scatter_minimal_unicast.
//
// Every device unicasts each destination's slice straight to that destination (multi-hop fabric unicast,
// no intermediate device touches the data), increments a per-source arrival counter with the send's last
// packet, and once all N-1 contributions have landed reduces them with its own slice into the output. One
// fabric traversal instead of the ring's N/2 store-and-forward steps, at ~2.3x the link traffic -- a
// latency play for small/medium shapes, not a bandwidth play.
//
// Minimal-first: Ring, TILE layout, any rank >= 2 and any scatter dim (divisible by the ring size in
// tile/page units), one worker core per link (which owns that link's forward and backward connection),
// no mux.
//
// persistent_buffers, when provided, must be exactly {output, staging} as produced by
// reduce_scatter_minimal_direct_create_persistent_buffers.
ttnn::Tensor reduce_scatter_minimal_direct(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    std::optional<uint32_t> cluster_axis = std::nullopt,
    std::optional<uint32_t> num_links = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<std::vector<ttnn::Tensor>>& persistent_buffers = std::nullopt,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt);

// Allocate the persistent buffer set {output, staging} sized to match a given input.
std::vector<ttnn::Tensor> reduce_scatter_minimal_direct_create_persistent_buffers(
    const ttnn::Tensor& input_tensor, int32_t dim, std::optional<uint32_t> cluster_axis = std::nullopt);

// Allocate ONLY the staging buffer (element [1] of the set above), for callers that own their output
// tensor already and just want the op's staging half. The output is an ordinary tiled tensor a caller
// can build directly (the input shape with `dim` divided by the ring size); staging is the part that
// cannot be reproduced by hand -- it is an opaque chunk-paged UINT8 tensor whose page size follows the
// op's chunk granularity and whose placement (L1 height-sharded over the whole compute grid / L1
// interleaved / DRAM) is chosen from the shape. Pass the result through as
// persistent_buffers = {your_output, this}. `dim` and `cluster_axis` must match the op call.
ttnn::Tensor reduce_scatter_minimal_direct_create_staging_buffer(
    const ttnn::Tensor& input_tensor, int32_t dim, std::optional<uint32_t> cluster_axis = std::nullopt);

}  // namespace ttnn::experimental
