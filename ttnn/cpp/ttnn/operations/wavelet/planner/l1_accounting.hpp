// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt_stl/assert.hpp>

#include "ttnn/operations/wavelet/device/protocol/lwt_config.hpp"

namespace ttnn::operations::wavelet {

namespace l1_detail {

constexpr uint64_t kSourceTileCircularBuffersBytes = uint64_t{8} * device_protocol::kLwtNarrowTileBytes;
constexpr uint64_t kBaseTileCircularBufferBytes = uint64_t{6} * device_protocol::kLwtNarrowTileBytes;
constexpr uint64_t kOutputTileCircularBufferBytes = uint64_t{6} * device_protocol::kLwtNarrowTileBytes;
constexpr uint64_t kCacheBytes = uint64_t{device_protocol::kLwtCacheStickCount} * device_protocol::kStickBytes;
constexpr uint64_t kSynchronizationBytes = 32;
constexpr uint64_t kMetadataBytes = uint64_t{2} * device_protocol::kRouteConfigPageBytes;

static_assert(kSourceTileCircularBuffersBytes == 16384);
static_assert(kBaseTileCircularBufferBytes == 12288);
static_assert(kOutputTileCircularBufferBytes == 12288);

}  // namespace l1_detail

[[nodiscard]] inline uint64_t checked_l1_allocation_bytes(
    const uint32_t workspace_elements,
    const uint32_t max_workspace_elements,
    const uint32_t workspace_mirror_elements,
    const uint32_t interleave_batch_sticks,
    const uint32_t architecture_scratch_bytes,
    const uint32_t capacity_bytes) {
    TT_FATAL(
        max_workspace_elements <= workspace_elements,
        "Logical workspace length {} exceeds allocated workspace length {}",
        max_workspace_elements,
        workspace_elements);
    TT_FATAL(interleave_batch_sticks > 0, "ILWT interleave batch must be non-zero");

    const uint64_t slots_bytes = uint64_t{3} * max_workspace_elements * sizeof(float);
    const uint64_t workspace_mirror_bytes = uint64_t{3} * workspace_mirror_elements * sizeof(float);
    const uint64_t padding_bytes = uint64_t{3} * (workspace_elements - max_workspace_elements) * sizeof(float);
    constexpr uint64_t circular_buffers_bytes =
        l1_detail::kSourceTileCircularBuffersBytes + l1_detail::kBaseTileCircularBufferBytes;
    const uint64_t output_bytes =
        l1_detail::kOutputTileCircularBufferBytes + uint64_t{interleave_batch_sticks} * device_protocol::kStickBytes;
    const uint64_t total_bytes = slots_bytes + workspace_mirror_bytes + circular_buffers_bytes +
                                 l1_detail::kCacheBytes + output_bytes + l1_detail::kSynchronizationBytes +
                                 l1_detail::kMetadataBytes + padding_bytes + architecture_scratch_bytes;
    TT_FATAL(
        total_bytes <= capacity_bytes,
        "tt-wavelet L1 allocation requires {} bytes/core, exceeding capacity {} by {} bytes",
        total_bytes,
        capacity_bytes,
        total_bytes - capacity_bytes);

    return total_bytes;
}

}  // namespace ttnn::operations::wavelet
