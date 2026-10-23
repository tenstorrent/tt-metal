// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/operations/common/shared_with_host/runtime_length.hpp"

// Per-worker slice of one (batch, channel) plane of the all-gather input, derived on device from the
// common row count so the reader and the writer of a worker always agree and a hit rewrites one word.
struct AllGatherWorkerSplit {
    uint32_t input_tile_id_start;
    uint32_t input_tile_id_end;
    uint32_t start_pages_read_in_row;
    uint32_t start_row_offset;
    uint32_t chunks_per_sync;
};

inline AllGatherWorkerSplit all_gather_worker_split(
    uint32_t global_worker_id,
    uint32_t global_worker_count,
    uint32_t input_Ht,
    uint32_t input_Wt,
    uint32_t output_Wt,
    uint32_t tiles_per_packet,
    uint32_t chunks_per_sync_or_zero) {
    const uint32_t pages = input_Ht * input_Wt;
    AllGatherWorkerSplit s;
    s.input_tile_id_start = ttnn::runtime_length::balanced_start(global_worker_id, pages, global_worker_count);
    s.input_tile_id_end = ttnn::runtime_length::balanced_end(global_worker_id, pages, global_worker_count);
    s.start_pages_read_in_row = s.input_tile_id_start % input_Wt;
    s.start_row_offset = s.input_tile_id_start / input_Wt * output_Wt;
    // Zero means "not pinned": one sync per packet, capped at 160 (sweep-derived heuristic).
    constexpr uint32_t kMaxChunksPerSync = 160;
    const uint32_t packets = (s.input_tile_id_end - s.input_tile_id_start) / tiles_per_packet;
    s.chunks_per_sync = chunks_per_sync_or_zero != 0
                            ? chunks_per_sync_or_zero
                            : ttnn::runtime_length::min_u32(packets > 1 ? packets : 1, kMaxChunksPerSync);
    return s;
}
