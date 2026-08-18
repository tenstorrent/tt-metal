// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Gather-extent derivation shared by the all-gather reader and writer on the trace-safe metadata path.
// Both kernels must clamp the gather to the SAME slab prefix or the cb_output producer/consumer page
// counts drift apart, so the formula lives here in one place instead of being duplicated per kernel.
//
// KEEP IN SYNC with the host reference, `compute_gather_valid_Ht`
// (ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp): on the scalar
// path the host computes the extent and passes it as a runtime arg, on the metadata path these kernels
// recompute it on-device from kv_actual_isl. The metadata==scalar bit-exact tests only guard the points
// they exercise, so a change to one side that is not mirrored in the other diverges silently elsewhere.

#pragma once

#include <cstdint>

namespace ring_attention_all_gather {

// gather_valid_Ht = ceil(logical_n / chunk_global) * chunk_local_tiles, where
// logical_nt = kv_actual_isl / TILE_HEIGHT + chunk_global_tiles and chunk_global_tiles =
// chunk_local_tiles * ring_size. TILE_HEIGHT is 32, hence the >> 5.
inline uint32_t compute_gather_valid_Ht(uint32_t kv_actual_isl, uint32_t chunk_local_tiles, uint32_t ring_size) {
    const uint32_t chunk_global_tiles = chunk_local_tiles * ring_size;
    const uint32_t logical_nt_local = (kv_actual_isl >> 5) + chunk_global_tiles;
    const uint32_t valid_slabs = (logical_nt_local + chunk_global_tiles - 1) / chunk_global_tiles;
    return valid_slabs * chunk_local_tiles;
}

// Store a bounded page count and derive this worker link's disjoint contiguous logical range. Reader and
// writer share this helper so metadata replay cannot update their CB protocol differently.
FORCE_INLINE void set_worker_page_range(
    uint32_t valid_pages,
    uint32_t num_links,
    uint32_t worker_link,
    uint32_t& stored_valid_pages,
    uint32_t& tile_id_start,
    uint32_t& tile_id_end) {
    stored_valid_pages = valid_pages;
    const uint32_t pages_per_link = valid_pages / num_links;
    const uint32_t remainder = valid_pages % num_links;
    tile_id_start = worker_link * pages_per_link + (worker_link < remainder ? worker_link : remainder);
    const uint32_t next_link = worker_link + 1;
    tile_id_end = next_link * pages_per_link + (next_link < remainder ? next_link : remainder);
}

}  // namespace ring_attention_all_gather
