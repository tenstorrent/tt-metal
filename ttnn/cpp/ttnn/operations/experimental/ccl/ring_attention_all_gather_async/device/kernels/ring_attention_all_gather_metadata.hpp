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

#include <array>
#include <cstddef>
#include <cstdint>

#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/metadata_scalar_read.hpp"

namespace ring_attention_all_gather {

struct LinkPageRange {
    uint32_t start;
    uint32_t end;
};

// Partition a valid page prefix as evenly as possible across links. Earlier links receive one
// additional page when the prefix is not evenly divisible. Reader and writer must use the same
// range or their cb_output producer/consumer page counts can diverge.
inline LinkPageRange compute_link_page_range(uint32_t valid_pages, uint32_t num_links, uint32_t link_idx) {
    const uint32_t pages_per_link = valid_pages / num_links;
    const uint32_t remainder = valid_pages % num_links;
    const uint32_t next_link_idx = link_idx + 1;
    return {
        link_idx * pages_per_link + (link_idx < remainder ? link_idx : remainder),
        next_link_idx * pages_per_link + (next_link_idx < remainder ? next_link_idx : remainder)};
}

// gather_valid_Ht = ceil(logical_n / chunk_global) * chunk_local_tiles, where
// logical_nt = kv_actual_isl / TILE_HEIGHT + chunk_global_tiles and chunk_global_tiles =
// chunk_local_tiles * ring_size. TILE_HEIGHT is 32, hence the >> 5.
inline uint32_t compute_gather_valid_Ht(
    uint32_t kv_actual_isl, uint32_t chunk_local_tiles, uint32_t ring_size, uint32_t cache_local_tile_rows) {
    const uint32_t chunk_global_tiles = chunk_local_tiles * ring_size;
    const uint32_t cache_global_tiles = cache_local_tile_rows * ring_size;
    kv_actual_isl = trace_metadata::bounded_kv_actual_isl(kv_actual_isl, chunk_global_tiles, cache_global_tiles);
    const uint32_t logical_nt_local = (kv_actual_isl >> 5) + chunk_global_tiles;
    const uint32_t valid_slabs = (logical_nt_local + chunk_global_tiles - 1) / chunk_global_tiles;
    return valid_slabs * chunk_local_tiles;
}

// Which cache-group tail the one-hop neighbor halo reads, derived on-device.
//
// KEEP IN SYNC with the host reference, chunked_sliding_halo_source_start_tile
// (ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sliding_window_work_plan.hpp) and its caller
// ChunkedSlidingHaloLayout::send_tail_start_tile. The result is linear in the chunk index
// (current_group == chunk index for chunk-aligned prefill), so on the scalar path the host rewrites the
// halo reader/writer page ranges every dispatch. A captured trace never runs that rewrite, which is why
// this has to be recomputed here instead.
inline uint32_t compute_halo_tail_start_Ht(
    uint32_t kv_actual_isl,
    uint32_t q_local_tile_rows,
    uint32_t ring_size,
    uint32_t halo_tile_rows,
    uint32_t source_device,
    uint32_t cache_local_tile_rows) {
    const uint32_t q_group_tile_rows = q_local_tile_rows * ring_size;
    if (q_group_tile_rows == 0 || halo_tile_rows > q_local_tile_rows) {
        return 0;
    }
    kv_actual_isl =
        trace_metadata::bounded_kv_actual_isl(kv_actual_isl, q_group_tile_rows, cache_local_tile_rows * ring_size);
    const uint32_t logical_k_tile_rows = (kv_actual_isl >> 5) + q_group_tile_rows;
    if (logical_k_tile_rows < q_group_tile_rows) {
        return 0;
    }
    const uint32_t current_group = logical_k_tile_rows / q_group_tile_rows - 1;
    if (current_group == 0 && source_device + 1 == ring_size) {
        return 0;
    }
    const uint32_t source_group = source_device + 1 == ring_size ? current_group - 1 : current_group;
    return source_group * q_local_tile_rows + q_local_tile_rows - halo_tile_rows;
}

// Shift a halo worker's page range from the group baked at program-create time to the group this chunk
// actually needs. Mirrors the host relocate_pages loop in apply_ring_joint_scalar_runtime_args; the
// delta is signed because a replay can run a chunk either side of the capturing one.
inline void relocate_halo_range(
    uint32_t runtime_origin_page, uint32_t baked_origin_page, uint32_t& tile_start, uint32_t& tile_end) {
    if (runtime_origin_page == baked_origin_page) {
        return;
    }
    if (runtime_origin_page > baked_origin_page) {
        const uint32_t delta = runtime_origin_page - baked_origin_page;
        tile_start += delta;
        tile_end += delta;
    } else {
        const uint32_t delta = baked_origin_page - runtime_origin_page;
        tile_start = tile_start > delta ? tile_start - delta : 0;
        tile_end = tile_end > delta ? tile_end - delta : 0;
    }
}

// Clamp each input to the valid slab prefix, then repartition that prefix across links. Reader and
// writer must update both range endpoints identically or their cb_output page counts can diverge.
template <size_t NumInputs>
inline void update_link_page_ranges_for_gather_extent(
    uint32_t gather_valid_Ht,
    uint32_t num_links,
    const std::array<uint32_t, NumInputs>& input_tensor_Wt,
    const std::array<uint32_t, NumInputs>& input_valid_pages,
    const std::array<uint32_t, NumInputs>& worker_link,
    std::array<uint32_t, NumInputs>& input_tile_id_start,
    std::array<uint32_t, NumInputs>& input_tile_id_end) {
    for (uint32_t input_idx = 0; input_idx < NumInputs; input_idx++) {
        const uint32_t gather_valid_pages = gather_valid_Ht * input_tensor_Wt[input_idx];
        const uint32_t valid_pages =
            input_valid_pages[input_idx] < gather_valid_pages ? input_valid_pages[input_idx] : gather_valid_pages;
        const auto link_page_range = compute_link_page_range(valid_pages, num_links, worker_link[input_idx]);
        input_tile_id_start[input_idx] = link_page_range.start;
        input_tile_id_end[input_idx] = link_page_range.end;
    }
}

}  // namespace ring_attention_all_gather
