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

// Shrink each input's page range to the valid slab prefix. Never grows a range: a caller that already
// has a tighter host-provided end keeps it.
template <size_t NumInputs>
inline void clamp_input_ranges_to_gather_extent(
    uint32_t gather_valid_Ht,
    const std::array<uint32_t, NumInputs>& input_tensor_Ht,
    const std::array<uint32_t, NumInputs>& input_tensor_Wt,
    std::array<uint32_t, NumInputs>& input_tile_id_end) {
    for (uint32_t input_idx = 0; input_idx < NumInputs; input_idx++) {
        const uint32_t valid_Ht =
            gather_valid_Ht < input_tensor_Ht[input_idx] ? gather_valid_Ht : input_tensor_Ht[input_idx];
        const uint32_t valid_pages = valid_Ht * input_tensor_Wt[input_idx];
        if (valid_pages < input_tile_id_end[input_idx]) {
            input_tile_id_end[input_idx] = valid_pages;
        }
    }
}

}  // namespace ring_attention_all_gather
