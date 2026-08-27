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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace ring_attention_all_gather {

struct LinkPageRange {
    uint32_t start;
    uint32_t end;
};

struct BankOwnedSlice {
    uint32_t first_page_offset;
    uint32_t page_count;
};

FORCE_INLINE BankOwnedSlice
get_bank_owned_slice(uint32_t output_page_base, uint32_t valid_pages, uint32_t bank, uint32_t num_dram_banks) {
    const uint32_t base_bank = output_page_base % num_dram_banks;
    const uint32_t first_page_offset = (bank + num_dram_banks - base_bank) % num_dram_banks;
    const uint32_t page_count =
        first_page_offset < valid_pages ? 1 + (valid_pages - 1 - first_page_offset) / num_dram_banks : 0;
    return {first_page_offset, page_count};
}

// Keep each packet physically contiguous within one owned bank, but rotate banks between packets so one reader
// prefetch window preserves DRAM-bank parallelism.
template <size_t NumDramBanks>
struct BankOwnedPacketSchedule {
    std::array<BankOwnedSlice, NumDramBanks> slices{};
    std::array<uint32_t, NumDramBanks> pages_consumed{};
    uint32_t first_bank;
    uint32_t bank_stride;
    uint32_t next_bank;
    uint32_t packet_pages;
    uint32_t packets_remaining = 0;

    FORCE_INLINE BankOwnedPacketSchedule(
        uint32_t output_page_base,
        uint32_t valid_pages,
        uint32_t worker_first_bank,
        uint32_t worker_bank_stride,
        uint32_t packet_size_in_pages) :
        first_bank(worker_first_bank),
        bank_stride(worker_bank_stride),
        next_bank(worker_first_bank),
        packet_pages(packet_size_in_pages) {
        for (uint32_t bank = first_bank; bank < NumDramBanks; bank += bank_stride) {
            slices[bank] = get_bank_owned_slice(output_page_base, valid_pages, bank, NumDramBanks);
            packets_remaining += (slices[bank].page_count + packet_size_in_pages - 1) / packet_size_in_pages;
        }
    }

    FORCE_INLINE bool next_packet(uint32_t& first_page_offset, uint32_t& page_count) {
        if (packets_remaining == 0) {
            return false;
        }
        for (uint32_t candidate = 0; candidate < NumDramBanks; ++candidate) {
            const uint32_t bank = next_bank;
            next_bank += bank_stride;
            if (next_bank >= NumDramBanks) {
                next_bank = first_bank;
            }
            if (pages_consumed[bank] == slices[bank].page_count) {
                continue;
            }
            page_count = std::min(packet_pages, slices[bank].page_count - pages_consumed[bank]);
            first_page_offset = slices[bank].first_page_offset + pages_consumed[bank] * NumDramBanks;
            pages_consumed[bank] += page_count;
            --packets_remaining;
            return true;
        }
        return false;
    }
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
inline uint32_t compute_gather_valid_Ht(uint32_t kv_actual_isl, uint32_t chunk_local_tiles, uint32_t ring_size) {
    const uint32_t chunk_global_tiles = chunk_local_tiles * ring_size;
    const uint32_t logical_nt_local = (kv_actual_isl >> 5) + chunk_global_tiles;
    const uint32_t valid_slabs = (logical_nt_local + chunk_global_tiles - 1) / chunk_global_tiles;
    return valid_slabs * chunk_local_tiles;
}

// Clamp each input to the valid slab prefix, then repartition that prefix across links. Reader and
// writer must update the effective page count and both range endpoints identically or their cb_output
// page counts can diverge. The bank-owned path consumes the effective page count directly.
template <size_t NumInputs>
inline void update_link_page_ranges_for_gather_extent(
    uint32_t gather_valid_Ht,
    uint32_t num_links,
    const std::array<uint32_t, NumInputs>& input_tensor_Wt,
    std::array<uint32_t, NumInputs>& input_valid_pages,
    const std::array<uint32_t, NumInputs>& worker_link,
    std::array<uint32_t, NumInputs>& input_tile_id_start,
    std::array<uint32_t, NumInputs>& input_tile_id_end) {
    for (uint32_t input_idx = 0; input_idx < NumInputs; input_idx++) {
        const uint32_t gather_valid_pages = gather_valid_Ht * input_tensor_Wt[input_idx];
        const uint32_t valid_pages =
            input_valid_pages[input_idx] < gather_valid_pages ? input_valid_pages[input_idx] : gather_valid_pages;
        input_valid_pages[input_idx] = valid_pages;
        const auto link_page_range = compute_link_page_range(valid_pages, num_links, worker_link[input_idx]);
        input_tile_id_start[input_idx] = link_page_range.start;
        input_tile_id_end[input_idx] = link_page_range.end;
    }
}

}  // namespace ring_attention_all_gather
