// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// fabric_all_gather terms (used by the factory and all kernels)
//
//   ring         Chips the gather travels along (closed or open). Each chip sends forward (to the next chip) and
//                backward. For one direction, "upstream" is the neighbour that sends to this chip.
//   rank         A chip's slot in the gathered output (0 .. num_ranks - 1).
//   shard        The pages one chip contributes: num_stripes stripes of stripe_pages pages, of which the first
//                active_stripe_pages are moved (partial gathers). Output page of (rank, stripe, page) =
//                (stripe * num_ranks + rank) * stripe_pages + page. A gather along the tile width has one stripe per
//                tile row; any other gather has one stripe.
//   chunk        Pages page, page + num_banks, page + 2 * num_banks, ... of one stripe sit next to each other in one
//                DRAM bank (interleaved tensors), so up to pages_per_chunk of them are one DRAM read and one fabric
//                packet. A core owns the banks first_bank, first_bank + bank_stride, ... and walks their chunks in a
//                fixed order (for_each_chunk), the same on every chip.
//   link worker  One core per (ring, direction, link). Its reader reads this chip's own shard from the input, then
//                relays shards from this chip's output once they have arrived; its sender sends every chunk one
//                fabric hop into the same pages of the next chip's output.
//   copy core    Writes this chip's own shard into its own output: one per link. A non-interleaved input (e.g. the
//                ND-sharded KV cache) has no contiguous chunks, so two per link convert it in input order and the
//                link workers then read their own shard from the output (for_each_input_run).
//   send list    The shards a link worker sends, in order: entry = rank | half << 16, entry 0 = its own shard.
//                half 1 / 2 = the first / second half of its banks: on an even ring the opposite shard goes half each
//                way. Relay entry k is what upstream sent as its entry k - 1.
//   arrival counter  +1 when an upstream entry has fully landed (fused onto its last packet). The reader waits for
//                k before relaying entry k; the sender waits for all of upstream's entries at the end, then resets it.
//   ready counter    Fence between calls: before sending, a sender waits until its downstream has signalled that it
//                has started this call (so a reused output is never overwritten early).

#include <cstdint>

namespace ttnn::operations::experimental::fabric_all_gather::chunk_walk {

constexpr uint32_t kWholeShard = 0, kFirstHalf = 1, kSecondHalf = 2;

constexpr uint32_t make_send_entry(uint32_t rank, uint32_t half) { return rank | (half << 16); }
constexpr uint32_t entry_rank(uint32_t entry) { return entry & 0xFFFF; }
constexpr uint32_t entry_half(uint32_t entry) { return entry >> 16; }

struct ShardGeometry {
    uint32_t num_stripes;
    uint32_t active_stripe_pages;
    uint32_t stripe_pages;
    uint32_t num_ranks;
};

inline uint32_t output_page(const ShardGeometry& shard, uint32_t rank, uint32_t stripe, uint32_t page) {
    return (stripe * shard.num_ranks + rank) * shard.stripe_pages + page;
}

// The owned banks a half covers: indices [begin, end) of first_bank, first_bank + bank_stride, ...
inline uint32_t owned_banks(uint32_t num_banks, uint32_t first_bank, uint32_t bank_stride) {
    return (num_banks - first_bank + bank_stride - 1) / bank_stride;
}
inline uint32_t half_begin(uint32_t num_owned, uint32_t half) { return half == kSecondHalf ? num_owned / 2 : 0; }
inline uint32_t half_end(uint32_t num_owned, uint32_t half) { return half == kFirstHalf ? num_owned / 2 : num_owned; }

// Walks the chunks of one shard (or half): stripe by stripe, the first chunk of every owned bank, then the second,
// ... visit(stripe, first_page, num_pages).
template <typename Visit>
inline void for_each_chunk(
    const ShardGeometry& shard,
    uint32_t num_banks,
    uint32_t pages_per_chunk,
    uint32_t first_bank,
    uint32_t bank_stride,
    uint32_t half,
    Visit&& visit) {
    const uint32_t num_owned = owned_banks(num_banks, first_bank, bank_stride);
    for (uint32_t stripe = 0; stripe < shard.num_stripes; ++stripe) {
        for (uint32_t row = 0; row * num_banks < shard.active_stripe_pages; row += pages_per_chunk) {
            for (uint32_t i = half_begin(num_owned, half); i < half_end(num_owned, half); ++i) {
                const uint32_t first_page = first_bank + i * bank_stride + row * num_banks;
                if (first_page >= shard.active_stripe_pages) {
                    continue;
                }
                const uint32_t pages_left = (shard.active_stripe_pages - first_page + num_banks - 1) / num_banks;
                visit(stripe, first_page, pages_left < pages_per_chunk ? pages_left : pages_per_chunk);
            }
        }
    }
}

// Number of chunks for_each_chunk visits (closed form: walking the shard on device just to count costs ~10 us).
inline uint32_t count_chunks(
    const ShardGeometry& shard,
    uint32_t num_banks,
    uint32_t pages_per_chunk,
    uint32_t first_bank,
    uint32_t bank_stride,
    uint32_t half) {
    const uint32_t num_owned = owned_banks(num_banks, first_bank, bank_stride);
    uint32_t per_stripe = 0;
    for (uint32_t i = half_begin(num_owned, half); i < half_end(num_owned, half); ++i) {
        const uint32_t bank = first_bank + i * bank_stride;
        const uint32_t pages =
            bank < shard.active_stripe_pages ? (shard.active_stripe_pages - bank + num_banks - 1) / num_banks : 0;
        per_stripe += (pages + pages_per_chunk - 1) / pages_per_chunk;
    }
    return per_stripe * shard.num_stripes;
}

// Active pages per stripe when the extent comes from a block-cyclic prefix (trace-safe path): the populated prefix
// rounded up to whole slabs, clamped to the full extent (same as high_bw_all_gather).
inline uint32_t active_stripe_pages_from_prefix(
    uint32_t prefix_start, uint32_t slab, uint32_t full_extent, uint32_t pages_per_slab) {
    if (slab == 0) {
        return 0;
    }
    const uint32_t rounded = ((prefix_start + slab + slab - 1) / slab) * slab;
    return ((rounded < full_extent ? rounded : full_extent) / slab) * pages_per_slab;
}

}  // namespace ttnn::operations::experimental::fabric_all_gather::chunk_walk
