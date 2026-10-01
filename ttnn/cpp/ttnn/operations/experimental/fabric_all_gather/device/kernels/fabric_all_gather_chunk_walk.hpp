// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// fabric_all_gather terms (used by the factory and every kernel)
//
// Logical vs physical
//   logical page      A page's index in the tensor (one row-major row, or one tile). "Contiguous" without a qualifier
//                     means logically contiguous: consecutive page indices.
//   physical location Where a page sits in DRAM: a bank and an address in it. Logically consecutive pages are NOT
//                     physically consecutive in general, and how they map depends on the memory layout:
//                       interleaved  page p is in physical bank p % num_dram_banks, at row p / num_dram_banks. So pages
//                                    p, p + num_dram_banks, p + 2 * num_dram_banks, ... are physically contiguous.
//                       ND-sharded   pages are grouped into shards of consecutive pages, each shard in one bank, so
//                                    logically consecutive pages are physically contiguous within a shard.
//   logical bank      page_in_slice % num_dram_banks. Within one outer slice of an interleaved tensor, the physical
//   bank
//                     is the logical bank rotated by where the slice starts, so different logical banks are different
//                     physical banks, and the pages of one logical bank are physically contiguous. "Bank" in names
//                     (first_owned_bank, owned_bank_stride, bank_half) means logical bank.
//
// What is gathered
//   ring              Chips the gather travels along (closed or open). Each chip sends forward (to the next chip) and
//                     backward. For one direction, "upstream" is the neighbour that sends to this chip.
//   rank              A chip's position in the gathered output (0 .. num_ranks - 1); not its position on the ring.
//   chip shard        The pages one chip contributes to the gather.
//   outer slice       A chip shard is num_outer_slices outer slices of pages_per_outer_slice pages: the dims before the
//                     gather dim index the outer slice, the gather dim and the dims after it index the page within it.
//                     The output interleaves ranks per outer slice: output page of (rank, outer_slice, page_in_slice) =
//                     (outer_slice * num_ranks + rank) * pages_per_outer_slice + page_in_slice. A gather along the tile
//                     width has one outer slice per tile row; most gathers (the KV cache) have one.
//   valid prefix      The first valid_pages_per_outer_slice pages of every outer slice: the part a partial gather moves
//                     (gathered_dim_size, or the prefix tensor). The gather dim is the slowest index within an outer
//                     slice, so a prefix of the gathered length is a prefix of pages.
//   cache slot        The batch index of a persistent cache that is gathered; its pages start at cache_slot_first_page.
//
// How it moves
//   fabric chunk      Up to pages_per_fabric_chunk pages of one logical bank of one outer slice: physically contiguous
//   in
//                     an interleaved tensor, so one DRAM read and one fabric packet (a page larger than the payload is
//                     a chunk of its own, sent as several packets). A core owns the logical banks first_owned_bank,
//                     first_owned_bank + owned_bank_stride, ... and walks their chunks in a fixed order
//                     (for_each_fabric_chunk), the same on every chip.
//   CB batch          chunks_per_cb_batch chunk slots of the chunk CB, reserved, read and handed over together. The CB
//                     holds two batches: the reader fills one while the sender (or copy writer) drains the other.
//   fabric link worker  One core per (ring, direction, link). Its reader reads this chip's own shard from the input,
//                     then forwards shards from this chip's output once they have arrived; its sender sends every chunk
//                     one fabric hop into the same pages of the next chip's output.
//   local copy core   Writes this chip's own shard into its own output, without the fabric: one per link.
//
// Protocol
//   outgoing shards   The shards a link worker sends, in order: make_outgoing_shard(rank, bank_half). Outgoing shard 0
//                     is the chip's own shard; outgoing shard k >= 1 is forwarded: what upstream sent as its k - 1.
//                     bank_half 1 / 2 = the first / second half of the owned banks: on an even ring the opposite shard
//                     goes half each way.
//   shards-arrived counter  +1 when one of upstream's outgoing shards has fully landed (fused onto its last packet).
//                     The reader waits for k before forwarding outgoing shard k; the sender waits for every shard
//                     expected from upstream at the end, then resets it.
//   downstream-started counter  Fence between calls: before sending, a sender waits until its downstream has
//                     signalled that it started this call, so a reused output is never overwritten early.
//
// ND-sharded (any non-interleaved) input: its fabric chunks are not physically contiguous, so the local copy cores
// (two per link) first convert the own shard into this chip's interleaved output, and the link workers read it from
// there.
//   contiguous input run  Up to pages_per_fabric_chunk logically consecutive pages that are also physically
//                     consecutive in the input: one DRAM read (for_each_contiguous_input_run).
//   conversion block  The unit a local copy core converts and reports: copy core c converts blocks c, c + n, ...
//   blocks-converted semaphore  Per local copy core: how many of its conversion blocks are in the output. A link worker
//                     waits for the blocks a chunk of its own shard touches.

#include <cstdint>

namespace ttnn::operations::experimental::fabric_all_gather::chunk_walk {

constexpr uint32_t kWholeChipShard = 0, kFirstBankHalf = 1, kSecondBankHalf = 2;

constexpr uint32_t make_outgoing_shard(uint32_t rank, uint32_t bank_half) { return rank | (bank_half << 16); }
constexpr uint32_t outgoing_shard_rank(uint32_t outgoing_shard) { return outgoing_shard & 0xFFFF; }
constexpr uint32_t outgoing_shard_bank_half(uint32_t outgoing_shard) { return outgoing_shard >> 16; }

struct ChipShardGeometry {
    uint32_t num_outer_slices;
    uint32_t valid_pages_per_outer_slice;
    uint32_t pages_per_outer_slice;
    uint32_t num_ranks;
};

inline uint32_t output_page_index(
    const ChipShardGeometry& geometry, uint32_t rank, uint32_t outer_slice, uint32_t page_in_slice) {
    return (outer_slice * geometry.num_ranks + rank) * geometry.pages_per_outer_slice + page_in_slice;
}

// The owned logical banks a bank half covers: indices [begin, end) of first_owned_bank, first_owned_bank +
// owned_bank_stride, ...
inline uint32_t num_owned_banks(uint32_t num_dram_banks, uint32_t first_owned_bank, uint32_t owned_bank_stride) {
    return (num_dram_banks - first_owned_bank + owned_bank_stride - 1) / owned_bank_stride;
}
inline uint32_t bank_half_begin(uint32_t num_owned, uint32_t bank_half) {
    return bank_half == kSecondBankHalf ? num_owned / 2 : 0;
}
inline uint32_t bank_half_end(uint32_t num_owned, uint32_t bank_half) {
    return bank_half == kFirstBankHalf ? num_owned / 2 : num_owned;
}

// Walks the fabric chunks of one shard (or bank half): outer slice by outer slice, the first chunk of every owned
// bank, then the second, ... visit(outer_slice, first_page_in_slice, num_pages).
template <typename Visit>
inline void for_each_fabric_chunk(
    const ChipShardGeometry& geometry,
    uint32_t num_dram_banks,
    uint32_t pages_per_fabric_chunk,
    uint32_t first_owned_bank,
    uint32_t owned_bank_stride,
    uint32_t bank_half,
    Visit&& visit) {
    const uint32_t num_owned = num_owned_banks(num_dram_banks, first_owned_bank, owned_bank_stride);
    for (uint32_t outer_slice = 0; outer_slice < geometry.num_outer_slices; ++outer_slice) {
        for (uint32_t bank_row = 0; bank_row * num_dram_banks < geometry.valid_pages_per_outer_slice;
             bank_row += pages_per_fabric_chunk) {
            for (uint32_t i = bank_half_begin(num_owned, bank_half); i < bank_half_end(num_owned, bank_half); ++i) {
                const uint32_t first_page_in_slice =
                    first_owned_bank + i * owned_bank_stride + bank_row * num_dram_banks;
                if (first_page_in_slice >= geometry.valid_pages_per_outer_slice) {
                    continue;
                }
                const uint32_t pages_left_in_bank =
                    (geometry.valid_pages_per_outer_slice - first_page_in_slice + num_dram_banks - 1) / num_dram_banks;
                visit(
                    outer_slice,
                    first_page_in_slice,
                    pages_left_in_bank < pages_per_fabric_chunk ? pages_left_in_bank : pages_per_fabric_chunk);
            }
        }
    }
}

// Number of chunks for_each_fabric_chunk visits (closed form: walking the shard on device just to count costs ~10 us).
// The sender relies on it to find each outgoing shard's last chunk, so the two must agree.
inline uint32_t count_fabric_chunks(
    const ChipShardGeometry& geometry,
    uint32_t num_dram_banks,
    uint32_t pages_per_fabric_chunk,
    uint32_t first_owned_bank,
    uint32_t owned_bank_stride,
    uint32_t bank_half) {
    const uint32_t num_owned = num_owned_banks(num_dram_banks, first_owned_bank, owned_bank_stride);
    uint32_t chunks_per_outer_slice = 0;
    for (uint32_t i = bank_half_begin(num_owned, bank_half); i < bank_half_end(num_owned, bank_half); ++i) {
        const uint32_t logical_bank = first_owned_bank + i * owned_bank_stride;
        const uint32_t pages =
            logical_bank < geometry.valid_pages_per_outer_slice
                ? (geometry.valid_pages_per_outer_slice - logical_bank + num_dram_banks - 1) / num_dram_banks
                : 0;
        chunks_per_outer_slice += (pages + pages_per_fabric_chunk - 1) / pages_per_fabric_chunk;
    }
    return chunks_per_outer_slice * geometry.num_outer_slices;
}

// Valid pages per outer slice when the valid length comes from a block-cyclic prefix (trace-safe path): the populated
// prefix (this prefill chunk's start + one KV slab) rounded up to whole KV slabs, clamped to the full gathered length
// (same as high_bw_all_gather).
inline uint32_t valid_pages_from_prefix_start(
    uint32_t prefix_start, uint32_t kv_slab, uint32_t full_gathered_length, uint32_t pages_per_kv_slab) {
    if (kv_slab == 0) {
        return 0;
    }
    const uint32_t rounded = ((prefix_start + kv_slab + kv_slab - 1) / kv_slab) * kv_slab;
    return ((rounded < full_gathered_length ? rounded : full_gathered_length) / kv_slab) * pages_per_kv_slab;
}

}  // namespace ttnn::operations::experimental::fabric_all_gather::chunk_walk
