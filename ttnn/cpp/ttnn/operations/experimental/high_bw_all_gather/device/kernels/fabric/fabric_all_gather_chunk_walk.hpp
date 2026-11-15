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
//   page lane         Lane k of an outer slice is its pages k, k + num_dram_banks, k + 2 * num_dram_banks, ...
//                     (page_in_slice % num_dram_banks == k): there are num_dram_banks lanes. In an interleaved tensor
//                     each lane is stored in exactly one DRAM bank, its pages one after another, and different lanes
//                     are in different banks. Which bank holds lane k depends on where the slice starts (page
//                     slice_start + page_in_slice is in bank (slice_start + page_in_slice) % num_dram_banks), but the
//                     kernels never need it: they only rely on those two properties.
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
//   fabric chunk      Up to pages_per_fabric_chunk consecutive pages of one page lane of one outer slice: one run in
//                     one DRAM bank of an interleaved tensor, so one DRAM read and one fabric packet (a page larger
//                     than the payload is a chunk of its own, sent as several packets).
//   row block         pages_per_fabric_chunk rows of every lane: num_dram_banks * pages_per_fabric_chunk consecutive
//                     pages of an outer slice, one chunk per lane (only the last row block of a slice can be partial).
//   chunk number      The shard's fabric chunks numbered outer slice by outer slice, row block by row block, lane by
//                     lane: dense, 0 .. num_outer_slices * fabric_chunks_per_outer_slice - 1.
//   owned chunks      A core owns chunk numbers first_owned_chunk, first_owned_chunk + owned_chunk_stride, ... and
//                     walks them in that order (for_each_fabric_chunk), the same on every chip. Dealing chunks rather
//                     than whole lanes keeps the cores' chunk counts within one of each other for any bank count (a
//                     Blackhole with a harvested DRAM bank has 7). When the stride divides
//                     num_dram_banks a core keeps the same lanes through an outer slice; otherwise its lanes rotate
//                     from row block to row block. Either way the cores of one row block read different banks.
//   CB batch          chunks_per_cb_batch chunk slots of the chunk CB, reserved, read and handed over together. The CB
//                     holds two batches: the reader fills one while the sender (or copy writer) drains the other.
//   fabric link worker  One core per (ring, direction, link). Its reader reads this chip's own shard from the input,
//                     then forwards shards from this chip's output once they have arrived; its sender sends every chunk
//                     one fabric hop into the same pages of the next chip's output.
//   local copy core   Writes this chip's own shard into its own output, without the fabric: one per link.
//
// Protocol
//   outgoing shards   The shards a link worker sends, in order: make_outgoing_shard(rank, chunk_half). Outgoing shard 0
//                     is the chip's own shard; outgoing shard k >= 1 is forwarded: what upstream sent as its k - 1.
//                     chunk_half 1 / 2 = the even / odd positions among the owned chunks: on an even ring the
//                     opposite shard goes half each way.
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
//   input run entry   What a local copy core's reader tells its copy writer about each run it read into the chunk CB,
//                     on input_run_cb (one 16 B page per run, two words used):
//                       [0] flat_page: the run's first page in the own shard, numbered over the valid pages only
//                           (outer_slice * valid_pages_per_outer_slice + page_in_slice);
//                       [1] num_pages in bits 0..30 (at most pages_per_fabric_chunk), and in bit 31 whether this
//                           is the last run of its conversion block.
//                     The chunk CB carries only page bytes, and only the reader knows where runs start and end (it
//                     probes physical contiguity through the input accessor), so the writer takes both from here: the
//                     position to place the run's pages in the output, and the block end, where it waits for its
//                     writes to land and then signals the blocks-converted semaphores.
//   conversion block  The unit a local copy core converts and reports: copy core c converts blocks c, c + n, ...
//   blocks-converted semaphore  Per local copy core: how many of its conversion blocks are in the output. A link worker
//                     waits for the blocks a chunk of its own shard touches.

#include <cstdint>

namespace ttnn::operations::experimental::high_bw_all_gather::chunk_walk {

constexpr uint32_t kWholeChipShard = 0, kFirstChunkHalf = 1, kSecondChunkHalf = 2;

constexpr uint32_t make_outgoing_shard(uint32_t rank, uint32_t chunk_half) { return rank | (chunk_half << 16); }
constexpr uint32_t outgoing_shard_rank(uint32_t outgoing_shard) { return outgoing_shard & 0xFFFF; }
constexpr uint32_t outgoing_shard_chunk_half(uint32_t outgoing_shard) { return outgoing_shard >> 16; }

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

// Chunk numbers per outer slice. Every row block but the last is whole (num_dram_banks chunks); the last one has a
// chunk in lanes 0 .. k - 1 only, so the numbers stay dense.
inline uint32_t fabric_chunks_per_outer_slice(
    const ChipShardGeometry& geometry, uint32_t num_dram_banks, uint32_t pages_per_fabric_chunk) {
    const uint32_t valid_pages = geometry.valid_pages_per_outer_slice;
    if (valid_pages == 0) {
        return 0;
    }
    const uint32_t pages_per_row_block = num_dram_banks * pages_per_fabric_chunk;
    const uint32_t last_row_block = (valid_pages - 1) / pages_per_row_block;
    const uint32_t pages_in_last_row_block = valid_pages - last_row_block * pages_per_row_block;
    return last_row_block * num_dram_banks +
           (pages_in_last_row_block < num_dram_banks ? pages_in_last_row_block : num_dram_banks);
}

// The chunk numbers a chunk half covers: every owned chunk, or every other one (even / odd positions).
struct OwnedChunks {
    uint32_t first;
    uint32_t stride;
};
inline OwnedChunks owned_chunks_of_half(uint32_t first_owned_chunk, uint32_t owned_chunk_stride, uint32_t chunk_half) {
    if (chunk_half == kWholeChipShard) {
        return {first_owned_chunk, owned_chunk_stride};
    }
    return {first_owned_chunk + (chunk_half == kSecondChunkHalf ? owned_chunk_stride : 0), 2 * owned_chunk_stride};
}

// Walks the owned fabric chunks of one shard (or chunk half) in chunk number order: outer slice by outer slice, row
// block by row block, lane by lane. visit(outer_slice, first_page_in_slice, num_pages).
template <typename Visit>
inline void for_each_fabric_chunk(
    const ChipShardGeometry& geometry,
    uint32_t num_dram_banks,
    uint32_t pages_per_fabric_chunk,
    uint32_t first_owned_chunk,
    uint32_t owned_chunk_stride,
    uint32_t chunk_half,
    const Visit& visit) {
    const uint32_t chunks_per_slice = fabric_chunks_per_outer_slice(geometry, num_dram_banks, pages_per_fabric_chunk);
    const uint32_t num_chunks = chunks_per_slice * geometry.num_outer_slices;
    const OwnedChunks owned = owned_chunks_of_half(first_owned_chunk, owned_chunk_stride, chunk_half);
    if (owned.first >= num_chunks) {
        return;
    }
    const uint32_t valid_pages = geometry.valid_pages_per_outer_slice;
    const uint32_t pages_per_row_block = num_dram_banks * pages_per_fabric_chunk;
    // (outer slice, chunk in slice, row block, lane) of the current chunk, advanced without a division per chunk
    uint32_t outer_slice = owned.first / chunks_per_slice, chunk_in_slice = owned.first % chunks_per_slice;
    uint32_t row_block = chunk_in_slice / num_dram_banks, lane = chunk_in_slice % num_dram_banks;
    for (uint32_t chunk = owned.first; chunk < num_chunks; chunk += owned.stride) {
        const uint32_t first_page_in_slice = row_block * pages_per_row_block + lane;
        const uint32_t pages_left_in_lane = (valid_pages - first_page_in_slice + num_dram_banks - 1) / num_dram_banks;
        visit(
            outer_slice,
            first_page_in_slice,
            pages_left_in_lane < pages_per_fabric_chunk ? pages_left_in_lane : pages_per_fabric_chunk);
        chunk_in_slice += owned.stride;
        lane += owned.stride;
        if (chunk_in_slice >= chunks_per_slice) {  // a later outer slice (the stride can skip a short one)
            const uint32_t next_chunk = chunk + owned.stride;
            outer_slice = next_chunk / chunks_per_slice;
            chunk_in_slice = next_chunk % chunks_per_slice;
            row_block = chunk_in_slice / num_dram_banks;
            lane = chunk_in_slice % num_dram_banks;
        } else {
            while (lane >= num_dram_banks) {
                lane -= num_dram_banks;
                ++row_block;
            }
        }
    }
}

// Number of chunks for_each_fabric_chunk visits. The sender relies on it to find each outgoing shard's last chunk, so
// the two must agree.
inline uint32_t count_fabric_chunks(
    const ChipShardGeometry& geometry,
    uint32_t num_dram_banks,
    uint32_t pages_per_fabric_chunk,
    uint32_t first_owned_chunk,
    uint32_t owned_chunk_stride,
    uint32_t chunk_half) {
    const uint32_t num_chunks =
        fabric_chunks_per_outer_slice(geometry, num_dram_banks, pages_per_fabric_chunk) * geometry.num_outer_slices;
    const OwnedChunks owned = owned_chunks_of_half(first_owned_chunk, owned_chunk_stride, chunk_half);
    return owned.first < num_chunks ? (num_chunks - owned.first + owned.stride - 1) / owned.stride : 0;
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

}  // namespace ttnn::operations::experimental::high_bw_all_gather::chunk_walk
