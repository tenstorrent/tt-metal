// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Chunk walk and closed-form counts shared by the HOST factory and the DEVICE kernels.
//
// Page model. One chip's shard, in pages, is A stripes of B_max pages; B (<= B_max) pages of each stripe are active
// (a partial gather). Output page of (rank g, stripe a, page b) is a * G * B_max + g * B_max + b, and the input page is
// slot_base + a * B_max + b. For a gather along an outer dim (every dim before it 1) A = 1 and the shard is one
// contiguous block; for a gather along the tile width A is the number of tile rows.
//
// Chunks. Pages b = r, r + NB, r + 2 NB, ... of one stripe (NB = number of DRAM banks, r the residue) sit
// consecutively in one bank on both the input and the output side, so a run of up to `run` of them is one chunk: one
// contiguous read and one contiguous fabric packet. A link worker owns the residues first, first + stride, ...
// (its bank set, nb residues) and visits them round-robin starting at set index `rot`. `part` 0 walks the whole
// set, 1 its first half, 2 its second half (a shard split between the two directions of a ring).
//
// Every count is closed form (a loop over at most NB residues), so the kernels can derive them on device when the
// active extent is only known there (trace-safe metadata path).
//
// Dependency-free: included by kernels.

#include <cstdint>

namespace ttnn::operations::experimental::fabric_all_gather::walk {

constexpr uint32_t walk_min(uint32_t a, uint32_t b) { return a < b ? a : b; }

// Residues a link worker owns.
constexpr uint32_t num_residues(uint32_t first, uint32_t stride, uint32_t num_banks) {
    return first < num_banks ? (num_banks - first + stride - 1) / stride : 0;
}

// Pages of residue r in a stripe of `stripe_pages` active pages.
constexpr uint32_t residue_pages(uint32_t stripe_pages, uint32_t r, uint32_t num_banks) {
    return r < stripe_pages ? (stripe_pages - r + num_banks - 1) / num_banks : 0;
}

// Chunks of one shard walked by a link worker (all stripes).
constexpr uint32_t port_chunks(
    uint32_t num_stripes,
    uint32_t stripe_pages,
    uint32_t num_banks,
    uint32_t run_pages,
    uint32_t first,
    uint32_t stride,
    uint32_t part) {
    const uint32_t nb = num_residues(first, stride, num_banks);
    const uint32_t half = nb / 2;
    const uint32_t lo = part == 2 ? half : 0;
    const uint32_t hi = part == 1 ? half : nb;
    uint32_t chunks = 0;
    for (uint32_t slot = lo; slot < hi; ++slot) {
        const uint32_t pages = residue_pages(stripe_pages, first + slot * stride, num_banks);
        chunks += (pages + run_pages - 1) / run_pages;
    }
    return chunks * num_stripes;
}

// Increment cadence. A sender's packet carries an increment of the receiver's arrival counter when it is every
// inc_every-th chunk it sends OR the last chunk of an entry (a shard). The second rule keeps a relay's wait within the
// shard it forwards: relaying chunk c needs only the upstream chunks of the same shard up to the next increment, never
// chunks of a later shard (which may themselves wait on this chip -- a cycle when shards are shorter than inc_every).
//
// Entries e = 1..n ending at e * size (all whole shards of `size` chunks): how many end off the inc_every grid (those
// ends carry an extra increment).
constexpr uint32_t entry_end_increments(uint32_t n, uint32_t size, uint32_t inc_every) {
    uint32_t count = 0;
    for (uint32_t e = 1; e <= n; ++e) {
        count += (size > 0 && (e * size) % inc_every != 0) ? 1 : 0;
    }
    return count;
}

// Arrival count that proves upstream chunk c (0-based, in upstream send order) has landed, when c belongs to an entry
// ending at `entry_end` (one past its last chunk) and `prior_ends` off-grid entry ends precede that entry.
constexpr uint32_t increments_through(uint32_t c, uint32_t entry_end, uint32_t prior_ends, uint32_t inc_every) {
    const uint32_t grid_next = (c / inc_every + 1) * inc_every;  // one past the next grid increment
    const uint32_t through = grid_next < entry_end ? grid_next : entry_end;
    return through / inc_every + prior_ends + ((through == entry_end && entry_end % inc_every != 0) ? 1u : 0u);
}

// Increments a receiver gets from an upstream that sends `whole` whole shards of `full` chunks, then (optionally) one
// half of `half` chunks.
constexpr uint32_t expected_increments(
    uint32_t whole, uint32_t full, uint32_t half_entries, uint32_t half, uint32_t inc_every) {
    uint32_t count = 0, sent = 0;
    for (uint32_t e = 0; e < whole + half_entries; ++e) {
        const uint32_t size = e < whole ? full : half;
        if (size == 0) {
            continue;
        }
        count += (sent + size) / inc_every - sent / inc_every;  // grid increments inside this entry
        sent += size;
        count += sent % inc_every != 0 ? 1 : 0;  // the entry's last chunk, if off the grid
    }
    return count;
}

// Active pages per stripe of a partial gather whose extent comes from a block-cyclic prefix (trace-safe path):
// the populated prefix start + slab is rounded up to whole slabs and clamped to the full extent. Identical to
// high_bw_all_gather's partition::gathered_dim_size_for_prefix / active_num_input_pages.
constexpr uint32_t prefix_stripe_pages(
    uint32_t start_global, uint32_t slab_global, uint32_t full_global, uint32_t pages_per_slab) {
    if (slab_global == 0) {
        return 0;
    }
    const uint32_t populated = start_global + slab_global;
    const uint32_t rounded = ((populated + slab_global - 1) / slab_global) * slab_global;
    return (walk_min(rounded, full_global) / slab_global) * pages_per_slab;
}

// Visit the chunks of one shard for one link worker, in the order every chip uses.
// f(stripe, page, n, idx): stripe a, first page b of the run (stripe-local), page count n, and idx = the chunk's
// index in the part-0 walk of this shard (the relay order: upstream chunk idx lands as our relay chunk idx).
template <typename F>
inline void for_each_chunk(
    uint32_t num_stripes,
    uint32_t stripe_pages,
    uint32_t num_banks,
    uint32_t run_pages,
    uint32_t first,
    uint32_t stride,
    uint32_t rot,
    uint32_t part,
    F&& f) {
    const uint32_t nb = num_residues(first, stride, num_banks);
    const uint32_t half = nb / 2;
    const uint32_t lo = part == 2 ? half : 0;
    const uint32_t hi = part == 1 ? half : nb;
    const uint32_t max_pages = (stripe_pages + num_banks - 1) / num_banks;  // pages of the fullest residue
    uint32_t idx = 0;
    for (uint32_t a = 0; a < num_stripes; ++a) {
        for (uint32_t m = 0; m < max_pages; m += run_pages) {
            uint32_t slot = rot;
            for (uint32_t i = 0; i < nb; ++i, ++slot) {
                if (slot >= nb) {
                    slot -= nb;
                }
                const uint32_t r = first + slot * stride;
                const uint32_t pages = residue_pages(stripe_pages, r, num_banks);
                if (m >= pages) {
                    continue;
                }
                if (slot >= lo && slot < hi) {
                    f(a, r + m * num_banks, walk_min(run_pages, pages - m), idx);
                }
                ++idx;
            }
        }
    }
}

}  // namespace ttnn::operations::experimental::fabric_all_gather::walk
