// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The all_gather walk's integer rules, shared verbatim by the kernels (via gather_walk.hpp) and the host
// (all_gather_geometry.cpp, the host gtest). Macro-free and dependency-free on purpose: only <cstdint>, every
// function constexpr, so both sides compile the same text and nothing is derived twice.
//
// Units (see gather_walk.hpp for the full glossary):
//   walk page -- min(input page, output page): the k = 1 chunk.
//   chunk     -- a walk page / k, the transfer unit. Chunk ids count chunks.
//   slot      -- a CB slot: chunk bytes, or the output's aligned page under the padded-slot policy.
//   stride    -- lanes per tile, in walk pages. xfer -- chunks per column.

#include <cstdint>

// Scatter limits of NocUnicastScatterCommandHeader (tt_metal/fabric/fabric_edm_packet_header.hpp,
// NOC_SCATTER_WRITE_{MAX,MIN}_CHUNKS). Mirrored here so this header stays dependency-free; gather_fabric.hpp,
// which includes the fabric header, static_asserts that they still match.
constexpr uint32_t gather_scatter_max_chunks = 4;
constexpr uint32_t gather_scatter_min_chunks = 2;

namespace gather_rules_detail {
constexpr uint32_t gcd(uint32_t a, uint32_t b) {
    while (b != 0) {
        const uint32_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}
// Chunk id of lane `lane`'s position q (page-major, sub-chunk-minor) in the walk from `first`.
constexpr uint32_t chunk_at(uint32_t first, uint32_t lane, uint32_t q, uint32_t stride, uint32_t k) {
    return (first / k + lane + (q / k) * stride) * k + q % k;
}
// Positions every lane holds: lane stride - 1 is the shortest.
constexpr uint32_t min_lane_len(uint32_t count, uint32_t stride, uint32_t k) { return k * ((count / k) / stride); }
constexpr uint32_t lane_len(uint32_t count, uint32_t stride, uint32_t k, uint32_t lane) {
    const uint32_t pages = count / k;
    return lane < pages ? k * ((pages - lane + stride - 1) / stride) : 0u;
}
}  // namespace gather_rules_detail

////////////////////////////////////////////////////////////////
// Transfer size
////////////////////////////////////////////////////////////////

// Units of `unit_size` bytes (a chunk, or a padded slot) in one transfer: the most that fits a packet, a single
// NOC command (`burst` = NOC_MAX_BURST_SIZE / hal::get_noc_max_burst_size_bytes()) and the host's run cap
// (0 = no cap). At least 1: a unit bigger than the packet is split by the packer.
constexpr uint32_t chunks_per_transfer(
    uint32_t packet_size, uint32_t unit_size, uint32_t run_cap_bytes, uint32_t burst) {
    uint32_t cap = packet_size < burst ? packet_size : burst;
    if (run_cap_bytes != 0 && run_cap_bytes < cap) {
        cap = run_cap_bytes;
    }
    return cap < unit_size ? 1u : cap / unit_size;
}

////////////////////////////////////////////////////////////////
// Scatter-segment cap (P6)
////////////////////////////////////////////////////////////////

// Worst-case walk pages one column touches. Tiles start page-aligned, so a column starts at sub-chunk
// (t * xfer) % k, which is a multiple of gcd(xfer, k): at most k - gcd into a page.
constexpr uint32_t column_pages(uint32_t xfer, uint32_t k) {
    return (xfer + k - gather_rules_detail::gcd(xfer, k) + k - 1) / k;
}

// Worst-case runs (scatter segments) in one column when a lane's run is guaranteed only `contig_pages`
// lane pages at a time (at an unknown phase); contig_pages == 0 means unbounded.
constexpr uint32_t column_segments(uint32_t xfer, uint32_t k, uint32_t contig_pages) {
    if (contig_pages == 0) {
        return 1;
    }
    const uint32_t pages = column_pages(xfer, k);
    return (pages - 1 + contig_pages - 1) / contig_pages + 1;
}

// The simple form: without cross-page runs every walk page in a column is its own segment, so
// xfer <= max_segments * k. Cross-page runs leave xfer uncapped here (0 = no cap).
constexpr uint32_t xfer_cap_for_segments(
    uint32_t k, bool cross_page, uint32_t max_segments = gather_scatter_max_chunks) {
    return cross_page ? 0u : max_segments * k;
}

// The general form: the largest xfer <= xfer_max whose worst column holds <= max_segments runs, given
// `contig_pages` (1 = no cross-page runs; the shortest guaranteed lane run otherwise; 0 = unbounded).
constexpr uint32_t xfer_for_segments(
    uint32_t xfer_max, uint32_t k, uint32_t contig_pages, uint32_t max_segments = gather_scatter_max_chunks) {
    for (uint32_t x = xfer_max; x > 1; --x) {
        if (column_segments(x, k, contig_pages) <= max_segments) {
            return x;
        }
    }
    return 1;
}

////////////////////////////////////////////////////////////////
// k
////////////////////////////////////////////////////////////////

// The smallest power-of-2 k that fills a transfer to >= fill_num / fill_den while the chunk stays a whole
// number of NoC words (chunk % max(read_align, 16) == 0, chunk * k == page). Falls back to the best-filling
// valid k. `packet_size` is the transfer budget: pass min(packet, run cap) so a run cap is respected.
// Fill = floor(cap / chunk) * chunk / cap; a chunk larger than the cap is split by the packer, fill =
// chunk / (packets * cap).
constexpr uint32_t choose_k(
    uint32_t page,
    uint32_t packet_size,
    uint32_t burst,
    uint32_t read_align,
    uint32_t max_k = 64,
    uint32_t fill_num = 9,
    uint32_t fill_den = 10) {
    const uint32_t align = read_align > 16 ? read_align : 16;
    const uint32_t cap = packet_size < burst ? packet_size : burst;
    uint32_t best_k = 1;
    uint64_t best_num = 0, best_den = 1;
    for (uint32_t k = 1; k <= max_k; k *= 2) {
        if (page % k != 0 || (page / k) % align != 0) {
            break;  // halving the chunk only makes alignment worse
        }
        const uint32_t chunk = page / k;
        const uint64_t num = chunk <= cap ? static_cast<uint64_t>(cap / chunk) * chunk : chunk;
        const uint64_t den = chunk <= cap ? cap : static_cast<uint64_t>((chunk + cap - 1) / cap) * cap;
        if (num * fill_den >= den * fill_num) {
            return k;
        }
        if (num * best_den > best_num * den) {
            best_k = k;
            best_num = num;
            best_den = den;
        }
    }
    return best_k;
}

// A legal k for `page`: a power of 2 whose chunk is a whole number of NoC words.
constexpr bool k_is_valid(uint32_t k, uint32_t page, uint32_t read_align) {
    const uint32_t align = read_align > 16 ? read_align : 16;
    return k != 0 && (k & (k - 1)) == 0 && page % k == 0 && (page / k) % align == 0;
}

////////////////////////////////////////////////////////////////
// Run sources
////////////////////////////////////////////////////////////////

// Whether a tensor's contiguity lines up with the walk, so a run can hold more than one chunk.
//   whole_page -- a run may go to the end of the accessor page: the walk page is the accessor page
//                 (cpp == k), or the walk is stride 1. Otherwise (a split input page walked at stride > 1)
//                 a run stops at the walk page's end. A page's own bytes are contiguous whether or not it
//                 is padded, so in-page runs need no `packed`.
//   cross_page -- a run may carry on into the walk's next page: the next lane page is the accessor's next
//                 contiguous page. Needs `packed` (a run steps by the aligned page while the CB steps by the
//                 slot, so they must be equal).
struct RunSource {
    bool whole_page;
    bool cross_page;
};

// `k` and `cpp` (accessor chunks per accessor page, a multiple of k); `stride` is the walk's (walk pages),
// `page_stride` the accessor's contiguous_page_stride() (accessor pages).
constexpr RunSource run_source(bool packed, uint32_t k, uint32_t cpp, uint32_t page_stride, uint32_t stride) {
    const bool whole_page = cpp == k || stride == 1;
    const bool cross_page = packed && (cpp == k ? page_stride == stride : (stride == 1 && page_stride == 1));
    return {whole_page, cross_page};
}

// The CT encoding of a RunSource (the host computes it; kernels decode it).
constexpr uint32_t run_source_bits(RunSource s) { return (s.whole_page ? 1u : 0u) | (s.cross_page ? 2u : 0u); }
constexpr RunSource run_source_from_bits(uint32_t bits) { return {(bits & 1u) != 0, (bits & 2u) != 0}; }

////////////////////////////////////////////////////////////////
// Resume
////////////////////////////////////////////////////////////////

// Where a walk (re)starts. Host-computed by walk_seek(); a walk from seqno 0 is walk_start(first).
struct WalkPos {
    uint32_t base;  // chunk id of lane 0 at the tile's first position (the tile's anchor)
    uint32_t slot;  // lane slot in the tile's lane order (lane = slot + rot, wrapped; rot = 0 off full tiles)
    uint32_t i;     // chunks into the column
};

constexpr WalkPos walk_start(uint32_t first) { return {first, 0, 0}; }

// Seqnos in the walk's full tiles: they form a prefix, and only they are rotated.
constexpr uint32_t full_tile_seqnos(uint32_t count, uint32_t stride, uint32_t xfer, uint32_t k) {
    return (gather_rules_detail::min_lane_len(count, stride, k) / xfer) * stride * xfer;
}

// The WalkPos of seqno `skip` in the walk of [first, first + count). first and count are whole walk pages.
// Full tiles in closed form; the clipped tail by a loop. skip == count gives a position past the end.
// The slot is rot-relative, so the result does not depend on rot.
constexpr WalkPos walk_seek(uint32_t first, uint32_t count, uint32_t skip, uint32_t stride, uint32_t xfer, uint32_t k) {
    using namespace gather_rules_detail;
    const uint32_t full = min_lane_len(count, stride, k) / xfer;
    const uint32_t tile = stride * xfer;
    if (skip < full * tile) {
        const uint32_t t = skip / tile;
        const uint32_t r = skip - t * tile;
        return {chunk_at(first, 0, t * xfer, stride, k), r / xfer, r % xfer};
    }
    const uint32_t lane0 = lane_len(count, stride, k, 0);
    uint32_t s = skip - full * tile;
    for (uint32_t t = full;; ++t) {
        const uint32_t q0 = t * xfer;
        if (q0 >= lane0) {
            return {chunk_at(first, 0, q0, stride, k), 0, 0};
        }
        for (uint32_t l = 0; l < stride; ++l) {
            const uint32_t len = lane_len(count, stride, k, l);
            const uint32_t h = len > q0 ? (len - q0 < xfer ? len - q0 : xfer) : 0u;
            if (s < h) {
                return {chunk_at(first, 0, q0, stride, k), l, s};
            }
            s -= h;
        }
    }
}

////////////////////////////////////////////////////////////////
// Rotation (host policy; the kernels only take the resulting rot)
////////////////////////////////////////////////////////////////

// The absolute lane slice `slice` of `num_slices` should start its full tiles on: slices spread evenly over
// the stride's lanes, so concurrent workers open different banks. `stagger_dir` offsets direction 1 by half a
// slice spacing (at least one lane), so a slice's two directions do not start on the same bank either.
constexpr uint32_t rot_target(uint32_t slice, uint32_t num_slices, uint32_t stride, bool stagger_dir, uint32_t dir) {
    if (stride <= 1 || num_slices == 0) {
        return 0;
    }
    uint64_t t = (static_cast<uint64_t>(slice) * stride) / num_slices;
    if (stagger_dir && dir != 0) {
        const uint32_t half = stride / (2 * num_slices);
        t += half > 0 ? half : 1u;
    }
    return static_cast<uint32_t>(t % stride);
}

// The rot that starts a walk whose first walk page is `first_page` on absolute lane `target`.
constexpr uint32_t rot_for_target(uint32_t target, uint32_t first_page, uint32_t stride) {
    return stride <= 1 ? 0u : (target % stride + stride - first_page % stride) % stride;
}

// Even ring: forward emits seqnos [0, h) of its walk (rot_f), backward [h, count) of its own (rot_b). The two
// partition the slice iff rot_f == rot_b, h is on a full-tile boundary, or h is inside the unrotated tail.
constexpr bool antipode_split_ok(
    uint32_t count, uint32_t h, uint32_t stride, uint32_t xfer, uint32_t k, uint32_t rot_f, uint32_t rot_b) {
    if (rot_f == rot_b) {
        return true;
    }
    const uint32_t tile = stride * xfer;
    return h % tile == 0 || h >= full_tile_seqnos(count, stride, xfer, k);
}

// rot_b, or rot_f where rot_b would break the even-ring split.
constexpr uint32_t antipode_rot_b(
    uint32_t count, uint32_t h, uint32_t stride, uint32_t xfer, uint32_t k, uint32_t rot_f, uint32_t rot_b) {
    return antipode_split_ok(count, h, stride, xfer, k, rot_f, rot_b) ? rot_b : rot_f;
}
