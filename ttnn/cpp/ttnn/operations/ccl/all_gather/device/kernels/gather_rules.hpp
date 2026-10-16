// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The all_gather walk's integer rules, shared verbatim by the kernels (via gather_walk.hpp) and the host
// (all_gather_geometry.cpp, all_gather_unicast_factory.cpp, the host gtests). Macro-free and dependency-free on
// purpose: only <cstdint>, every function constexpr, so both sides compile the same text and nothing is derived
// twice.
//
// Units (see gather_walk.hpp for the full glossary):
//   walk page -- the k = 1 transfer unit: the input's aligned page (Matched, Concat) or the output's content page
//                (Split).
//   chunk     -- a walk page / k, the transfer unit. Chunk ids count chunks.
//   slot      -- a CB slot: chunk bytes, or the output's aligned page (padded slot).
//   stride    -- lanes per tile, in walk pages. xfer -- chunks per column.

#include <cstdint>

// Scatter limits of NocUnicastScatterCommandHeader (tt_metal/fabric/fabric_edm_packet_header.hpp,
// NOC_SCATTER_WRITE_{MAX,MIN}_CHUNKS). Mirrored here so this header stays dependency-free; gather_fabric.hpp,
// which includes the fabric header, static_asserts that they still match.
constexpr uint32_t gather_scatter_max_chunks = 4;
constexpr uint32_t gather_scatter_min_chunks = 2;

// NoC write alignment (NOC_{L1,DRAM}_WRITE_ALIGNMENT_BYTES), 16 B on Wormhole and Blackhole. Every chunk and slot
// is a multiple of it: a run is sent as one scatter chunk starting at its source offset in the packet, so source
// and destination must stay write-aligned. The host cannot query it from TTNN (Hal::get_write_alignment is not
// reachable there), so this one constant serves the host's layout check, choose_k's chunk floor and the kernels'
// static_asserts (check_walk_ct).
constexpr uint32_t gather_noc_write_alignment = 16;

namespace gather_rules_detail {
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
constexpr uint32_t chunk_align(uint32_t read_align) {
    return read_align > gather_noc_write_alignment ? read_align : gather_noc_write_alignment;
}
}  // namespace gather_rules_detail

////////////////////////////////////////////////////////////////
// xfer
////////////////////////////////////////////////////////////////

// The largest xfer for units of `unit_size` bytes (a chunk, or a padded slot): the most that fits a packet, a
// single NOC command (`burst` = NOC_MAX_BURST_SIZE / hal::get_noc_max_burst_size_bytes()) and the host's run cap
// (0 = no cap). At least 1: a unit bigger than the packet is split by the packer.
constexpr uint32_t max_xfer(uint32_t packet_size, uint32_t unit_size, uint32_t run_cap_bytes, uint32_t burst) {
    uint32_t cap = packet_size < burst ? packet_size : burst;
    if (run_cap_bytes != 0 && run_cap_bytes < cap) {
        cap = run_cap_bytes;
    }
    return cap < unit_size ? 1u : cap / unit_size;
}

////////////////////////////////////////////////////////////////
// k
////////////////////////////////////////////////////////////////

// The smallest power-of-2 k that fills a transfer to >= fill_num / fill_den while the chunk stays a whole
// number of NoC words (chunk % max(read_align, gather_noc_write_alignment) == 0, chunk * k == page). Falls back
// to the best-filling valid k. `packet_size` is the transfer budget: pass min(packet, run cap) so a run cap is
// respected. Fill = floor(cap / chunk) * chunk / cap; a chunk larger than the cap is split by the packer, fill =
// chunk / (packets * cap).
constexpr uint32_t choose_k(
    uint32_t page,
    uint32_t packet_size,
    uint32_t burst,
    uint32_t read_align,
    uint32_t max_k = 64,
    uint32_t fill_num = 9,
    uint32_t fill_den = 10) {
    const uint32_t align = gather_rules_detail::chunk_align(read_align);
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
    const uint32_t align = gather_rules_detail::chunk_align(read_align);
    return k != 0 && (k & (k - 1)) == 0 && page % k == 0 && (page / k) % align == 0;
}

////////////////////////////////////////////////////////////////
// Slot and stride
////////////////////////////////////////////////////////////////

// A CB slot is the chunk, or -- only for one chunk per output page at k = 1 -- the output's padded aligned page,
// so that L1 steps like the output. Checked by the host (WalkGeometry::make) and the kernels (check_walk_ct).
constexpr bool slot_is_valid(uint32_t chunk, uint32_t slot, uint32_t out_cpp, uint32_t k) {
    return slot == chunk || (slot > chunk && out_cpp == 1 && k == 1);
}

// The walk stride: walk pages between neighbours in memory. Concat packs several walk pages into one output page
// (out_cpp > k), so neighbours are one walk page apart; otherwise it is the output accessor's
// contiguous_page_stride(). The host derives it from its mirror of that value; the kernels ASSERT it against
// their accessor (check_walk_stride).
constexpr uint32_t walk_stride_for(uint32_t out_cpp, uint32_t k, uint32_t out_page_stride) {
    return out_cpp > k ? 1u : out_page_stride;
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

// The output's run source: packed when its aligned page is exactly its chunks' slots.
constexpr RunSource output_run_source(
    uint32_t out_aligned_page, uint32_t slot, uint32_t k, uint32_t out_cpp, uint32_t out_page_stride, uint32_t stride) {
    return run_source(out_aligned_page == out_cpp * slot, k, out_cpp, out_page_stride, stride);
}

// The input's run source. A run may merge chunks only where L1 (slot stride) and the input step alike: with a
// padded slot (slot != chunk) the chunks inside an input page sit at chunk stride, so there is one read per chunk,
// and in-page runs are off.
constexpr RunSource input_run_source(
    uint32_t in_aligned_page,
    uint32_t chunk,
    uint32_t slot,
    uint32_t k,
    uint32_t in_cpp,
    uint32_t in_page_stride,
    uint32_t stride) {
    if (slot != chunk && in_cpp != 1) {
        return {false, false};
    }
    RunSource s = run_source(in_aligned_page == in_cpp * slot, k, in_cpp, in_page_stride, stride);
    s.whole_page = s.whole_page && slot == chunk;
    return s;
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
    uint32_t lane;  // lane in the tile
    uint32_t i;     // chunks into the column
};

constexpr WalkPos walk_start(uint32_t first) { return {first, 0, 0}; }

// The WalkPos of seqno `skip` in the walk of [first, first + count). first and count are whole walk pages.
// Full tiles (every lane holds xfer chunks) in closed form; the clipped tail by a loop. skip == count gives a
// position past the end.
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
// Relay (unicast)
////////////////////////////////////////////////////////////////

// Unicast relays by reading back, from this device's own output, what upstream delivered there, into CB
// slots. A NoC read needs its source and destination congruent mod the source memory's read alignment
// (`out_read_alignment`: hal DRAM alignment, 64 B Blackhole / 32 B Wormhole, or hal L1 alignment, 16 B). The
// CB base is DRAM-aligned (program.cpp allocates CBs at the DRAM alignment) and every slot sits at a multiple
// of the slot size, so the rule is about the output offset of a chunk versus its slot offset:
//
//   out_cpp == 1 -- every chunk is its own output page, which starts aligned. With slot == out_aligned_page
//                   (the padded slot: relay_needs_padded_slot) every slot is aligned too, and the pad bytes
//                   ride into the output page's own pad. Legal whenever the chunk fits the page.
//   out_cpp  > 1 -- (Concat, or k > 1) chunk j sits at j * chunk inside the page, which for Concat is *shared with
//                   other devices*, so the slot cannot be widened (a whole-page relay would clobber a neighbour's
//                   chunks): the CB stays at chunk stride, which needs the page exactly tiled by chunks (`packed`)
//                   and every chunk offset aligned (chunk % out_read_alignment == 0).
//
// Routing evaluates it on the k = 1 walk page (GatherGeometry::relay_ok); the unicast factory re-checks it on the
// chunk it actually walks (WalkGeometry::relay_ok).
constexpr bool relay_read_ok(uint32_t chunk, uint32_t out_cpp, uint32_t out_aligned_page, uint32_t out_read_alignment) {
    if (chunk == 0 || out_read_alignment == 0) {
        return false;
    }
    if (out_cpp == 1) {
        return out_aligned_page >= chunk;
    }
    return out_aligned_page == out_cpp * chunk && chunk % out_read_alignment == 0;
}

// The padded slot the relay needs (and the only case it changes anything): one chunk per output page, and
// that page is padded past the chunk.
constexpr bool relay_needs_padded_slot(uint32_t chunk, uint32_t out_cpp, uint32_t out_aligned_page) {
    return out_cpp == 1 && out_aligned_page > chunk;
}

////////////////////////////////////////////////////////////////
// Unicast schedule
////////////////////////////////////////////////////////////////

// Stripes a direction sends from device `dev`: ring -> N / 2; line forward -> dev + 1, backward -> N - dev; 0 at a
// line's dead end (a sink direction).
constexpr uint32_t relay_iters(uint32_t num_devices, bool is_ring, uint32_t dev, bool is_forward) {
    if (is_ring) {
        return num_devices / 2;
    }
    if (is_forward) {
        return dev + 1 < num_devices ? dev + 1 : 0u;
    }
    return dev > 0 ? num_devices - dev : 0u;
}

// Whether any unicast device reads its output back at all: the most stripes one direction sends is N / 2 on a
// ring and N - 1 on a line, and only stripes after the first are relayed. With no relay (2 devices; a 3-device
// ring) the relay rule above does not apply.
constexpr bool unicast_relays(uint32_t num_devices, bool is_ring) {
    return (is_ring ? num_devices / 2 : (num_devices > 0 ? num_devices - 1 : 0)) > 1;
}

// One (device, direction, slice)'s part of the store-and-forward schedule, at seqno level. The kernels run it as:
//   iteration i sends stripe (dev + i * stripe_step) % N, seqnos [skip, skip + take) of the slice's walk, where
//   (skip, take) = (final_skip, final_take) on the last iteration and (0, slice_count) otherwise; iteration 0 also
//   writes the same seqnos into our own output iff do_local_write. The reader of iteration i >= 1 waits for
//   data_valid >= (i - 1) * slice_count + skip + (chunks read so far), and finishes with a wait for total_chunks.
// The host gtest test_all_gather_schedule.cpp simulates every device and direction from this function: every output
// chunk written exactly once, no relay read of an unwritten chunk, no deadlock, total_chunks == chunks delivered.
struct UnicastSchedule {
    uint32_t num_iters = 0;       // stripes this direction sends; 0 = sink direction
    uint32_t stripe_step = 0;     // forward N - 1 (i.e. -1), backward +1
    bool do_local_write = false;  // iteration 0 also writes its seqnos into our own output
    uint32_t num_granular = 0;    // leading iterations the downstream relays (data_valid at granularity)
    uint32_t final_skip = 0;      // last iteration: seqnos [final_skip, final_skip + final_take)
    uint32_t final_take = 0;
    uint32_t total_chunks = 0;  // chunks upstream delivers into our output: the reader's completion wait
};

constexpr UnicastSchedule unicast_schedule(
    uint32_t num_devices, bool is_ring, uint32_t dev, bool is_forward, uint32_t slice_count) {
    UnicastSchedule s{};
    // Even ring: the antipode device receives the antipode stripe as halves, one from each direction. The split is
    // by seqno, not chunk id, so the positions data_valid counts still line up downstream.
    const bool ring_even_split = is_ring && (num_devices % 2 == 0);
    const uint32_t fwd_iters = relay_iters(num_devices, is_ring, dev, true);
    s.num_iters = relay_iters(num_devices, is_ring, dev, is_forward);
    s.stripe_step = is_forward ? num_devices - 1 : 1u;
    // Our own stripe goes into our output once: forward writes it unless forward is a sink, then backward does.
    // On a 2-device ring the one iteration is also the split one, so each direction sends only its half and both
    // must write it locally.
    s.do_local_write = is_forward ? fwd_iters > 0 : (fwd_iters == 0 || (ring_even_split && s.num_iters == 1));
    if (s.num_iters > 0) {
        // Downstream relays all but its last stripe; a mid-stripe signal to a sink costs ~2% on an 8-device line
        // (measured), and nothing reads it early.
        const uint32_t down_dev = is_ring ? 0u : (is_forward ? dev + 1 : dev - 1);
        const uint32_t down_iters = relay_iters(num_devices, is_ring, down_dev, is_forward);
        s.num_granular = down_iters > 0 ? down_iters - 1 : 0u;
    }
    s.final_take = slice_count;
    if (ring_even_split) {
        const uint32_t half = slice_count / 2;
        s.final_skip = is_forward ? 0u : half;
        s.final_take = is_forward ? half : slice_count - half;
    }
    const uint32_t num_recv = is_ring ? num_devices / 2 : (is_forward ? dev : num_devices - 1 - dev);
    // The even-ring antipode stripe arrives as a half: the upstream runs this same direction, so it delivers our
    // final_take, not a whole stripe.
    s.total_chunks = num_recv * slice_count - (ring_even_split ? slice_count - s.final_take : 0u);
    return s;
}
