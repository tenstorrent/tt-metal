// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>

#include <tt-metalium/math.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::ccl {

////////////////////////////////////////////////////////////////
// Sizing the chunk machinery. Both factories use this; neither one's own tuning lives here.
//
// Glossary (same words as kernels/chunk_walk.hpp and kernels/chunk_packets.hpp):
//   chunk           -- the unit we move. min(input page, output page).
//   chunks_per_page -- how many chunks share one page. One number per side, and one of them is 1.
//   bank_step       -- page-id step to the next chunk sitting next to this one in memory.
//                      Interleaved: the bank count. 1 means chunks are already neighbours.
//   run             -- chunks next to each other in memory. One NOC command, one packet segment.
//   payload_chunks  -- chunks the fabric payload could hold.
//   packet_chunks   -- chunks a packet really holds: min(payload_chunks, 4 runs).
//   entry           -- one CB page, in chunks. Always a whole number of packets.
//   stripe          -- the op's boundary: our chunks per output row, which is where a run stops.
//
// The one number the op hands over is `stripe`. Everything below follows from it.
////////////////////////////////////////////////////////////////

// --- chunk ---
// The kernel always reads whole *aligned* input pages into L1 (required by the input's NoC read
// alignment, DRAM or L1) but writes at output *content* (unaligned) granularity, which is why the
// chunk is the smaller of the two and why only one side ever holds several chunks per page.
struct ChunkSizes {
    uint32_t chunk_size;           // the unit we move
    uint32_t in_chunks_per_page;   // chunks per input page. > 1 when the input page is the bigger one
    uint32_t out_chunks_per_page;  // chunks per output page. > 1 when the output page is the bigger one
};

inline ChunkSizes chunk_sizes(const Tensor& input_tensor, const Tensor& output_tensor) {
    const uint32_t input_page_size = input_tensor.buffer()->aligned_page_size();
    const uint32_t input_content = input_tensor.buffer()->page_size();
    const uint32_t output_content = output_tensor.buffer()->page_size();

    // The input page bigger: the chunk is an output page, and an input page holds several.
    // Otherwise: the chunk is a whole aligned input page, and an output page may hold several.
    const bool input_is_bigger = input_content > output_content;
    ChunkSizes sizes{
        .chunk_size = input_is_bigger ? output_content : input_page_size,
        .in_chunks_per_page = input_is_bigger ? input_content / output_content : 1u,
        .out_chunks_per_page = input_is_bigger ? 1u : output_content / input_content,
    };

    TT_FATAL(
        sizes.out_chunks_per_page == 1 || input_page_size == input_content,
        "all_gather cannot share an output page between chunks of a padded input page: {} B of content in a {} B "
        "page. Use an unpadded input page, or the same buffer type on both sides.",
        input_content,
        input_page_size);
    return sizes;
}

// --- how many chunks a packet holds ---
// A packet carries at most four scatter segments (NOC_SCATTER_WRITE_MAX_CHUNKS in
// tt_metal/fabric/fabric_edm_packet_header.hpp), and a segment is one run. On an interleaved output
// the chunks next to each other in memory are bank_step page ids apart, and a run may not leave our
// stripe, so a run averages stripe / min(stripe, bank_step) chunks. Where four of those come to
// less than the payload, the segment count is what ends the packet and the rest of the payload is
// dead space -- which matters because the CB entry is sized in packets below.
inline uint32_t packet_chunks_of(uint32_t payload_chunks, uint32_t stripe, uint32_t bank_step) {
    const uint32_t run_den = std::min(stripe, std::max(1u, bank_step));
    const uint32_t four_runs = 4 * stripe / run_den;
    return std::clamp(four_runs, 1u, payload_chunks);
}

// --- the fabric payload this shape wants ---
// The op cannot choose this: the payload is a device-level fabric config, fixed before the op runs.
// So this only reports. Three things set it, and only the first two are geometry:
//
//  1. Whole chunks. A fractional tail is payload the packer can never use -- at a 2 KB chunk the
//     15232 B Blackhole maximum carries 7 chunks and wastes 896 B of every slot it fills.
//  2. No more than four runs' worth. Past that the segment count ends the packet before the payload
//     does and the rest of the slot is dead space. This is what made a 10-chunk stripe 37% slower
//     than a 32-chunk one at the maximum payload.
//  3. Around 8 KB. A fabric slot is sized by the payload, so a larger payload means fewer slots in
//     ERISC L1 and a shallower pipeline. Measured on bf16 at 20 MB of output: 8192 beats the 15232
//     maximum by 9-28% at every stripe, and beats 10240 and 12288 by 12-16% at a 32-chunk stripe
//     where all three fill a packet completely -- so this is not a fill effect. Below ~6 KB it turns
//     around again (4352 is far worse than 8192), so the optimum is interior, not "as small as
//     possible".
//
// Rule 3 is measured at a 2 KB chunk only, and the byte scale is the part least likely to travel:
// it comes from slot counts and link latency, neither of which knows about chunks. Rules 1 and 2 are
// arithmetic. 4 * chunk_size is the shape-independent answer -- a run is never shorter than one
// chunk, so four of them always fill a 4-chunk payload whatever the stripe.
inline uint32_t ideal_payload_bytes(uint32_t chunk_size, uint32_t stripe, uint32_t bank_step, uint32_t ceiling) {
    constexpr uint32_t target_bytes = 8 * 1024;
    const uint32_t run_den = std::min(stripe, std::max(1u, bank_step));
    const uint32_t by_segments = std::max(1u, 4 * stripe / run_den);
    const uint32_t by_target = std::max(1u, target_bytes / chunk_size);
    const uint32_t by_hardware = std::max(1u, ceiling / chunk_size);
    return std::min(std::min(by_segments, by_target), by_hardware) * chunk_size;
}

// The hardware ceiling on a fabric payload, which is not the configured payload. Mirrors
// FabricEriscDatamoverBuilder::max_packet_payload_size_bytes_* in
// tt_metal/fabric/erisc_datamover_builder.hpp, which is not reachable from here. 0 = unknown, in
// which case say nothing rather than advise blind.
inline uint32_t fabric_payload_ceiling(tt::ARCH arch) {
    switch (arch) {
        case tt::ARCH::WORMHOLE_B0: return 7616;
        case tt::ARCH::BLACKHOLE: return 15232;
        default: return 0;
    }
}

// Say so when the fabric payload is not what this shape wants. Quotes what a packet actually gets
// out of the configured payload, because that is the number that costs time -- but note the smaller
// payload can win even at full fill, by leaving room for more slots, so the advice is not only about
// the waste.
inline void report_payload(
    uint32_t configured, uint32_t chunk_size, uint32_t stripe, uint32_t bank_step, tt::ARCH arch) {
    const uint32_t ceiling = fabric_payload_ceiling(arch);
    if (ceiling == 0) {
        return;
    }
    const uint32_t ideal = ideal_payload_bytes(chunk_size, stripe, bank_step, ceiling);
    if (ideal == configured) {
        return;
    }
    const uint32_t carried = packet_chunks_of(std::max(1u, configured / chunk_size), stripe, bank_step) * chunk_size;
    log_warning(
        tt::LogOp,
        "all_gather: fabric packet payload is {} B and a packet here fills {} B of it ({}%). This shape wants {} B: a "
        "whole number of {} B chunks, at most four runs\' worth, and small enough to leave the fabric its slot depth. "
        "Set FabricRouterConfig::max_packet_payload_size_bytes.",
        configured,
        carried,
        (100 * carried) / std::max(1u, configured),
        ideal,
        chunk_size);
}

// --- how long a run may be ---
// Capping a run at one packet's worth makes a walk column exactly one packet, worth 4-7% where it
// applies. But it shortens every NOC transfer too, and below roughly 10 KB that costs more than the
// packet it saves: at a 1088 B chunk the same rule asks for 5-9 KB transfers and loses 6-10%.
// Returns 0 for "no cap", which is what the kernels expect.
inline uint32_t run_max_of(uint32_t packets, uint32_t payload_chunks, uint32_t chunk_size) {
    constexpr uint32_t min_transfer_bytes = 10 * 1024;
    const bool worth_capping = packets < payload_chunks && packets * chunk_size >= min_transfer_bytes;
    return worth_capping ? packets : 0u;
}

// --- how big a CB entry is ---
// The writer flushes at every entry boundary, so an entry has to be a whole number of packets: an
// entry that is not cuts a packet every time round, and a cut packet still costs a whole fabric
// slot. A 32 KB entry over a 14336 B packet measured 11-17% slower at long stripes for exactly
// that reason.
//
// Within that, an entry also has to be big enough for the reader/writer handshake to amortise --
// one wait_front/pop_front pair plus a read barrier per entry. At a small fabric packet the
// one-packet entry is only a few KB and the handshake shows: an 8-device line at a 8192 B packet
// was worth 26% once the entry grew. `floor_bytes` is that floor.
//
// `unit` is what a packet actually carries, and `least_units` is the caller's own tuning. The
// result is clamped to the L1 an entry may use.
inline uint32_t entry_chunks_of(
    uint32_t unit,
    uint32_t chunk_size,
    uint32_t least_units,
    uint32_t floor_bytes,
    uint32_t cb_depth,
    uint32_t max_l1_space) {
    const uint32_t unit_bytes = unit * chunk_size;
    uint32_t units = std::max(std::max(1u, least_units), tt::div_up(floor_bytes, unit_bytes));
    const uint32_t units_that_fit = max_l1_space / (cb_depth * unit_bytes);
    if (units_that_fit < units) {
        log_warning(
            tt::LogOp,
            "all_gather CB entry shortened from {} to {} packet(s) by L1 headroom ({} B available); performance may "
            "regress.",
            units,
            std::max(1u, units_that_fit),
            max_l1_space);
        units = std::max(1u, units_that_fit);
    }
    return units * unit;
}

}  // namespace ttnn::operations::ccl
