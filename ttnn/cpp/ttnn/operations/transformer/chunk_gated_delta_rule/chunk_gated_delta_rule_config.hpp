// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Program configs for chunk_gated_delta_rule: selects the device implementation and how it is laid
// out on the chip. The alternative you pass selects the program (mono / phased / fused);
// A std::nullopt config lets the op choose (between phased and fused) based on a cost model.
//

#pragma once

#include <cstdint>
#include <optional>
#include <variant>

namespace ttnn::transformer {

// The original single-kernel op: one core per head runs prep and scan for every chunk. The slowest
// path, kept as the benchmark/debug reference. Does not accept flat (token-major) q/k/v.
struct ChunkGdnMonoProgramConfig {};

// Prep fanned over the grid -> seven fp32 intermediates in DRAM -> scan: two programs. The bit-exact
// reference the fused path is gated against, and the fallback where no fused geometry fits the grid.
struct ChunkGdnPhasedProgramConfig {
    // Scan: one sender core per head multicasts the six shared inputs to that head's V-block siblings
    // instead of every sibling re-reading the same DRAM pages. Off = every core reads for itself (A/B).
    bool use_mcast = true;
    // Scan: one core per head (NV=1, the full V) instead of the V-block split. Measurement only — it
    // is the NV=1 upper bound of the per-step chain time.
    bool scan_serial = false;
    // Prep: BH cores (one per head) instead of the whole grid. Measurement only.
    bool prep_serial = false;
};

// One program: per head, NP producer cores run prep and NoC-write the seven intermediates into NV
// receiver cores' CBs; nothing goes through DRAM. The geometry fields default to the calibrated cost
// model's pick for (grid, BH, NC, Vt); pinning one of num_producers / num_receivers makes the model
// fill the other so the pair still fits the grid.
struct ChunkGdnFusedProgramConfig {
    std::optional<uint32_t>
        num_producers;  // NP per head (clamped to the chunk count); the pool size P with producer_pool
    std::optional<uint32_t> num_receivers;  // NV per head; must divide Vt = V / 32
    // Core map. true = row-local: one head per row with its producers east of its receivers, so no
    // two heads share a NoC link; false = row-major 1xNV receiver rectangles with the producers on the
    // remaining cores. nullopt = row-local whenever the geometry has such a layout. Ignored with producer_pool.
    std::optional<bool> row_local;
    // Hand-off ring slots per CB, 1..8: how many chunks a producer may run ahead. nullopt = the model's pick.
    std::optional<uint32_t> handoff_depth;
    bool unicast = true;  // per-receiver unicast writes; false = the linked multicast chain
    bool posted = false;  // posted unicast data writes, VALID ordered by in-order delivery; needs unicast
    // One producer pool for every head instead of NP producers per head: the receivers and BH*NPH HOME
    // producers take the row-local layout of the largest NPH the pool size allows, and every other core of
    // the pool is an EXTRA producer serving all heads. num_producers is then the pool size P (default: every
    // core the receivers leave free, at most BH*NC).
    bool producer_pool = false;
    // Share of every head's chunks the extras take, in [0, 1]; nullopt = NX / P, the balanced load.
    std::optional<float> pool_extra_share;
    // With producer_pool: the producers claim their chunks at run time from per-head counters as they become free
    // (a home producer from its own head, the extras rotating over the heads), instead of the static item lists; the
    // receivers credit whichever producer registered as a chunk's owner. Same outputs, bit for bit; only the timing
    // changes.
    bool dynamic_handoff = false;
};

using ChunkGdnProgramConfig =
    std::variant<ChunkGdnMonoProgramConfig, ChunkGdnPhasedProgramConfig, ChunkGdnFusedProgramConfig>;

enum class ChunkGdnWyInverse : uint32_t {
    // The forward-substitution solve wherever it is supported (Blackhole, chunk_size == 32), Horner everywhere else.
    AUTO = 0,
    // invert_block: quadrant split, two 15-term Horner inverses and an exact off-diagonal on the matrix
    // engine (~60 LLK calls per chunk). Every architecture and chunk size; the reference the solve is
    // validated against.
    HORNER = 1,
    // One SFPU forward-substitution solve reading the pre-negated factor as fp32 in place. Blackhole-only,
    // chunk_size == 32, fused and phased paths (the mono program is Horner-only).
    FORWARD_SUBSTITUTION = 2,
};

}  // namespace ttnn::transformer
