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
    std::optional<uint32_t> num_producers;  // NP per head (clamped to the chunk count)
    std::optional<uint32_t> num_receivers;  // NV per head; must divide Vt = V / 32
    // Core map. true = row-local: one head per row with its producers east of its receivers, so no
    // two heads share a NoC link; false = row-major 1xNV receiver rectangles with the producers on the
    // remaining cores. nullopt = row-local whenever the geometry has such a layout.
    std::optional<bool> row_local;
    uint32_t handoff_depth = 2;  // hand-off ring slots per CB, 1..8: how many chunks a producer may run ahead
    bool unicast = true;         // per-receiver unicast writes; false = the linked multicast chain
    bool posted = false;         // posted unicast data writes, VALID ordered by in-order delivery; needs unicast
};

using ChunkGdnProgramConfig =
    std::variant<ChunkGdnMonoProgramConfig, ChunkGdnPhasedProgramConfig, ChunkGdnFusedProgramConfig>;

}  // namespace ttnn::transformer
