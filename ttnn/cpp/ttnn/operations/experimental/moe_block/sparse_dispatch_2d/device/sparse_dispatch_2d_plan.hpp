// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host math of the sparse 2D dispatch that is not shared with the other ops: each port's input rings and output
// streams (wiring), and the program geometry (CB sizes, the row group / packing defaults). Device-free.

#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "ttnn/operations/experimental/moe_block/common/mesh_routing.hpp"

namespace ttnn::operations::experimental::moe_block::sparse_dispatch_2d {

// CB indices of the port cores (the manifest is listed first so it sits at the program's CB base: the same address on
// every port core of every chip).
inline constexpr uint32_t CB_ROWS = 0, CB_META = 1, CB_PLAN = 2, CB_STAGE = 3, CB_MANIFEST = 5;
inline constexpr uint32_t META_PAGE = 16;
inline constexpr uint32_t SEM_WORDS = 16;  // per port core: 4 semaphore words, 16 B apart

struct StreamWiring {
    uint32_t on = 0;          // 0 off, 1 on, 2 terminal (END(GMAX) only)
    Dir target = Dir::XPlus;  // the port's direction at the neighbour
    uint32_t ring = 0;        // its ring there
    bool operator==(const StreamWiring&) const = default;
};

// sparse_all_gather_2d._wiring: port (chip, d)'s inputs and output streams. ring_mask bit k: input ring k exists
// (0: rows continuing in d from the chip behind, its port d; 1 / 2: rows turning into d's axis that arrived moving
// + / - along the other axis); bit 3 + k: ring k is a terminal stream. streams[s] (0: continue, 1 / 2: turn + / -).
struct PortWiring {
    uint32_t ring_mask = 0;
    std::array<StreamWiring, 3> streams{};
    bool operator==(const PortWiring&) const = default;
};
PortWiring wiring(const MeshPlan& plan, ChipCoord chip, Dir d);

// GMAX: the last END generation, (C - 1) + (R - 1) - 1.
inline uint32_t gmax(const MeshTopology& topo) { return (topo.C - 1) + (topo.R - 1) - 1; }

// The MoE exchange's row group: 8 rows while 2 x width <= 7168, else 2 (the CBs of
// 2 x group rows stay small for wide rows).
inline uint32_t default_group(uint32_t width) { return 2 * width <= 7168 ? 8 : 2; }
// Rows a fabric packet: 2 when two rows fit the payload.
inline uint32_t default_pack(uint32_t row_bytes, size_t max_payload_bytes) {
    return 2 * static_cast<size_t>(row_bytes) <= max_payload_bytes ? 2 : 1;
}

// CB sizes in bytes on every port core (sparse_all_gather_2d's descriptor).
struct CbSizes {
    uint32_t rows = 0, meta = 0, plan = 0, stage = 0, manifest = 0;
    uint32_t total() const { return rows + meta + plan + stage + manifest; }
    // In the block's arena: each CB at a 64 B aligned offset, the manifest at 0 (the same address on every port core of
    // every chip, which the remote senders write at); arena_offsets() = {manifest, rows, meta, plan, stage}.
    std::array<uint32_t, 5> arena_offsets() const {
        const auto a = [](uint32_t v) { return (v + 63u) / 64u * 64u; };
        const uint32_t o_rows = a(manifest), o_meta = o_rows + a(rows), o_plan = o_meta + a(meta), o_stage = o_plan + a(plan);
        return {0u, o_rows, o_meta, o_plan, o_stage};
    }
    uint32_t arena_bytes() const { return arena_offsets()[4] + (stage + 63u) / 64u * 64u; }
};
CbSizes cb_sizes(
    const MeshTopology& topo,
    uint32_t S,
    uint32_t row_bytes,
    uint32_t plan_bytes,
    uint32_t group,
    uint32_t manifest_batch);

}  // namespace ttnn::operations::experimental::moe_block::sparse_dispatch_2d
