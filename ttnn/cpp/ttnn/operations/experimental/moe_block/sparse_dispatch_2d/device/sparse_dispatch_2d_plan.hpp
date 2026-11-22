// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host math of the sparse 2D dispatch that is not shared with the other ops: each port's input rings and output
// streams (wiring), and the program geometry (CB sizes, the row group / packing defaults). Device-free.

#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/operations/experimental/moe_block/common/mesh_routing.hpp"
#include "ttnn/operations/experimental/moe_block/common/port_placement.hpp"
#include "ttnn/operations/experimental/moe_block/sparse_dispatch_2d/device/kernels/sparse_dispatch_2d_layout.hpp"

namespace ttnn::operations::experimental::moe_block::sparse_dispatch_2d {

// CB indices of the port cores (the manifest is listed first so it sits at the program's CB base: the same address on
// every port core of every chip).
inline constexpr uint32_t CB_ROWS = 0, CB_META = 1, CB_PLAN = 2, CB_STAGE = 3, CB_MANIFEST = 5;
// The port kernels' per-core values (common/args_table.hpp): a CB of ARGS_BYTES on every port core, the reader's block
// at ARGS_READER_OFF and the sender's at ARGS_SENDER_OFF (each at most ARGS_KERNEL_BYTES: the args, then the fixups).
inline constexpr uint32_t CB_ARGS = 6, ARGS_KERNEL_BYTES = 128, ARGS_READER_OFF = 0, ARGS_SENDER_OFF = 128;
inline constexpr uint32_t ARGS_BYTES = 256;
inline constexpr uint32_t META_PAGE = sparse_dispatch_2d_layout::kEntryBytes;
inline constexpr uint32_t SEM_WORDS = 16;  // per port core: 4 semaphore words, 16 B apart

struct StreamWiring {
    uint32_t on = 0;          // 0 off, 1 on, 2 terminal (END(GMAX) only)
    Dir target = Dir::XPlus;  // the port's direction at the neighbour
    uint32_t ring = 0;        // its ring there
};

// Port (chip, d)'s inputs and output streams. ring_mask bit k: input ring k exists
// (0: rows continuing in d from the chip behind, its port d; 1 / 2: rows turning into d's axis that arrived moving
// + / - along the other axis); bit 3 + k: ring k is a terminal stream. streams[s] (0: continue, 1 / 2: turn + / -).
struct PortWiring {
    uint32_t ring_mask = 0;
    std::array<StreamWiring, 3> streams{};
};
PortWiring wiring(const MeshPlan& plan, ChipCoord chip, Dir d);

// One single-core range per distinct port cell of every chip, in port-map order. The port kernels, CBs and semaphore
// words sit on the union: a cell is one address on every chip, so a port addresses its peer's.
std::vector<tt::tt_metal::CoreRange> port_cell_ranges(const PortCores& cores);

// GMAX: the last END generation, (C - 1) + (R - 1) - 1.
inline uint32_t gmax(const MeshTopology& topo) { return (topo.C - 1) + (topo.R - 1) - 1; }

// The row groups the dispatch supports (rows a reader -> sender hand-off; its port cores hold 2 x group rows). In the
// MoE block the block's L1 plan picks one (BlockL1Plan.dispatch_group: the largest that leaves the block's plan as
// the smallest does); on its own the dispatch takes the largest.
// Powers of two, 2 .. 8: with pack 2 (default_pack) a group of at least 2 lets the sender pair adjacent rows into one
// fabric packet (a group of 1 works but sends every row alone). A larger group keeps the reader further ahead of the
// sender, at 2 x group rows of CB_ROWS an L1 cost per port core.
// TODO: the cap of 8 was chosen empirically (no model behind it); 16 was not measured.
inline constexpr std::array<uint32_t, 3> kGroups = {2, 4, 8};
// The entries the sender stages for a neighbour port before one fused write + counter increment (a fabric packet of
// kManifestBatch x 16 B). The fused packet is slow, so the entries are batched and it is sent once a batch (or when the
// sender is about to wait). CB_STAGE holds 3 streams x 2 halves x kManifestBatch entries. TODO: 16 was chosen
// empirically (it amortises the packet without a large stage CB); no derived bound.
inline constexpr uint32_t kManifestBatch = 16;
// Rows a fabric packet: 2 when two rows fit the payload.
inline uint32_t default_pack(uint32_t row_bytes, size_t max_payload_bytes) {
    return 2 * static_cast<size_t>(row_bytes) <= max_payload_bytes ? 2 : 1;
}

// CB sizes in bytes on every port core.
struct CbSizes {
    uint32_t rows = 0, meta = 0, plan = 0, stage = 0, manifest = 0, args = ARGS_BYTES;
    // In the block's arena: each CB at a 64 B aligned offset, the manifest at 0 (the same address on every port core of
    // every chip, which the remote senders write at); arena_offsets() = {manifest, rows, meta, plan, stage, args}.
    std::array<uint32_t, 6> arena_offsets() const {
        const auto a = [](uint32_t v) { return (v + 63u) / 64u * 64u; };
        const uint32_t o_rows = a(manifest), o_meta = o_rows + a(rows), o_plan = o_meta + a(meta),
                       o_stage = o_plan + a(plan), o_args = o_stage + a(stage);
        return {0u, o_rows, o_meta, o_plan, o_stage, o_args};
    }
    uint32_t arena_bytes() const { return arena_offsets()[5] + (args + 63u) / 64u * 64u; }
};
CbSizes cb_sizes(
    const MeshTopology& topo,
    uint32_t S,
    uint32_t row_bytes,
    uint32_t plan_bytes,
    uint32_t group,
    uint32_t manifest_batch);

}  // namespace ttnn::operations::experimental::moe_block::sparse_dispatch_2d
