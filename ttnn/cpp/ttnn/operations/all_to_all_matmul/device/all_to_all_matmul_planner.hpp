// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

#include <tt-metalium/vocabulary/core_coord.hpp>
#include <tt-metalium/memory/cross_program/global_semaphore.hpp>
#include <tt-metalium/vocabulary/mesh_coord.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

// The host plan of all_to_all_matmul (the snapshot's all_to_all_matmul_program_descriptor.py: _topology_plan,
// run()'s geometry / send schedule / tail split, _resources). Three memoized layers, each returned by const&:
//   mesh_topology   per (mesh id, mesh shape, cluster_axis, num_links): routes, links, port cores
//   solve_plan      per (topology, PlanInputs): geometry, send lists, slot work of every core of every chip
//   mesh_resources  per mesh_topology entry: global semaphores, grow-only
namespace ttnn::operations::all_to_all_matmul {

// ---- Blocking / transport knobs: the only values the source ever selected ----
constexpr uint32_t kWeightBlockBytes = 256 * 1024;      // one weight block (Kt x block_n tiles) -> block_n
constexpr uint32_t kWeightDepth = 2;                    // weight blocks in cb_weight (capped by blocks per core)
constexpr uint32_t kWeightPrefetchBlocks = 1;           // weight blocks of a slot read before its landing wait
constexpr uint32_t kWeightReadChunkBytes = 16 * 1024;   // weight bytes in flight per compute core
constexpr uint32_t kOutDepth = 2;                       // output row groups in cb_out
constexpr uint32_t kPortBatchBytes = 87 * 1024;         // packets per port batch = this / packet bytes
constexpr uint32_t kPortReadInflightBytes = 40 * 1024;  // port reader window = ceil(this / packet bytes), 1..15
constexpr uint32_t kMaxPortWindow = 15;                 // NoC transaction ids 1..15
constexpr uint32_t kSendRunEntries = 3;                 // grouped send: entries per destination run
constexpr double kSendRunMaxLocalRatio = 6.0;           // grouped send only up to this DRAM / exchanged ratio
constexpr uint32_t kTailSplitMaxHeads = 8;              // interleaved path
constexpr uint32_t kTailSplitMaxHeadsGrouped = 12;      // grouped path
constexpr uint32_t kTailMsplitHeads = 4;                // grouped path only (else 0)
constexpr uint32_t kTailMsplitPieces = 3;
constexpr uint32_t kTailMsplitPrefetch = 2;
constexpr uint32_t kASlotBudgetBytes = 384 * 1024;  // A_RESIDENT predicate
constexpr uint32_t kMaxHeadsPerCore = 8;            // arrival counters per compute core
constexpr uint32_t kMaxPacketHeaders = 12;          // packet header pool per data-movement RISC
// Collapsed knobs (one value in the source): send order "near", fence per destination, pack across entries
// (when packet_tiles <= slot_tiles), landing estimate "share", helper pick "pool", weight pace 0, output write
// chunk 0, BLOCK_HEADS 1.
constexpr bool kFencePerDest = true;
constexpr uint32_t kOutChunkTiles = 0;

// Ring directions of a port.
constexpr uint32_t kFwd = 0;
constexpr uint32_t kBwd = 1;
// Port (d, l) index in ChipRoute::ports / ChipPlan::ports: the source's dict order (l outer, d inner).
inline uint32_t port_index(uint32_t d, uint32_t l) { return (l * 2) + d; }

struct ChipRoute {
    uint32_t p = 0;                                              // position in the group
    std::vector<tt::tt_metal::distributed::MeshCoordinate> grp;  // the group, by position
    tt::tt_metal::distributed::MeshCoordinate next{0, 0};
    tt::tt_metal::distributed::MeshCoordinate prev{0, 0};
    std::vector<int32_t> route;                  // per position t: kFwd / kBwd first-hop port, -1 at t == p
    std::array<std::vector<uint32_t>, 2> links;  // per direction: the first num_links forwarding link indices
    std::vector<tt::tt_metal::CoreCoord> ports;  // logical port core, by port_index(d, l)
};

struct MeshTopology {
    uint32_t G = 0;
    uint32_t num_links = 0;
    std::map<tt::tt_metal::distributed::MeshCoordinate, ChipRoute> chips;
    std::vector<tt::tt_metal::CoreCoord> port_cells;     // every chip's port cores, sorted (x, y), unique
    std::vector<tt::tt_metal::CoreCoord> compute_cells;  // grid minus port_cells, row-major (y outer, x inner)
    tt::tt_metal::CoreRangeSet compute_cores;            // compute_cells as merged rectangles
    uint32_t payload_bytes = 0;  // tt::tt_fabric::get_tt_fabric_max_payload_size_bytes(), read once per mesh
};

// Routes (get_eth_forwarding_direction), links (get_forwarding_link_indices) and port placement: per (d, l), the
// worker with the fewest NoC1 hops to get_forwarding_eth_core(src, peer, link), measured as
// closest.distance_in_noc_hops + get_worker_noc_hop_distance(...), ties by logical (y, x), skipping taken cores;
// on any failure the fallback (2 * num_links centred cores of logical row 0). Throws RuntimeError
// (TT_THROW) when a route leaves through neither ring neighbour.
const MeshTopology& mesh_topology(
    tt::tt_metal::distributed::MeshDevice* mesh_device, uint32_t cluster_axis, uint32_t num_links);

// Everything solve_plan reads besides the topology.
struct PlanInputs {
    uint32_t B = 0, Hl = 0, S = 0, K = 0, H = 0, N = 0;
    uint32_t in_tile_bytes = 0;  // input page size (tile of the input's dtype)
    uint32_t w_tile_bytes = 0;   // weight page size
    uint32_t payload_bytes = 0;  // tt::tt_fabric::get_tt_fabric_max_payload_size_bytes()
    bool output_in_l1 = false;
    bool fp32_dest_acc_en = true;
    auto operator<=>(const PlanInputs&) const = default;
};

struct SlotWork {
    uint32_t h = 0;
    bool wait = false;  // remote head or helper slot
    uint32_t pre = 1;   // weight blocks before the landing wait
    uint32_t nb0 = 0, nb1 = 0, m0 = 0, m1 = 0;
    std::optional<tt::tt_metal::CoreCoord> helper;  // tail N-split: the helper core (logical) this slot forwards to
    uint32_t helper_slot = 0;                       // the helper's slot index
    uint32_t pace = 0;
};

struct CoreWork {
    tt::tt_metal::CoreCoord core;  // logical
    std::vector<SlotWork> slots;   // empty for an idle active core
};

struct PortDest {
    tt::tt_metal::distributed::MeshCoordinate coord{0, 0};
    std::optional<uint32_t> ready_index;  // ready counter of this port for that chip (none: no data to it)
};
struct PortReady {
    uint32_t dest_index = 0;  // into PortWork::dests
    tt::tt_metal::distributed::MeshCoordinate target_chip{0, 0};
    tt::tt_metal::CoreCoord target_port_core;  // logical, on target_chip
    uint32_t ready_index = 0;
};
struct PortSub {
    uint32_t dest_index = 0;
    uint32_t tiles = 0;
    tt::tt_metal::CoreCoord core;  // logical destination compute core
    uint32_t slot = 0;             // its slot index (landing offset slot * slot_bytes, arrival counter slot)
};
struct PortWork {
    tt::tt_metal::CoreCoord core;  // logical
    tt::tt_metal::distributed::MeshCoordinate peer{0, 0};
    uint32_t link = 0;
    std::vector<uint32_t> entries;  // j | t << port_reader_rt::ENTRY_DEST_SHIFT
    std::vector<PortDest> dests;
    std::vector<PortReady> readies;
    std::vector<PortSub> subs;
    bool has_conn = false;
    uint32_t hdrs_per_dest = 1;
};

struct ChipPlan {
    uint32_t q = 0;
    std::vector<CoreWork> cores;  // every active cell, in active_cells order
    std::vector<PortWork> ports;  // by port_index(d, l)
    tt::tt_metal::CoreRangeSet port_cores;
};

struct Plan {
    // geometry
    uint32_t Kt = 0, Nt = 0, Mq = 0, Mq_t = 0, St = 0;
    bool retile = false;
    uint32_t stage_tiles = 0, block_m = 0, slot_tiles = 0, slot_bytes = 0;
    uint32_t block_n = 0, num_n_blocks = 0, out_subblock_h = 1, out_subblock_w = 1;
    // transport
    uint32_t packet_tiles = 0, packet_bytes = 0, batch_chunks = 0, port_window = 0;
    bool grouped = false, pack = false, tail_split = false;
    // compute cores
    uint32_t weight_depth = 0, weight_chunk_tiles = 0, w_prefetch = 0;
    uint32_t slots_per_core_max = 0, cap_tiles = 0, n_ready = 1;
    std::vector<tt::tt_metal::CoreCoord> active_cells;
    tt::tt_metal::CoreRangeSet active_cores;
    tt::tt_metal::CoreRangeSet all_port_cores;
    std::map<tt::tt_metal::distributed::MeshCoordinate, ChipPlan> chips;
};

// Memoized on (&topology, inputs). TT_FATALs mirror the source's RuntimeErrors (R2 weight block, too many head
// slots, A_RESIDENT, header pool) and asserts.
const Plan& solve_plan(const MeshTopology& topology, const PlanInputs& inputs);

// The landing scratch a call uses: HEIGHT_SHARDED L1 over topology.compute_cores, ROW_MAJOR, shard
// (cap_tiles * TILE_HEIGHT, TILE_WIDTH), shape [n_compute * cap_tiles * TILE_HEIGHT, TILE_WIDTH], `dtype`, TILE.
tt::tt_metal::TensorSpec scratch_spec(const MeshTopology& topology, const Plan& plan, tt::tt_metal::DataType dtype);

// Per-mesh global semaphores (zero-initialised): arrival counters on the compute cores, one per head slot
// (>= slots_per_core_max), and ready counters on the port cores (>= n_ready). Created on the mesh's first need,
// before the call allocates any L1 tensor; synchronizes the mesh when it creates any. Grow-only, kept in a
// ttsl::Indestructible map.
struct MeshResources {
    std::vector<tt::tt_metal::GlobalSemaphore> arrival;
    std::vector<tt::tt_metal::GlobalSemaphore> ready;
    std::vector<uint32_t> arrival_addrs;
    std::vector<uint32_t> ready_addrs;
};
const MeshResources& mesh_resources(
    tt::tt_metal::distributed::MeshDevice* mesh_device, const MeshTopology& topology, const Plan& plan);

}  // namespace ttnn::operations::all_to_all_matmul
