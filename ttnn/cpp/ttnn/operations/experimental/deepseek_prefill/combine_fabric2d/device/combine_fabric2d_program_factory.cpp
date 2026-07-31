// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "combine_fabric2d_program_factory.hpp"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
// pipeline_get_forwarding_direction: which way the fabric routes src -> dst. The assignment of movements
// to producers must agree with it, because a producer can only inject on its own cable.
#include <tt-metalium/experimental/fabric/pipeline_builder.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
// create_device_tensor: allocates device memory from a TensorSpec with no host data, which is what the
// op-internal forwarding buffer wants (never initialised, never read back).
#include "ttnn/tensor/tensor_ops.hpp"
#include <tt_stl/assert.hpp>
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

namespace {

// ---------------------------------------------------------------------------------------------
// L1 layout — identical on every tensix core of every device in the mesh.
//
// Uniformity is load-bearing: a producer on chip A addresses the drain sink of the peer worker on chip B
// without knowing anything about B's harvesting or eth positions. All offsets are relative to the L1
// unreserved base (uniform for a given arch/config).
// ---------------------------------------------------------------------------------------------
constexpr uint32_t PKT_HDR_DRAIN_OFF = 0x0000;  // scratch header for the drain's filler packets
// Target of the drain's value-0 atomic increments. A remote atomic inc needs a real L1 address on the
// far chip; nothing ever reads this word, which is exactly the point.
constexpr uint32_t DRAIN_SINK_OFF = 0x0400;
// 1 kB of per-worker telemetry, in the gap before the source ring, so it costs the payload nothing.
// Sized for expansion; the current record is 22 words.
constexpr uint32_t TELEMETRY_OFF = 0x0800;
constexpr uint32_t TELEMETRY_SIZE = 0x0400;
constexpr uint32_t PROD_BUF_OFF = 0x1000;  // the reader -> producer token ring
constexpr uint32_t L1_SLACK = 0x1000;      // keep clear of whatever sits at the very top of L1
// Every ring slot carries the token plus a fixed metadata tail the reader fills in for the producer.
// 64 rather than the 32 currently used keeps the slot stride DRAM-aligned (14336 + 64 = 64 * 225), which
// is what lets phase 9 point a fabric write straight at a (token_size + 64)-byte forwarding-buffer page.
// Tail layout, all uint64_t so the producer can consume it without sub-word loads:
//   [ 0.. 7] final destination DRAM address on the FINAL destination chip
//   [ 8..15] final destination chip id
//   [16..23] command word: 1 = write to the address at [24..31], 2 = forward (phase 9)
//   [24..31] the address this hop writes to
//   [32..63] reserved
constexpr uint32_t CMBF2D_SLOT_TAIL_BYTES = 64;
// Forwarded tokens between semaphore bumps to the downstream reader. A bump always follows a sentinel
// regardless, so this only sets how finely the downstream reader can pipeline within a chunk. Hardcoded
// for P9.2 correctness; P9.3 promotes it to an op parameter and sweeps it.
constexpr uint32_t CMBF2D_FWD_BUMP_EVERY = 8;
static_assert(PKT_HDR_DRAIN_OFF < DRAIN_SINK_OFF, "drain sink overlaps the drain packet header");
static_assert(DRAIN_SINK_OFF < TELEMETRY_OFF, "telemetry region overlaps the drain sink");
static_assert(TELEMETRY_OFF + TELEMETRY_SIZE <= PROD_BUF_OFF, "telemetry region overlaps the token ring");

// ---------------------------------------------------------------------------------------------
// Forwarding buffer (phase 9): DRAM staging for tokens passing THROUGH this chip.
//
// Phase 9 stops the fabric forwarding multi-hop packets and does it in the op instead. A producer's
// packets then only ever travel one hop, so anything bound further lands in the NEXT chip's forwarding
// buffer and is re-sent from there.
//
// Geometry. The buffer is quartered by (routing plane, send direction) — the pair that uniquely identifies
// the upstream producer from the downstream chip's point of view — and each quarter holds
// `fwd_chunks_per_quarter` chunks of `fwd_pages_per_chunk` pages. The last page a chunk actually uses is a
// sentinel marking its end, so a chunk need not be filled to capacity.
//
// Chunk count per quarter: a chunk arriving at C in one direction is one (source, destination) pair whose
// path passes through C and continues. Summing over upstream distance k >= 1 of the movements from that
// source whose distance exceeds k gives sum_{k=1..H-1} (H-k) = H(H-1)/2, where H = extent/2. That is 6 for
// an 8-ring, matching the plan. It is the worst case of the two directions: the direction that does NOT
// carry the diametrically-opposite chip needs only (H-1)(H-2)/2, but both quarters are sized alike.
// ---------------------------------------------------------------------------------------------
uint32_t fwd_chunks_per_quarter(uint32_t extent) {
    const uint32_t half = extent / 2;
    return half * (half - 1) / 2;
}

// Quarters = (routing plane, direction) pairs = num_links * 2.
uint32_t fwd_total_chunks(uint32_t extent, uint32_t num_links) {
    return fwd_chunks_per_quarter(extent) * num_links * 2;
}

// Pages a chunk can hold. Phase 10 makes chunk lengths DATA-DEPENDENT: a chunk is one producer's share of
// one (origin chip -> this chip) run merged over the experts it serves, and how many tokens that is depends
// on where the router happened to send things. Only the expectation is known here:
//
//     avg tokens per chunk = seq_len_per_chip * num_experts_per_tok
//                            / (num_dispatch_groups * dispatch_group_size * num_links)
//
// (each token picks num_experts_per_tok experts; each expert lives in one dispatch group and on one chip of
// it, so a (origin, destination) pair carries 1/(NDG * dgs) of the traffic, split again across the planes).
// For the 8x4 prefill shape that is 640*2/(4*8*2) = 20 tokens, and it scales linearly with seq.
//
// We provision 4x the mean plus the sentinel. That is a HEURISTIC, good for the roughly-balanced traffic
// random routing produces: the measured spread of tokens per (origin, destination) pair is max/mean 1.43 at
// seq 640 and tighter as seq grows, and the binomial tail concentrates further. It is NOT a bound. Sizing
// the buffer properly — and handling the case where a chunk does not fit — is a later stage; until then an
// overflow would silently scribble on the next chunk, which is why the reader carries a development-only
// overflow counter.
uint32_t fwd_pages_per_chunk(
    uint32_t seq_len_per_chip,
    uint32_t num_experts_per_tok,
    uint32_t num_dispatch_groups,
    uint32_t dispatch_group_size,
    uint32_t num_links) {
    const uint32_t denom = num_dispatch_groups * dispatch_group_size * num_links;
    const uint32_t mean = (seq_len_per_chip * num_experts_per_tok + denom - 1) / denom;
    return 4 * mean + 1;
}

// Telemetry record word layout, shared with the producer kernel. Kept explicit (rather than a struct)
// because the kernel writes it by index and the host reads it by index.
enum TelemetryWord : uint32_t {
    TELEM_MAGIC = 0,  // written LAST by the kernel; zeroed at kernel entry
    TELEM_TOKENS_SENT = 1,
    TELEM_TOKEN_SIZE = 2,
    TELEM_NUM_L1_SLOTS = 3,
    TELEM_T_START_LO = 4,  // written by the READER at its first DRAM read: the effective-BW window opens
    TELEM_T_START_HI = 5,
    TELEM_T_FIRST_SEND_LO = 6,
    TELEM_T_FIRST_SEND_HI = 7,
    TELEM_T_LAST_SEND_LO = 8,
    TELEM_T_LAST_SEND_HI = 9,
    TELEM_T_DRAINED_LO = 10,  // when the EDM drain proved every packet reached the far chip
    TELEM_T_DRAINED_HI = 11,
    TELEM_EDM_SLOTS = 12,      // EDM sender-channel buffer slots (the send pipeline's depth)
    TELEM_DRAIN_PACKETS = 13,  // header-only fillers the drain pushed (= edm_slots - 1)
    TELEM_OUT_BASE_PAGE = 14,  // first page written in the PEER chip's output region
    TELEM_BATCH = 15,          // ring slots claimed/released per trip
    // Stall attribution: where the producer's cycles actually go. Three disjoint buckets on the same wall
    // clock as the timestamps above, so they are directly comparable to the send window.
    TELEM_WAIT_SLOT_CY_LO = 16,  // blocked in wait_for_empty_write_slot => the eth side is the limiter
    TELEM_WAIT_SLOT_CY_HI = 17,
    TELEM_ISSUE_CY_LO = 18,  // issuing the payload: header stamp + 2 NoC writes
    TELEM_ISSUE_CY_HI = 19,
    TELEM_RING_WAIT_CY_LO = 20,  // blocked on the reader => the DRAM read side is the limiter
    TELEM_RING_WAIT_CY_HI = 21,
    // Whole-kernel span: producer entry (before the fabric connection open) to exit (after teardown).
    // The send loop alone is t_last_send - t_first_send; this is the number total-time work reduces.
    TELEM_T_KERNEL_START_LO = 22,
    TELEM_T_KERNEL_START_HI = 23,
    TELEM_T_KERNEL_END_LO = 24,
    TELEM_T_KERNEL_END_HI = 25,
    TELEM_NUM_WORDS = 26,
};
// Bumped whenever the record layout changes, so a stale record from an older kernel reads as invalid
// instead of being misparsed.
constexpr uint32_t TELEMETRY_MAGIC = 0xCF2D0006u;

// ---------------------------------------------------------------------------------------------
// Physical-column geometry
//
// A worker and the eth core it drives must share a PHYSICAL (noc0) x. That is the only coordinate
// space in which cores of different types are comparable: logical coords are per-core-type dense
// indices, and virtual/translated coords put eth and tensix in disjoint ranges (on Blackhole an eth
// core's translated coord is literally (20 + eth_channel, 25)). Match on physical x; convert back to
// logical only for kernel placement and to virtual only for NoC addressing.
// ---------------------------------------------------------------------------------------------
struct DeviceGeometry {
    // physical worker x -> (physical worker y -> logical worker core), restricted to cores this op may
    // program (inside compute_with_storage_grid_size — the dispatch column is NOT in here).
    std::map<uint32_t, std::map<uint32_t, CoreCoord>> columns;
};

DeviceGeometry build_device_geometry(tt::tt_metal::IDevice* dev, const CoreCoord& compute_grid) {
    DeviceGeometry geom;
    for (uint32_t ly = 0; ly < compute_grid.y; ly++) {
        for (uint32_t lx = 0; lx < compute_grid.x; lx++) {
            const CoreCoord logical{lx, ly};
            const auto phys = tt::tt_metal::experimental::Device::get_physical_core_from_logical_core(
                dev, logical, tt::CoreType::WORKER);
            geom.columns[static_cast<uint32_t>(phys.x)][static_cast<uint32_t>(phys.y)] = logical;
        }
    }
    return geom;
}

// The worker in `phys_x` adjacent to the eth row (smallest physical y — eth sits at the low-y edge of
// the grid). nullopt if that column has no programmable worker at all: it may be harvested (differs
// per chip within one mesh) or host the dispatch cores.
std::optional<CoreCoord> adjacent_worker_in_column(const DeviceGeometry& geom, uint32_t phys_x) {
    auto it = geom.columns.find(phys_x);
    if (it == geom.columns.end() || it->second.empty()) {
        return std::nullopt;
    }
    return it->second.begin()->second;
}

struct WorkerPlacement {
    CoreCoord eth_logical;    // the fabric router this worker drives
    uint32_t eth_phys_x = 0;  // its physical column
    uint32_t link_idx = 0;    // link index (routing plane) to open the connection on
    tt::tt_fabric::FabricNodeId peer_node{tt::tt_fabric::MeshId{0}, 0};  // chip across the cable
    CoreCoord worker_logical;                                            // where the producer goes
    CoreCoord worker_physical;  // noc0 coords; x == eth_phys_x unless relocated
    CoreCoord worker_virtual;   // what a remote producer must address
    bool in_eth_column = true;  // false => relocated, no longer adjacent to its router
};

// Every worker this device will host, keyed by the eth core it serves.
struct DevicePlacement {
    std::map<CoreCoord, WorkerPlacement> by_eth_logical;
};

// Decide where this device puts a worker for each of its fabric eth cores.
//
// Depends ONLY on this device's own eth set and its own harvesting, so the answer is stable no matter
// who asks — which is what lets a neighbor rely on it. Deliberately decides for the WHOLE device at
// once: a producer's args need the peer worker's coords on the neighbor chip, and those are NOT simply
// (neighbor_eth.x, y_min) because the neighbor's eth column may itself be tensix-unfriendly.
DevicePlacement decide_placement(
    ttnn::MeshDevice* mesh,
    const ttnn::MeshCoordinate& coord,
    uint32_t axis,
    uint32_t num_links,
    const CoreCoord& compute_grid) {
    auto* dev = mesh->get_device(coord);
    const auto self_node = mesh->get_fabric_node_id(coord);
    const auto geom = build_device_geometry(dev, compute_grid);
    const auto mesh_shape = mesh->shape();

    // This device's fabric eth cores: num_links toward the forward axis neighbor and num_links toward
    // the backward one. Every one is full duplex, so every one gets a producer.
    struct EthEntry {
        CoreCoord eth_logical;
        uint32_t eth_phys_x;
        uint32_t link_idx;
        tt::tt_fabric::FabricNodeId peer_node;
    };
    std::vector<EthEntry> eths;
    for (int delta : {1, -1}) {
        const auto nbr =
            coord.get_neighbor(mesh_shape, delta, static_cast<int32_t>(axis), ttnn::MeshCoordinate::BoundaryMode::WRAP);
        TT_FATAL(nbr.has_value(), "combine_fabric2d: no axis-{} neighbor of {} at delta {}", axis, coord, delta);
        if (*nbr == coord) {
            continue;  // degenerate axis; nothing to talk to
        }
        const auto nbr_node = mesh->get_fabric_node_id(*nbr);
        const auto links = tt::tt_fabric::get_forwarding_link_indices(self_node, nbr_node);
        const uint32_t n = std::min<uint32_t>(num_links, links.size());
        TT_FATAL(n > 0, "combine_fabric2d: no forwarding links from {} to {}", self_node, nbr_node);
        for (uint32_t k = 0; k < n; k++) {
            const CoreCoord eth_logical =
                tt::tt_fabric::get_forwarding_link_logical_eth_core(self_node, nbr_node, links[k]);
            const auto eth_phys = tt::tt_metal::experimental::Device::get_physical_core_from_logical_core(
                dev, eth_logical, tt::CoreType::ETH);
            eths.push_back(EthEntry{eth_logical, static_cast<uint32_t>(eth_phys.x), links[k], nbr_node});
        }
        if (n < num_links) {
            log_warning(
                tt::LogOp,
                "combine_fabric2d {}: only {} of {} requested links toward {}",
                self_node,
                n,
                num_links,
                nbr_node);
        }
    }
    // Deterministic order so the relocation fallback is reproducible.
    std::sort(
        eths.begin(), eths.end(), [](const EthEntry& a, const EthEntry& b) { return a.eth_phys_x < b.eth_phys_x; });

    // Columns that must stay free for a co-located worker: any column holding one of OUR eth cores.
    // Taking one for a relocated worker could displace the worker that belongs there.
    std::set<uint32_t> eth_columns;
    for (const auto& e : eths) {
        eth_columns.insert(e.eth_phys_x);
    }
    std::set<uint32_t> used_columns;

    auto make_placement = [&](const EthEntry& e, const CoreCoord& worker, bool in_eth_column) {
        WorkerPlacement wp;
        wp.eth_logical = e.eth_logical;
        wp.eth_phys_x = e.eth_phys_x;
        wp.link_idx = e.link_idx;
        wp.peer_node = e.peer_node;
        wp.worker_logical = worker;
        wp.worker_physical =
            tt::tt_metal::experimental::Device::get_physical_core_from_logical_core(dev, worker, tt::CoreType::WORKER);
        wp.worker_virtual = dev->virtual_core_from_logical_core(worker, tt::CoreType::WORKER);
        wp.in_eth_column = in_eth_column;
        return wp;
    };

    DevicePlacement placement;
    // Pass 1: everyone who can sit in their own eth column does.
    for (const auto& e : eths) {
        const auto w = adjacent_worker_in_column(geom, e.eth_phys_x);
        if (!w.has_value()) {
            continue;
        }
        auto wp = make_placement(e, *w, /*in_eth_column=*/true);
        TT_FATAL(
            wp.worker_physical.x == e.eth_phys_x,
            "combine_fabric2d: worker/eth column mismatch ({} vs {})",
            wp.worker_physical.x,
            e.eth_phys_x);
        used_columns.insert(e.eth_phys_x);
        placement.by_eth_logical.emplace(e.eth_logical, wp);
    }
    // Pass 2: relocate the rest to the leftmost column that is neither one of our eth columns nor
    // already taken. Runs after pass 1 so a relocated worker can never steal a co-located worker's
    // column, whatever order the eth cores come in.
    for (const auto& e : eths) {
        if (placement.by_eth_logical.count(e.eth_logical)) {
            continue;
        }
        std::optional<CoreCoord> chosen;
        uint32_t chosen_x = 0;
        for (const auto& [phys_x, rows] : geom.columns) {  // std::map => ascending x, i.e. leftmost first
            if (eth_columns.count(phys_x) || used_columns.count(phys_x) || rows.empty()) {
                continue;
            }
            chosen = rows.begin()->second;
            chosen_x = phys_x;
            break;
        }
        TT_FATAL(
            chosen.has_value(),
            "combine_fabric2d {}: eth core ({},{}) is in physical column x={} with no programmable worker, and no "
            "unoccupied column is left to relocate its worker to.",
            self_node,
            e.eth_logical.x,
            e.eth_logical.y,
            e.eth_phys_x);
        used_columns.insert(chosen_x);
        placement.by_eth_logical.emplace(e.eth_logical, make_placement(e, *chosen, /*in_eth_column=*/false));
        log_warning(
            tt::LogOp,
            "combine_fabric2d {}: eth core ({},{}) physical column x={} has no programmable worker (harvested or "
            "dispatch); relocated its worker to column x={}, adding {} NoC hop(s) to its router.",
            self_node,
            e.eth_logical.x,
            e.eth_logical.y,
            e.eth_phys_x,
            chosen_x,
            chosen_x > e.eth_phys_x ? chosen_x - e.eth_phys_x : e.eth_phys_x - chosen_x);
    }
    return placement;
}

// Lazy device -> placement cache. A device's placement must be decided before that device can serve as
// anyone's neighbor, so building the op for D also forces placement for D's cable neighbors — but NOT
// their programs, which are built on their own create call.
class PlacementCache {
public:
    PlacementCache(ttnn::MeshDevice* mesh, uint32_t axis, uint32_t num_links, const CoreCoord& compute_grid) :
        mesh_(mesh), axis_(axis), num_links_(num_links), compute_grid_(compute_grid) {}

    const DevicePlacement& get(const ttnn::MeshCoordinate& coord) {
        auto it = cache_.find(coord);
        if (it == cache_.end()) {
            it = cache_.emplace(coord, decide_placement(mesh_, coord, axis_, num_links_, compute_grid_)).first;
        }
        return it->second;
    }

private:
    ttnn::MeshDevice* mesh_;
    uint32_t axis_;
    uint32_t num_links_;
    CoreCoord compute_grid_;
    std::map<ttnn::MeshCoordinate, DevicePlacement> cache_;
};

struct L1Layout {
    uint32_t pkt_hdr_drain;
    uint32_t drain_sink;
    uint32_t telemetry;
    uint32_t ring;          // num_l1_slots tokens, filled by the reader and drained by the producer
    uint32_t pkt_hdr_ring;  // one prebuilt payload header per ring slot
    // Phase 10: the reader's copy of the control tensors, read once at startup and then indexed from L1.
    // Laid out as [expert_offsets: dispatch_group_size x num_routed_experts][counts: num_routed_experts]
    // [region_offsets: num_routed_experts], all uint32. Unlike everything above it, nothing on another chip
    // addresses this, so it can sit at the end — but it is still computed identically everywhere.
    uint32_t control;
};

// The telemetry address depends only on the L1 base, so the readback path can resolve it without any of
// the run's sizing.
uint32_t telemetry_addr(ttnn::MeshDevice* mesh) {
    return static_cast<uint32_t>(mesh->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1)) +
           TELEMETRY_OFF;
}

L1Layout compute_l1_layout(
    ttnn::MeshDevice* mesh,
    uint32_t num_l1_slots,
    uint32_t token_size_bytes,
    uint32_t control_bytes,
    uint32_t sem_floor) {
    const uint32_t base =
        static_cast<uint32_t>(mesh->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1));
    L1Layout l;
    l.pkt_hdr_drain = base + PKT_HDR_DRAIN_OFF;
    l.drain_sink = base + DRAIN_SINK_OFF;
    l.telemetry = base + TELEMETRY_OFF;
    l.ring = base + PROD_BUF_OFF;
    // One prebuilt header per ring slot, past the ring itself. A slot is the token plus its metadata tail.
    l.pkt_hdr_ring = l.ring + num_l1_slots * (token_size_bytes + CMBF2D_SLOT_TAIL_BYTES);
    const uint32_t hdr_ring_bytes =
        num_l1_slots * static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_packet_header_size_bytes());
    // 64-byte aligned: DRAM reads need a DRAM_ALIGNMENT-aligned L1 destination on Blackhole
    // (LOG_BASE_2_OF_DRAM_ALIGNMENT = 6), and the control region is read straight out of DRAM.
    l.control = (l.pkt_hdr_ring + hdr_ring_bytes + 63u) & ~63u;
    const uint32_t end = l.control + control_bytes;
    TT_FATAL(
        end + L1_SLACK <= sem_floor,
        "combine_fabric2d: L1 layout needs {} B (ends at 0x{:x}) but the global-semaphore region starts at "
        "0x{:x}. Reduce num_l1_slots ({}) or the token page ({} B).",
        end - base,
        end,
        sem_floor,
        num_l1_slots,
        token_size_bytes);
    return l;
}

// ---------------------------------------------------------------------------------------------
// Destination -> producer assignment
//
// One unit of work a single producer executes: everything this chip owes ONE destination chip, narrowed to
// the slice of each run this producer is responsible for. Where those tokens live and how many there are is
// NOT known here — that is in the control tensors, and the reader works it out on device. All this struct
// carries is which destination and which slice.
//
// The slice is the fraction (split_idx, split_count): of a run of n tokens the producer takes
// [n*split_idx/split_count, n*(split_idx+1)/split_count). (0,1) is the whole run, (0,2)/(1,2) the halves,
// (0..3,4) the quarters. Deliberately a fraction pair rather than an enum of named cases, so stage 3 can
// try uneven or per-expert splits without touching the kernel.
// ---------------------------------------------------------------------------------------------
struct ProducerAssignment {
    uint32_t dst_chip_id = 0;
    uint32_t dst_row = 0;      // destination's index along `axis`, i.e. its row in expert_offsets
    uint32_t ring_offset = 0;  // hops from this chip along +axis (extent-k means k hops backward)
    uint32_t split_idx = 0;
    uint32_t split_count = 1;
};

// Ring offset from `from` to `to` along `axis`, in [0, extent).
uint32_t ring_offset(const ttnn::MeshCoordinate& from, const ttnn::MeshCoordinate& to, uint32_t axis, uint32_t extent) {
    return (to[static_cast<int32_t>(axis)] + extent - from[static_cast<int32_t>(axis)]) % extent;
}

// Split this device's destinations across its producers.
//
// Since P9.2 the op does its own forwarding: every packet travels exactly ONE hop, to the chip across the
// sending producer's own cable, and the next hop is decided by the downstream reader. So the DIRECTION a
// destination is served from is ours to choose — it is simply which producer we hand it to. (In P8 it was
// not: `fabric_set_unicast_route` stamped a precomputed per-destination path, and injecting on a cable that
// was not that path's first hop landed the packet on the wrong chip.)
//
// With R chips on the ring and H = R/2:
//   * the producer whose cable steps +1 takes ring offsets +1 .. +(H-1);
//   * the one stepping -1 takes -1 .. -(H-1);
//   * the offset-H destination (diametrically opposite, equally far either way) is shared by ALL
//     2*num_links producers — a quarter of every run each.
// Within one ring offset the `num_links` planes split each run between them.
//
// That balances the link load at H^2/2 tokens per direction instead of H(H+1)/2 and H(H-1)/2 — 8x instead
// of 10x and 6x on an 8-ring, a 20% cut in the bottleneck. Every producer serves m = H destinations, so the
// forwarding geometry (incoming = m(m-1)/2 chunks per quarter, own forwarding m-1, re-forwarded
// (m-1)(m-2)/2) is uniform too.
//
// The same-chip destination (ring offset 0) is deliberately absent: it needs no cable, so it is not a
// producer's work. Each core does its quarter of it directly, after the fabric work — see the local phase
// in the reader.
//
// Phase 10: nothing here reads the control tensors. The split is pure ring geometry plus a fraction, which
// is what lets it stay a host-side decision even though the token counts are only known on device.
std::map<CoreCoord, std::vector<ProducerAssignment>> assign_destinations_to_producers(
    const CombineFabric2dParams& args,
    const ttnn::MeshCoordinate& coord,
    const DevicePlacement& self_placement,
    const std::map<CoreCoord, ttnn::MeshCoordinate>& far_coord_by_eth,
    const std::map<ttnn::MeshCoordinate, tt::tt_fabric::FabricNodeId>& node_by_coord,
    uint32_t axis,
    uint32_t extent) {
    const uint32_t half = extent / 2;

    std::map<CoreCoord, std::vector<ProducerAssignment>> out;
    for (const auto& [eth_logical, wp] : self_placement.by_eth_logical) {
        const auto fit = far_coord_by_eth.find(eth_logical);
        TT_FATAL(fit != far_coord_by_eth.end(), "combine_fabric2d {}: no far coord resolved for an eth core", coord);
        const uint32_t step = ring_offset(coord, fit->second, axis, extent);
        TT_FATAL(
            step == 1 || step == extent - 1,
            "combine_fabric2d {}: eth core ({},{}) cables to a chip {} hops away along axis {}; the op needs "
            "every cable to reach an immediate ring neighbour",
            coord,
            eth_logical.x,
            eth_logical.y,
            step,
            axis);
        const bool forward = (step == 1);

        auto& list = out[eth_logical];
        auto claim = [&](uint32_t offset, uint32_t split_idx, uint32_t split_count) {
            ttnn::MeshCoordinate dst_coord = coord;
            dst_coord[static_cast<int32_t>(axis)] = (coord[static_cast<int32_t>(axis)] + offset) % extent;
            const auto& dst_node = node_by_coord.at(dst_coord);
            list.push_back(ProducerAssignment{
                .dst_chip_id = static_cast<uint32_t>(dst_node.chip_id),
                .dst_row = dst_coord[static_cast<int32_t>(axis)],
                .ring_offset = offset,
                .split_idx = split_idx,
                .split_count = split_count});
        };

        // Nearest destination first. The work schedule below reorders this if assignment_order asks, which
        // is why the order is not baked into the kernel.
        for (uint32_t k = 1; k < half; k++) {
            claim(forward ? k : extent - k, wp.link_idx, args.num_links);
        }
        // The opposite chip: (plane, direction) picks which quarter of each run this producer takes.
        claim(half, wp.link_idx * 2 + (forward ? 0u : 1u), args.num_links * 2);
        TT_FATAL(
            list.size() == half,
            "combine_fabric2d {}: producer got {} assignment(s), expected {}",
            coord,
            list.size(),
            half);
    }
    return out;
}

// Per-coordinate program: one reader+producer pair per fabric eth core of this device, on a worker core
// in that eth core's physical column, each owning that eth channel's single fabric connection.
//
// Each producer serves SEVERAL destinations (see assign_destinations_to_producers): its plane's slice of
// everything this chip owes every other chip on the axis, plus its quarter of the same-chip copy. What that
// amounts to in tokens is discovered on device from the control tensors.
tt::tt_metal::ProgramDescriptor build_program_for_coord(
    const CombineFabric2dParams& args,
    const ttnn::MeshCoordinate& coord,
    PlacementCache& placements,
    const std::map<uint32_t, ttnn::MeshCoordinate>& chip_to_coord,
    const L1Layout& l1,
    uint32_t batch,
    uint32_t ring_filled_addr,
    uint32_t ring_freed_addr,
    uint32_t fwd_arrived_addr,
    uint32_t token_size_bytes,
    uint32_t num_routed_experts,
    uint32_t pages_per_chunk,
    tt::tt_metal::Buffer* dram_out_buf,
    tt::tt_metal::Buffer* dram_in_buf,
    tt::tt_metal::Buffer* dram_fwd_buf,
    tt::tt_metal::Buffer* dram_meta_buf,
    tt::tt_metal::Buffer* dram_counts_buf,
    tt::tt_metal::Buffer* dram_region_buf,
    tt::tt_metal::Buffer* dram_expert_offsets_buf) {
    tt::tt_metal::ProgramDescriptor desc;
    auto* mesh = args.device;
    auto* dev = mesh->get_device(coord);
    const auto self_node = mesh->get_fabric_node_id(coord);
    const uint32_t dram_out_addr = static_cast<uint32_t>(dram_out_buf->address());
    const uint32_t dram_in_addr = static_cast<uint32_t>(dram_in_buf->address());
    // Op-internal forwarding staging. Passed to both kernels from P9.1 so the plumbing is in place and
    // verified; P9.2 is what starts writing to it.
    const uint32_t dram_fwd_addr = static_cast<uint32_t>(dram_fwd_buf->address());
    const uint32_t dram_meta_addr = static_cast<uint32_t>(dram_meta_buf->address());
    const uint32_t dram_counts_addr = static_cast<uint32_t>(dram_counts_buf->address());
    const uint32_t dram_region_addr = static_cast<uint32_t>(dram_region_buf->address());
    const uint32_t dram_expert_offsets_addr = static_cast<uint32_t>(dram_expert_offsets_buf->address());
    const uint32_t axis = args.axis;
    const uint32_t extent = mesh->shape()[static_cast<int32_t>(axis)];

    // Which of the `num_routed_experts` columns this chip hosts, and where it sits on the ring. The dispatch
    // group is this device's position on the OTHER mesh axis; with one group per column of a 2D mesh that is
    // just the other coordinate. Same derivation as the production reader's compile-time `offset`.
    const uint32_t my_row = coord[static_cast<int32_t>(axis)];
    const uint32_t other_axis = axis == 0 ? 1 : 0;
    const uint32_t experts_per_group = args.experts_per_chip * args.dispatch_group_size;
    const uint32_t num_dispatch_groups = num_routed_experts / experts_per_group;
    const uint32_t my_group =
        mesh->shape().dims() > 1 ? coord[static_cast<int32_t>(other_axis)] % num_dispatch_groups : 0u;
    const uint32_t my_expert_base = my_group * experts_per_group + my_row * args.experts_per_chip;

    const auto& self_placement = placements.get(coord);

    // ---- Phase A: resolve, for every eth core, the chip at the far end of its cable and the worker there.
    // Cable truth, not plane-index arithmetic: our producer writes into this eth core's EDM, so a
    // single-hop packet physically emerges at the far end of THIS cable.
    std::map<CoreCoord, ttnn::MeshCoordinate> far_coord_by_eth;
    std::map<CoreCoord, CoreCoord> peer_virtual_by_eth;
    for (const auto& [eth_logical, wp] : self_placement.by_eth_logical) {
        const auto far = dev->get_connected_ethernet_core(eth_logical);
        const uint32_t far_chip = static_cast<uint32_t>(std::get<0>(far));
        const CoreCoord far_eth_logical = std::get<1>(far);
        auto cit = chip_to_coord.find(far_chip);
        TT_FATAL(
            cit != chip_to_coord.end(),
            "combine_fabric2d {}: eth core ({},{}) cables to chip {}, which is not in this mesh",
            self_node,
            eth_logical.x,
            eth_logical.y,
            far_chip);
        // Forces the neighbor's placement if not yet decided. We only READ it; the neighbor's programs
        // are built on its own create call.
        const auto& peer_placement = placements.get(cit->second);
        auto pit = peer_placement.by_eth_logical.find(far_eth_logical);
        TT_FATAL(
            pit != peer_placement.by_eth_logical.end(),
            "combine_fabric2d {}: our eth core ({},{}) cables to eth core ({},{}) on {}, but that core is not in the "
            "neighbor's fabric eth set, so it has no worker. Link selection disagrees across the cable.",
            self_node,
            eth_logical.x,
            eth_logical.y,
            far_eth_logical.x,
            far_eth_logical.y,
            mesh->get_fabric_node_id(cit->second));
        far_coord_by_eth.emplace(eth_logical, cit->second);
        peer_virtual_by_eth.emplace(eth_logical, pit->second.worker_virtual);
    }

    // Find, on chip `nbr`, the worker whose cable points the same way (`step`) on the same plane. That is
    // the reader which will drain the forwarding-buffer quarter our producer writes — NOT the worker at the
    // far end of our own cable, whose cable points back at us.
    auto downstream_same_direction_worker =
        [&](const ttnn::MeshCoordinate& nbr, uint32_t step, uint32_t link) -> CoreCoord {
        const auto& nbr_placement = placements.get(nbr);
        ttnn::MeshCoordinate want = nbr;
        want[static_cast<int32_t>(axis)] = (nbr[static_cast<int32_t>(axis)] + step) % extent;
        auto* nbr_dev = mesh->get_device(nbr);
        for (const auto& [eth2, wp2] : nbr_placement.by_eth_logical) {
            if (wp2.link_idx != link) {
                continue;
            }
            const auto far2 = nbr_dev->get_connected_ethernet_core(eth2);
            const auto it = chip_to_coord.find(static_cast<uint32_t>(std::get<0>(far2)));
            if (it != chip_to_coord.end() && it->second == want) {
                return wp2.worker_virtual;
            }
        }
        TT_FATAL(
            false,
            "combine_fabric2d: chip {} has no plane-{} cable continuing to {}, so the forwarding chain has "
            "nowhere to go",
            nbr,
            link,
            want);
        return CoreCoord{0, 0};
    };

    // ---- Phase B: split this device's destinations across its producers.
    std::map<ttnn::MeshCoordinate, tt::tt_fabric::FabricNodeId> node_by_coord;
    for (const auto& c : ttnn::MeshCoordinateRange(mesh->shape())) {
        node_by_coord.emplace(c, mesh->get_fabric_node_id(c));
    }
    const auto assignments =
        assign_destinations_to_producers(args, coord, self_placement, far_coord_by_eth, node_by_coord, axis, extent);

    // Coverage: every remote destination's every run must be claimed exactly once, in whole. Token counts
    // are unknown here, so the check is on the FRACTIONS — for each ring offset the claimed slices must be
    // a partition of [0,1), which for split_idx in [0,split_count) means the indices are distinct and there
    // are exactly split_count of them. That is the property the subdivision exists to preserve, and it is
    // checkable without knowing how many tokens any run holds.
    {
        std::map<uint32_t, std::set<uint32_t>> claimed_by_offset;
        std::map<uint32_t, uint32_t> split_count_by_offset;
        for (const auto& [eth, list] : assignments) {
            for (const auto& a : list) {
                const auto [_, fresh] = claimed_by_offset[a.ring_offset].insert(a.split_idx);
                TT_FATAL(
                    fresh,
                    "combine_fabric2d {}: two producers both claim slice {} of {} for ring offset {}",
                    coord,
                    a.split_idx,
                    a.split_count,
                    a.ring_offset);
                TT_FATAL(
                    a.split_idx < a.split_count,
                    "combine_fabric2d {}: slice {} is out of range for a {}-way split",
                    coord,
                    a.split_idx,
                    a.split_count);
                auto& want = split_count_by_offset[a.ring_offset];
                TT_FATAL(
                    want == 0 || want == a.split_count,
                    "combine_fabric2d {}: ring offset {} is split {} ways by one producer and {} ways by "
                    "another; the slices would not tile the run",
                    coord,
                    a.ring_offset,
                    want,
                    a.split_count);
                want = a.split_count;
            }
        }
        TT_FATAL(
            claimed_by_offset.size() == extent - 1,
            "combine_fabric2d {}: producers cover {} of the {} remote chips on axis {}",
            coord,
            claimed_by_offset.size(),
            extent - 1,
            axis);
        for (const auto& [offset, slices] : claimed_by_offset) {
            TT_FATAL(
                slices.size() == split_count_by_offset.at(offset),
                "combine_fabric2d {}: ring offset {} has {} of its {} slices claimed, so part of every run "
                "to that chip would never be sent",
                coord,
                offset,
                slices.size(),
                split_count_by_offset.at(offset));
        }
    }

    std::string summary;
    for (const auto& [eth_logical, wp] : self_placement.by_eth_logical) {
        const ttnn::MeshCoordinate& far_coord = far_coord_by_eth.at(eth_logical);
        const CoreCoord peer_virtual = peer_virtual_by_eth.at(eth_logical);
        const auto& my_assignments = assignments.at(eth_logical);

        // Phase-9 forwarding geometry for this producer. m = how many destinations it serves (distances
        // 1..m in its direction), and everything else follows:
        //   incoming chunks   = m(m-1)/2   (a chunk per (source, destination) pair passing through us)
        //   own forwarding    = m-1        (distances 2..m)
        //   re-forwarded      = (m-1)(m-2)/2
        // own + re-forwarded == incoming, which is what makes the upstream writer and downstream reader
        // agree on how many chunks a quarter holds without exchanging anything.
        const uint32_t m = static_cast<uint32_t>(my_assignments.size());
        const uint32_t num_incoming_chunks = m * (m - 1) / 2;
        TT_FATAL(
            num_incoming_chunks <= fwd_chunks_per_quarter(extent),
            "combine_fabric2d {}: producer expects {} incoming chunks but a quarter only holds {}",
            coord,
            num_incoming_chunks,
            fwd_chunks_per_quarter(extent));
        // Quarter index from (plane, direction). Both the upstream producer that writes it and the
        // downstream reader that drains it compute this the same way, which is the whole point.
        const uint32_t step = ring_offset(coord, far_coord, axis, extent);
        const uint32_t my_quarter = wp.link_idx * 2 + (step == 1 ? 0u : 1u);
        // The downstream worker continuing in the SAME direction on the SAME plane.
        const CoreCoord fwd_worker = downstream_same_direction_worker(far_coord, step, wp.link_idx);

        // Work schedule: one entry per unit of work, high bit set = "relay forwarding chunk k", clear =
        // "own assignment k". Forwarding chunks MUST appear in increasing order whatever the ordering, since
        // the upstream writer fills a quarter's chunks in order and we read them the same way.
        std::vector<uint32_t> schedule;
        constexpr uint32_t SCHED_FWD = 0x80000000u;
        if (args.assignment_order == 0) {
            // Nearest destination first, then every forwarding chunk.
            for (uint32_t i = 0; i < m; i++) {
                schedule.push_back(i);
            }
            for (uint32_t c = 0; c < num_incoming_chunks; c++) {
                schedule.push_back(SCHED_FWD | c);
            }
        } else {
            // Furthest first with forwarding interleaved, so downstream cores get work as early as possible.
            // After the j-th own assignment (1-based) the cumulative number of forwarding chunks emitted is
            // the triangular number T(j-2) = (j-2)(j-1)/2 — which for m=4 reproduces exactly
            //   own3, own2, fwd0, own1, fwd1, fwd2, own0, fwd3, fwd4, fwd5.
            uint32_t emitted_fwd = 0;
            for (uint32_t j = 1; j <= m; j++) {
                schedule.push_back(m - j);  // own assignments furthest -> nearest
                const uint32_t want = (j >= 2) ? (j - 2) * (j - 1) / 2 : 0;
                for (; emitted_fwd < want && emitted_fwd < num_incoming_chunks; emitted_fwd++) {
                    schedule.push_back(SCHED_FWD | emitted_fwd);
                }
            }
            for (; emitted_fwd < num_incoming_chunks; emitted_fwd++) {
                schedule.push_back(SCHED_FWD | emitted_fwd);
            }
        }
        TT_FATAL(
            schedule.size() == m + num_incoming_chunks,
            "combine_fabric2d {}: schedule has {} entries, expected {} own + {} forwarding",
            coord,
            schedule.size(),
            m,
            num_incoming_chunks);

        // ---- Producer (writer RISC, NOC_1): drains the L1 ring to the destination chips' DRAM over fabric.
        tt::tt_metal::KernelDescriptor prod;
        prod.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/kernels/dataflow/"
            "producer_combine_fabric2d.cpp";
        prod.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
        prod.core_ranges = CoreRangeSet(CoreRange(wp.worker_logical));
        prod.compile_time_args = {
            args.num_l1_slots,
            token_size_bytes,
            CMBF2D_SLOT_TAIL_BYTES,
            static_cast<uint32_t>(wp.peer_node.chip_id),
            *wp.peer_node.mesh_id,
            static_cast<uint32_t>(peer_virtual.x),
            static_cast<uint32_t>(peer_virtual.y),
            l1.ring,
            l1.pkt_hdr_ring,
            l1.pkt_hdr_drain,
            l1.drain_sink,
            l1.telemetry,
            args.stall_telemetry,
            dram_out_addr,
            batch,
            ring_filled_addr,
            ring_freed_addr,
            static_cast<uint32_t>(wp.worker_virtual.x),
            static_cast<uint32_t>(wp.worker_virtual.y),
            // The downstream chip's worker that continues in OUR direction on OUR plane: the reader that
            // drains the forwarding quarter we write. Distinct from `peer_virtual`, whose cable points back.
            static_cast<uint32_t>(fwd_worker.x),
            static_cast<uint32_t>(fwd_worker.y),
            fwd_arrived_addr,
            args.fwd_bump_every,
        };
        // The producer no longer needs the assignment table: every send is one hop to its own cable's peer,
        // and both the command and the destination address arrive per token in the slot's metadata tail.
        // TensorAccessorArgs for the interleaved output buffer (compile-time config).
        tt::tt_metal::TensorAccessorArgs(dram_out_buf).append_to(prod.compile_time_args);
        prod.config = tt::tt_metal::DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            // NOC_1 routes -Y first, so worker (eth row + 1) -> eth core is a single hop.
            .noc = tt::tt_metal::NOC::NOC_1,
        };
        auto prod_id = static_cast<tt::tt_metal::KernelHandle>(desc.kernels.size());
        desc.kernels.push_back(std::move(prod));

        // ---- Reader (reader RISC, NOC_0): streams each assignment's tokens from local DRAM into the ring.
        // Separate NoC from the producer's eth sends, so the two do not contend.
        tt::tt_metal::KernelDescriptor rdr;
        rdr.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/kernels/dataflow/"
            "reader_combine_fabric2d.cpp";
        rdr.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
        rdr.core_ranges = CoreRangeSet(CoreRange(wp.worker_logical));
        rdr.compile_time_args = {
            args.num_l1_slots,
            token_size_bytes,
            CMBF2D_SLOT_TAIL_BYTES,
            batch,
            l1.ring,
            ring_filled_addr,
            ring_freed_addr,
            static_cast<uint32_t>(wp.worker_virtual.x),
            static_cast<uint32_t>(wp.worker_virtual.y),
            l1.telemetry,
            dram_in_addr,
            dram_out_addr,
            dram_fwd_addr,
            fwd_chunks_per_quarter(extent),
            pages_per_chunk,
            my_quarter,
            num_incoming_chunks,
            fwd_arrived_addr,
            static_cast<uint32_t>(wp.peer_node.chip_id),  // the chip one hop away, across our own cable
            static_cast<uint32_t>(my_assignments.size()),
            static_cast<uint32_t>(schedule.size()),
            // ---- Phase 10: what to move is in the control tensors, so the reader needs their addresses and
            // the shape constants to index them with.
            dram_meta_addr,
            dram_counts_addr,
            dram_region_addr,
            dram_expert_offsets_addr,
            num_routed_experts,
            args.experts_per_chip,
            // First of the `experts_per_chip` columns this chip hosts. Same derivation the production reader
            // does (reader_combine.cpp): group * experts_per_chip * dispatch_group_size + row *
            // experts_per_chip, where the group is this device's column and the row its index along `axis`.
            my_expert_base,
            args.num_experts_per_tok,
            args.dispatch_group_size,
            // Our quarter of the same-chip run, executed after the fabric work. Every core takes a slice so
            // the local copy is not one core's serial tail.
            my_quarter,
            args.num_links * 2,
            static_cast<uint32_t>(my_row),
            l1.control,
        };
        // Schedule block, then the per-assignment block. The schedule says WHEN each own assignment and each
        // relayed forwarding chunk is worked on; the assignment block says WHICH destination each own
        // assignment serves and which slice of its runs this producer owns.
        for (uint32_t e : schedule) {
            rdr.compile_time_args.push_back(e);
        }
        // Per-assignment block: [dst_chip_id, dst_row, split_idx, split_count] x num_assignments. No token
        // ranges: the reader derives those from expert_offsets on device.
        for (const auto& a : my_assignments) {
            rdr.compile_time_args.push_back(a.dst_chip_id);
            rdr.compile_time_args.push_back(a.dst_row);
            rdr.compile_time_args.push_back(a.split_idx);
            rdr.compile_time_args.push_back(a.split_count);
        }
        // Accessors, in this order: dispatched buffer (local token reads), output (final-address
        // arithmetic), forwarding buffer (local reads AND next-hop address arithmetic), then the three
        // control tensors the reader reads once at startup.
        tt::tt_metal::TensorAccessorArgs(dram_in_buf).append_to(rdr.compile_time_args);
        tt::tt_metal::TensorAccessorArgs(dram_out_buf).append_to(rdr.compile_time_args);
        tt::tt_metal::TensorAccessorArgs(dram_fwd_buf).append_to(rdr.compile_time_args);
        tt::tt_metal::TensorAccessorArgs(dram_meta_buf).append_to(rdr.compile_time_args);
        tt::tt_metal::TensorAccessorArgs(dram_counts_buf).append_to(rdr.compile_time_args);
        tt::tt_metal::TensorAccessorArgs(dram_region_buf).append_to(rdr.compile_time_args);
        tt::tt_metal::TensorAccessorArgs(dram_expert_offsets_buf).append_to(rdr.compile_time_args);
        rdr.config = tt::tt_metal::DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::NOC_0,
        };
        desc.kernels.push_back(std::move(rdr));  // no fabric connection => no rt args
        {
            std::vector<uint32_t> rt_raw;
            rt_raw.push_back(1u);  // num_connections
            const std::vector<tt::tt_fabric::FabricNodeId> dst_nodes = {wp.peer_node};
            const std::vector<uint32_t> conn_links = {wp.link_idx};
            tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
                self_node, dst_nodes, conn_links, desc, prod_id, wp.worker_logical, rt_raw);
            tt::tt_metal::KernelDescriptor::RTArgList rt;
            rt.append(rt_raw);
            desc.kernels[prod_id].emplace_runtime_args(wp.worker_logical, rt);
        }

        std::string alist;
        for (const auto& a : my_assignments) {
            alist += fmt::format(
                "{}off{} row{} chip{} slice {}/{}",
                alist.empty() ? "" : ",",
                a.ring_offset,
                a.dst_row,
                a.dst_chip_id,
                a.split_idx,
                a.split_count);
        }
        summary += fmt::format(
            "{}[eth({},{}) phys_x {} link {} -> worker logical ({},{}) phys ({},{}){} nbr {} virt ({},{}) | {}]",
            summary.empty() ? "" : " ",
            eth_logical.x,
            eth_logical.y,
            wp.eth_phys_x,
            wp.link_idx,
            wp.worker_logical.x,
            wp.worker_logical.y,
            wp.worker_physical.x,
            wp.worker_physical.y,
            wp.in_eth_column ? "" : " RELOCATED",
            far_coord,
            peer_virtual.x,
            peer_virtual.y,
            alist);
    }

    log_info(
        tt::LogOp,
        "combine_fabric2d {} {}: {} reader+producer pair(s): {}",
        coord,
        self_node,
        self_placement.by_eth_logical.size(),
        summary);
    return desc;
}

}  // namespace

CombineFabric2dTelemetry read_telemetry(ttnn::MeshDevice* mesh_device, uint32_t num_links, uint32_t axis) {
    TT_FATAL(mesh_device != nullptr, "combine_fabric2d read_telemetry: mesh_device is null");
    const auto mesh_shape = mesh_device->shape();
    TT_FATAL(axis < mesh_shape.dims(), "combine_fabric2d: axis {} out of range for mesh shape {}", axis, mesh_shape);

    const auto grid = mesh_device->compute_with_storage_grid_size();
    const uint32_t addr = telemetry_addr(mesh_device);

    CombineFabric2dTelemetry out;
    out.clock_mhz = static_cast<uint32_t>(mesh_device->get_devices().front()->get_clock_rate_mhz());

    // Physical chip id -> mesh coordinate, so a worker's record can name the chip its tokens landed on in
    // the same coordinate space the caller uses to index per-device tensor shards.
    std::map<uint32_t, ttnn::MeshCoordinate> chip_to_coord;
    for (const auto& c : ttnn::MeshCoordinateRange(mesh_shape)) {
        chip_to_coord.emplace(static_cast<uint32_t>(mesh_device->get_device(c)->id()), c);
    }

    for (const auto& coord : ttnn::MeshCoordinateRange(mesh_shape)) {
        auto* dev = mesh_device->get_device(coord);
        // Placement is a pure function of (device, axis, num_links), so recomputing it here reproduces
        // exactly the cores the op programmed — no state has to survive from the run.
        const auto placement = decide_placement(mesh_device, coord, axis, num_links, grid);
        for (const auto& [eth_logical, wp] : placement.by_eth_logical) {
            CombineFabric2dWorkerTelemetry w;
            w.device_id = static_cast<uint32_t>(dev->id());
            w.mesh_coord.assign(coord.coords().begin(), coord.coords().end());
            w.worker_logical = wp.worker_logical;
            w.worker_physical = wp.worker_physical;
            w.eth_logical = eth_logical;
            w.eth_phys_x = wp.eth_phys_x;
            w.link_idx = wp.link_idx;
            w.relocated = !wp.in_eth_column;
            w.peer_mesh_id = *wp.peer_node.mesh_id;
            w.peer_chip_id = static_cast<uint32_t>(wp.peer_node.chip_id);
            // The chip this worker's tokens actually land on is the far end of ITS cable — the same
            // resolution the program factory used to pick the destination page range.
            const auto far = dev->get_connected_ethernet_core(eth_logical);
            const auto fit = chip_to_coord.find(static_cast<uint32_t>(std::get<0>(far)));
            if (fit != chip_to_coord.end()) {
                w.peer_coord.assign(fit->second.coords().begin(), fit->second.coords().end());
            }

            std::vector<uint32_t> words;
            const bool ok = tt::tt_metal::detail::ReadFromDeviceL1(
                dev, wp.worker_logical, addr, TELEM_NUM_WORDS * sizeof(uint32_t), words);
            if (ok && words.size() >= TELEM_NUM_WORDS && words[TELEM_MAGIC] == TELEMETRY_MAGIC) {
                w.valid = true;
                w.tokens_sent = words[TELEM_TOKENS_SENT];
                w.token_size_bytes = words[TELEM_TOKEN_SIZE];
                w.num_l1_slots = words[TELEM_NUM_L1_SLOTS];
                w.batch = words[TELEM_BATCH];
                w.t_start = static_cast<uint64_t>(words[TELEM_T_START_LO]) |
                            (static_cast<uint64_t>(words[TELEM_T_START_HI]) << 32);
                w.t_first_send = static_cast<uint64_t>(words[TELEM_T_FIRST_SEND_LO]) |
                                 (static_cast<uint64_t>(words[TELEM_T_FIRST_SEND_HI]) << 32);
                w.t_last_send = static_cast<uint64_t>(words[TELEM_T_LAST_SEND_LO]) |
                                (static_cast<uint64_t>(words[TELEM_T_LAST_SEND_HI]) << 32);
                w.t_drained = static_cast<uint64_t>(words[TELEM_T_DRAINED_LO]) |
                              (static_cast<uint64_t>(words[TELEM_T_DRAINED_HI]) << 32);
                w.edm_slots = words[TELEM_EDM_SLOTS];
                w.drain_packets = words[TELEM_DRAIN_PACKETS];
                w.out_base_page = words[TELEM_OUT_BASE_PAGE];
                w.wait_slot_cycles = static_cast<uint64_t>(words[TELEM_WAIT_SLOT_CY_LO]) |
                                     (static_cast<uint64_t>(words[TELEM_WAIT_SLOT_CY_HI]) << 32);
                w.issue_cycles = static_cast<uint64_t>(words[TELEM_ISSUE_CY_LO]) |
                                 (static_cast<uint64_t>(words[TELEM_ISSUE_CY_HI]) << 32);
                w.ring_wait_cycles = static_cast<uint64_t>(words[TELEM_RING_WAIT_CY_LO]) |
                                     (static_cast<uint64_t>(words[TELEM_RING_WAIT_CY_HI]) << 32);
                w.t_kernel_start = static_cast<uint64_t>(words[TELEM_T_KERNEL_START_LO]) |
                                   (static_cast<uint64_t>(words[TELEM_T_KERNEL_START_HI]) << 32);
                w.t_kernel_end = static_cast<uint64_t>(words[TELEM_T_KERNEL_END_LO]) |
                                 (static_cast<uint64_t>(words[TELEM_T_KERNEL_END_HI]) << 32);
            }
            out.workers.push_back(std::move(w));
        }
    }
    return out;
}

tt::tt_metal::WorkloadDescriptor CombineFabric2dProgramFactory::create_workload_descriptor(
    const CombineFabric2dParams& operation_attributes,
    const CombineFabric2dInputs& tensor_args,
    ttnn::Tensor& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    auto* mesh_device = operation_attributes.device;
    const auto mesh_shape = mesh_device->shape();
    const uint32_t axis = operation_attributes.axis;
    TT_FATAL(axis < mesh_shape.dims(), "combine_fabric2d: axis {} out of range for mesh shape {}", axis, mesh_shape);
    TT_FATAL(
        mesh_shape[axis] > 1,
        "combine_fabric2d: mesh axis {} has extent {}; need at least 2 chips to send anywhere",
        axis,
        mesh_shape[axis]);

    const uint32_t extent = mesh_shape[static_cast<int32_t>(axis)];
    TT_FATAL(
        extent % 2 == 0,
        "combine_fabric2d: axis {} extent {} must be even — the all-destinations pattern relies on a "
        "diametrically-opposite chip at ring offset extent/2",
        axis,
        extent);

    // A token is one page of the dispatched buffer — the op does not take a token size, it reads it off the
    // tensor the caller staged, exactly as the production op does.
    auto* dram_in_buf = tensor_args.dispatched_buffer.buffer();
    TT_FATAL(dram_in_buf != nullptr, "combine_fabric2d: dispatched_buffer has no device buffer");
    const uint32_t token_size_bytes = static_cast<uint32_t>(dram_in_buf->aligned_page_size());
    TT_FATAL(
        token_size_bytes % sizeof(uint32_t) == 0,
        "combine_fabric2d: token page {} B must be a multiple of 4",
        token_size_bytes);

    const uint32_t fabric_max_payload = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    // From phase 9 a forwarded packet carries the token PLUS its routing tail, so the payload the fabric
    // must accept is token + tail, not just token.
    TT_FATAL(
        token_size_bytes + CMBF2D_SLOT_TAIL_BYTES <= fabric_max_payload,
        "combine_fabric2d: token page {} B + {} B routing tail exceeds the fabric max payload {}. Raise "
        "max_payload_size in the device's fabric_router_config.",
        token_size_bytes,
        CMBF2D_SLOT_TAIL_BYTES,
        fabric_max_payload);

    const auto grid = mesh_device->compute_with_storage_grid_size();
    // The reader/producer ring handshake is two monotonic single-writer counters. They live in
    // GlobalSemaphores rather than in the op's own L1 region for one reason: the framework ZEROES them
    // before launch. Raw L1 keeps whatever the previous program left there, and a stale `freed` underflows
    // the reader's free-slot arithmetic — which is a silent buffer overwrite, not a clean failure.
    // Allocated on the full worker grid so the addresses are uniform across the mesh.
    const CoreRangeSet all_workers(CoreRange(CoreCoord{0, 0}, CoreCoord{grid.x - 1, grid.y - 1}));
    auto ring_filled_sem =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, all_workers, 0, tt::tt_metal::BufferType::L1);
    auto ring_freed_sem =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, all_workers, 0, tt::tt_metal::BufferType::L1);
    // Forwarding arrivals: bumped by the UPSTREAM chip's producer, polled by this chip's reader on the same
    // (plane, direction). ONE semaphore suffices for all four quarters — each quarter is drained by a
    // different worker core, so the per-core copy at this uniform L1 offset already separates them, and the
    // producer simply targets the right core.
    auto fwd_arrived_sem =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, all_workers, 0, tt::tt_metal::BufferType::L1);
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, {});
    const uint32_t ring_filled_addr = static_cast<uint32_t>(ring_filled_sem.address());
    const uint32_t ring_freed_addr = static_cast<uint32_t>(ring_freed_sem.address());
    const uint32_t fwd_arrived_addr = static_cast<uint32_t>(fwd_arrived_sem.address());
    // The reader's L1 copy of the control tensors: the expert_offsets slice (one row per origin chip) plus
    // the counts and region offsets. A few kB, read once at startup and indexed from L1 thereafter.
    // Plus one 64-byte-aligned landing pad at the front for the per-token metadata record: a DRAM read
    // needs a 64-byte-aligned L1 destination on Blackhole, which no offset inside a ring slot's tail can
    // give (the tail starts at token_size, and its free half is only 32-byte aligned).
    const uint32_t control_num_routed_experts =
        static_cast<uint32_t>(tensor_args.expert_token_counts.logical_shape()[-1]);
    const uint32_t control_bytes = 64 + static_cast<uint32_t>(sizeof(uint32_t)) * control_num_routed_experts *
                                            (operation_attributes.dispatch_group_size + 2);
    const auto l1 = compute_l1_layout(
        mesh_device,
        operation_attributes.num_l1_slots,
        token_size_bytes,
        control_bytes,
        std::min(std::min(ring_filled_addr, ring_freed_addr), fwd_arrived_addr));
    // Slots move between the reader and the producer in half-ring batches, so one half can be refilled
    // while the other drains. This is the knob that amortises the ring bookkeeping over several packets.
    const uint32_t batch = std::max(1u, operation_attributes.num_l1_slots / 2);
    log_info(
        tt::LogOp,
        "combine_fabric2d L1: ring 0x{:x} ({} slots x {} B, batch {}) hdr_ring 0x{:x} drain_sink 0x{:x}",
        l1.ring,
        operation_attributes.num_l1_slots,
        token_size_bytes,
        batch,
        l1.pkt_hdr_ring,
        l1.drain_sink);

    // Physical chip id -> mesh coordinate, to turn a cable's far chip into a placement lookup.
    std::map<uint32_t, ttnn::MeshCoordinate> chip_to_coord;
    for (const auto& c : ttnn::MeshCoordinateRange(mesh_shape)) {
        chip_to_coord.emplace(static_cast<uint32_t>(mesh_device->get_device(c)->id()), c);
    }

    PlacementCache placements(mesh_device, axis, operation_attributes.num_links, grid);

    // Every buffer here is interleaved DRAM whose base address is uniform across the mesh, so a producer can
    // address the same buffer on any chip by page index. That is what lets a token carry a final destination
    // address computed on the chip it started from.
    auto* dram_out_buf = tensor_return_value.buffer();
    auto* dram_meta_buf = tensor_args.dispatched_metadata.buffer();
    auto* dram_counts_buf = tensor_args.expert_token_counts.buffer();
    auto* dram_region_buf = tensor_args.expert_region_offsets.buffer();
    auto* dram_expert_offsets_buf = tensor_args.expert_offsets.buffer();
    TT_FATAL(dram_out_buf != nullptr, "combine_fabric2d: output tensor has no device buffer");
    TT_FATAL(dram_meta_buf != nullptr, "combine_fabric2d: dispatched_metadata has no device buffer");
    TT_FATAL(dram_counts_buf != nullptr, "combine_fabric2d: expert_token_counts has no device buffer");
    TT_FATAL(dram_region_buf != nullptr, "combine_fabric2d: expert_region_offsets has no device buffer");
    TT_FATAL(dram_expert_offsets_buf != nullptr, "combine_fabric2d: expert_offsets has no device buffer");
    TT_FATAL(
        dram_out_buf->aligned_page_size() == token_size_bytes,
        "combine_fabric2d: output page size {} must equal the token page size {} — the op moves whole tokens "
        "between the two",
        dram_out_buf->aligned_page_size(),
        token_size_bytes);
    // The output holds one page per (token, top-k slot). Checked against the real buffer because only it
    // knows the per-device page count.
    const uint32_t out_pages = static_cast<uint32_t>(dram_out_buf->num_pages());
    const uint32_t want_out_pages = operation_attributes.seq_len_per_chip * operation_attributes.num_experts_per_tok;
    TT_FATAL(
        out_pages >= want_out_pages,
        "combine_fabric2d: output holds {} pages but seq_len_per_chip x num_experts_per_tok = {} are needed",
        out_pages,
        want_out_pages);
    TT_FATAL(
        dram_meta_buf->num_pages() == dram_in_buf->num_pages(),
        "combine_fabric2d: metadata has {} pages but the dispatched buffer has {}; they index the same flat "
        "buffer",
        dram_meta_buf->num_pages(),
        dram_in_buf->num_pages());

    const uint32_t num_routed_experts = static_cast<uint32_t>(tensor_args.expert_token_counts.logical_shape()[-1]);
    const uint32_t num_dispatch_groups =
        num_routed_experts / (operation_attributes.experts_per_chip * operation_attributes.dispatch_group_size);
    TT_FATAL(num_dispatch_groups >= 1, "combine_fabric2d: derived num_dispatch_groups is 0");

    tt::tt_metal::WorkloadDescriptor workload_descriptor;
    workload_descriptor.semaphores.push_back(ring_filled_sem);
    workload_descriptor.semaphores.push_back(ring_freed_sem);
    workload_descriptor.semaphores.push_back(fwd_arrived_sem);

    // ---- Op-internal forwarding buffer. Never initialised and never read back: it is pure staging for
    // tokens passing through a chip on their way somewhere else (phase 9). One page per token, and the page
    // is token + tail so a single fabric write lands both. Fused into ONE page rather than split across a
    // payload and a metadata region precisely because nothing outside the op reads it — so the "one page =
    // one token" property that the caller's regions must keep does not apply here, and we save a DRAM
    // read and a DRAM write per forwarded token.
    //
    // Allocated with create_device_tensor (no host data, no upload) and parked on
    // WorkloadDescriptor::buffers wrapped in a shared_ptr<Tensor>: holding only a shared_ptr<MeshBuffer>
    // would let DeviceStorage::deallocate free the memory when the local Tensor dies at the end of this
    // function (workload_descriptor.hpp:19-36). Being a mesh tensor, its device-local address is uniform
    // across the mesh by construction, which is what lets a producer address the NEXT chip's buffer.
    const uint32_t fwd_page_bytes = token_size_bytes + CMBF2D_SLOT_TAIL_BYTES;
    TT_FATAL(
        fwd_page_bytes % sizeof(uint32_t) == 0,
        "combine_fabric2d: forwarding page {} B must be a multiple of 4",
        fwd_page_bytes);
    const uint32_t fwd_chunks = fwd_total_chunks(extent, operation_attributes.num_links);
    const uint32_t pages_per_chunk = fwd_pages_per_chunk(
        operation_attributes.seq_len_per_chip,
        operation_attributes.num_experts_per_tok,
        num_dispatch_groups,
        operation_attributes.dispatch_group_size,
        operation_attributes.num_links);
    const uint32_t fwd_pages = fwd_chunks * pages_per_chunk;
    const ttnn::TensorSpec fwd_spec(
        ttnn::Shape({fwd_pages, fwd_page_bytes / static_cast<uint32_t>(sizeof(uint32_t))}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM}));
    // Throws if it does not fit DRAM, which IS the "verify it fits" check this stage asks for.
    auto fwd_owner = std::make_shared<ttnn::Tensor>(tt::tt_metal::create_device_tensor(fwd_spec, mesh_device));
    auto* dram_fwd_buf = fwd_owner->buffer();
    TT_FATAL(dram_fwd_buf != nullptr, "combine_fabric2d: forwarding buffer has no device buffer");
    TT_FATAL(
        dram_fwd_buf->aligned_page_size() == fwd_page_bytes,
        "combine_fabric2d: forwarding page size is {} B after alignment but the op addresses it as {} B. "
        "The token page + {} must be a multiple of the DRAM alignment.",
        dram_fwd_buf->aligned_page_size(),
        fwd_page_bytes,
        CMBF2D_SLOT_TAIL_BYTES);
    workload_descriptor.buffers.push_back({fwd_owner, dram_fwd_buf});
    log_info(
        tt::LogOp,
        "combine_fabric2d forwarding buffer: {} chunks ({} per quarter x {} planes x 2 directions) x {} pages "
        "(4x the mean chunk + sentinel) x {} B = {:.1f} MB per device at 0x{:x}",
        fwd_chunks,
        fwd_chunks_per_quarter(extent),
        operation_attributes.num_links,
        pages_per_chunk,
        fwd_page_bytes,
        static_cast<double>(fwd_pages) * fwd_page_bytes / 1e6,
        dram_fwd_buf->address());

    for (const auto& coord : tensor_coords.coords()) {
        auto desc = build_program_for_coord(
            operation_attributes,
            coord,
            placements,
            chip_to_coord,
            l1,
            batch,
            ring_filled_addr,
            ring_freed_addr,
            fwd_arrived_addr,
            token_size_bytes,
            num_routed_experts,
            pages_per_chunk,
            dram_out_buf,
            dram_in_buf,
            dram_fwd_buf,
            dram_meta_buf,
            dram_counts_buf,
            dram_region_buf,
            dram_expert_offsets_buf);
        workload_descriptor.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }
    return workload_descriptor;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
