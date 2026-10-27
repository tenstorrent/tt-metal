// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include "ttnn/tensor/tensor_spec.hpp"
#include "all_gather_device_operation_types.hpp"

namespace tt::tt_metal::distributed {
class MeshDevice;
}

// Host plan for all_gather: facts, geometry, legality per algorithm, the choice model, the schedule and core
// placement. Only make_facts queries the machine; everything else is a pure function of its arguments, so host tests
// inject facts and all_gather() and the factories compute the same plan.
//
// Data flow: AllGatherParams + tensor specs -> PlanIn (make_plan_in) -> Plan (make_plan, or expand for a choice
// already made) -> WorkerSched per worker (schedule).
namespace ttnn::operations::ccl::all_gather_plan {

// ---- structural constants ----

// CB slots per worker: one filling while the other drains. Reads in flight per slot are the throughput lever.
inline constexpr uint32_t kCbSlots = 2;
// Packets per slot at most: bounds the L1 search (slot_bytes is still checked against L1).
inline constexpr uint32_t kMaxPacketsPerSlot = 32;

// The native algorithms, in index order.
inline constexpr std::array kNativeAlgos = {Algo::Unicast, Algo::Multicast};

// Index of a native algorithm in the per-algorithm arrays below.
constexpr size_t algo_index(Algo algo) { return static_cast<size_t>(algo); }

// ---- per-arch tuning ----

// Every value the op takes from measurement rather than from a hardware fact, for one arch at the fabric payload and
// link count it was measured at. Arrays indexed [algo][topology] are [unicast, multicast][line, ring]. Any other arch,
// payload or link count runs on the Blackhole entry and is logged once as uncalibrated; calibrating an arch means
// adding its entry to kTunings in all_gather_plan.cpp.
struct Tuning {
    tt::ARCH arch = tt::ARCH::Invalid;
    uint32_t payload = 0;  // fabric max payload the entry was measured at, bytes
    uint32_t links = 0;    // links per axis it was measured at

    // Choice model T = fixed_us + max(B / bw_gbps, unicast bank time), B = bytes on the busiest link-direction.
    std::array<std::array<double, 2>, 2> fixed_us{};  // microseconds
    std::array<std::array<double, 2>, 2> bw_gbps{};   // GB/s of payload on the busiest link-direction
    double bank_gbps = 0;                             // GB/s one output bank sustains, reads and writes together

    // Unicast workers per link-direction: a ring always 3 if ring_n3, a line 3 from n3_min_packets packets on the
    // busiest link-direction, else 2.
    bool ring_n3 = false;
    uint64_t n3_min_packets = 0;

    // Slot reads: about slot_read_bytes of payload per read barrier (packets_per_slot = ceil(slot_read_bytes / packet
    // cap)).
    uint32_t slot_read_bytes = 0;
    // Unicast data_valid cadence: about signals_per_iter signals per worker iteration, clamped to
    // [cadence_min_packets, cadence_max_packets] packets.
    uint32_t signals_per_iter = 0, cadence_min_packets = 0, cadence_max_packets = 0;

    std::array<uint32_t, kMaxWorkers> mux_buffers{};  // Mux V2 buffers per channel, [n - 1]
    tt::tt_metal::NOC mux_forwarder_noc = tt::tt_metal::NOC::NOC_0;
    // The writer's NOC; the reader takes the other one (two RISCs of a core use different NOCs).
    std::array<std::array<tt::tt_metal::NOC, 2>, 2> writer_noc{};
};

// The entry for `arch`, or Blackhole's when it has none.
const Tuning& tuning(tt::ARCH arch);

// Mux V2 buffers per channel for n channels (1 <= n <= kMaxWorkers).
uint32_t mux_buffers(tt::ARCH arch, uint32_t n);

// ---- facts ----

// The op's only HAL / fabric / allocator / sub-device queries. The Mux V2 map ends are memoized, so a program-cache hit
// constructs no FabricMuxV2Config.
Facts make_facts(
    const tt::tt_metal::distributed::MeshDevice& mesh_device,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<CoreRangeSet>& sub_core_grid);

// ---- plan input ----

// Logical neighbor directions, numbered as eth_chan_directions: E/W run along axis 1 (E forward), N/S along axis 0
// (S forward).
enum class Dir : uint32_t { East, West, North, South };

// A unicast link-direction: forward sends to the next device on the axis.
inline constexpr uint32_t kFwd = 0;
inline constexpr uint32_t kBwd = 1;

// The input's and output's pages, from their specs; sizes in bytes.
struct Pages {
    uint32_t in_page_bytes = 0, out_page_bytes = 0;                  // page content
    uint32_t in_aligned_page_bytes = 0, out_aligned_page_bytes = 0;  // bank pitch
    uint32_t in_read_align = 0, out_read_align = 0;
    uint32_t in_pages = 0;
    uint32_t pages_per_stripe_row = 0;  // input pages this device contributes to one output row
    uint32_t out_page_stride = 1;       // the output's contiguous_page_stride()
    // Output pages contiguous (out_page_stride apart) from the first page of a stripe row; unbounded for interleaved.
    uint32_t out_run_pages = 1;
    // The fewest output banks (DRAM banks, L1 banks or shard cores) that hold one device's stripe; 0: unknown.
    uint32_t stripe_banks = 0;
    bool operator==(const Pages&) const = default;
};

struct PlanIn {
    Facts facts;
    Pages pages;
    std::string shape_reason;        // the shape rule that rules out every native algorithm (empty: none)
    uint32_t N = 0;                  // devices in the collective
    uint32_t links = 0;              // links available to the collective
    bool ring = false;               // the 1D axis wraps; false for N == 2 (both directions reach one neighbor)
    bool true_2d = false;            // 2D fabric with both axes active
    bool bent_axis = false;          // a 2D fabric view bends an active axis
    bool neighbor_exchange = false;  // the fabric forwards one hop only
    std::array<uint32_t, 2> axis_devices{1, 1};
    std::array<bool, 2> axis_ring{};  // per axis, as `ring`
};

// The composite rule that needs only the input (gather-dim padding). all_gather() checks it before building the output
// spec, which can TT_FATAL on a legacy config that cannot hold a padded gathered shape.
std::optional<std::string> gather_dim_padding_reason(const tt::tt_metal::TensorSpec& in, int32_t dim_from_end);

// The output spec is the persistent output's when there is one.
PlanIn make_plan_in(
    const AllGatherParams& params, const tt::tt_metal::TensorSpec& in, const tt::tt_metal::TensorSpec& out);

// ---- geometry ----

// The data cut into units: counts are in units of unit_bytes (all_gather_walk.hpp names the same quantities).
struct Geometry {
    uint32_t unit_bytes = 0;        // the cut quantum
    uint32_t in_page_units = 0;     // input page footprint
    uint32_t out_page_units = 0;    // output page footprint
    uint32_t stripe_units = 0;      // units this device contributes to one output row
    uint32_t rows = 0;              // output rows
    uint32_t element_units = 0;     // units per walk element: out_page_units when a stripe row is whole pages, else 1
    uint32_t elements = 0;          // walk elements of one stripe
    uint32_t max_packet_bytes = 0;  // fabric packet payload limit, a multiple of unit_bytes
    bool dst_cross = false;         // runs may continue across output pages
    bool src_cross = false;         // runs may continue across input pages
    // The packet builder's payload cap and the slot's payload stride: at most 4 of the longest segments the walk makes.
    uint32_t packet_cap = 0;
    bool operator==(const Geometry&) const = default;
};

// nullopt: these pages cannot be cut into aligned units. unicast: the relay also reads its own output, so the unit
// covers the output's read alignment too.
std::optional<Geometry> make_geometry(const PlanIn& in, bool unicast);

// ---- legality ----

// Why a native algorithm cannot run `in` at the least resources (1 worker, 1 link); empty when it can. A legal input
// never fatals later; the per-n mux-map and placement checks run again for the n and links actually used.
std::string verdict(const PlanIn& in, Algo algo);

// ---- slices and placement ----

struct Slice {
    uint32_t e0 = 0, ne = 0;
    bool operator==(const Slice&) const = default;
};

// count contiguous ranges covering [0, elements), sizes within 1 of each other; empty when count > elements.
std::vector<Slice> make_slices(uint32_t elements, uint32_t count);

// One unicast (link, direction): its mux and its workers. Worker w connects to mux channel w, and the mux has
// workers.size() channels, so channels and connections come from this one field.
struct Group {
    uint32_t link = 0, dir = 0;
    CoreCoord mux;
    std::vector<CoreCoord> workers;
    bool operator==(const Group&) const = default;
};

// Groups ordered (link 0 fwd, link 0 bwd, link 1 fwd, ...), each a mux plus n workers on one NOC row (workers feed a
// mux on their own row fastest), groups on distinct rows first, then rows doubled up. nullopt: the cores do not fit.
// Its own placement rather than choose_worker_cores, which has no notion of a group sharing a row.
std::optional<std::vector<Group>> place_unicast(const CoreRangeSet& cores, uint32_t links, uint32_t n);

// One core per link, in choose_worker_cores order (ranges in order, row-major within each).
std::optional<std::vector<CoreCoord>> place_multicast(const CoreRangeSet& cores, uint32_t links);

// ---- the plan ----

struct Plan {
    Choice choice;
    Geometry geo;  // the chosen algorithm's (empty for composite)
    uint32_t N = 0;
    bool ring = false;
    tt::ARCH arch = tt::ARCH::Invalid;   // selects the Tuning entry
    std::string composite_reason;        // why composite (empty otherwise)
    uint32_t slot_bytes = 0;             // CB page = one slot (ag::slot_offsets); kCbSlots of them per worker
    std::vector<Slice> slices;           // unicast: links * workers, (link, w) -> link * workers + w; multicast: links
    std::vector<Group> groups;           // unicast
    std::vector<CoreCoord> mcast_cores;  // multicast, one per link
    bool operator==(const Plan&) const = default;
};

// Legality -> choice (model) -> shrink (workers 3 -> 2 -> 1, then links, then the other algorithm, then composite,
// each warned once) -> expand. An input composite cannot run either is a TT_FATAL with the reason (fewer than 2
// devices, or a neighbor-exchange fabric past 2 devices).
Plan make_plan(const PlanIn& in);

// The plan for a native choice already made (no decisions, no logging); the factories call this on a program-cache
// miss, and host tests to name any algorithm, worker count or link count. TT_FATAL if the algorithm is illegal for
// `in`, or the choice exceeds its links, cores, mux L1 or slot L1.
Plan expand(const PlanIn& in, const Choice& choice);
Plan expand(const AllGatherParams& params, const tt::tt_metal::TensorSpec& in, const tt::tt_metal::TensorSpec& out);

// Unicast: the device has a neighbor in this direction (a mux exists for it).
bool dir_active(const Plan& plan, uint32_t dev, uint32_t dir);

// One worker of one device. Counts are in units (unit_bytes) unless named otherwise. Iteration i sends stripe
// (stripe0 + i * stripe_step) % N, walking positions [0, ne) of the slice, except the last iteration, which walks
// [last_skip, last_skip + last_take). An idle direction has iters == 0 but still opens, inits and closes.
struct WorkerSched {
    uint32_t e0 = 0, ne = 0;  // slice: walk elements [e0, e0 + ne)
    uint32_t iters = 0;
    uint32_t stripe0 = 0, stripe_step = 0;
    uint32_t last_skip = 0, last_take = 0;  // elements
    uint32_t iter_units = 0;                // units of one full iteration (ne * element_units)
    // Unicast: signal data_valid every this many slots and at each iteration end.
    uint32_t cadence_slots = 0;
    // Init credits the writer waits for and consumes before its first send.
    uint32_t init_wait = 0;
    // What data_sem must reach before the reader exits: unicast data_valid units from upstream; multicast done credits.
    uint32_t end_wait = 0;
    // Iteration 0 also writes this device's stripe into its own output.
    bool local_copy = false;
    // Unicast: one init credit to the neighbor's opposite-direction worker w; multicast: one to every other device.
    bool init_send = false;
    bool operator==(const WorkerSched&) const = default;
};

// dir and w are 0 for multicast.
WorkerSched schedule(const Plan& plan, uint32_t dev, uint32_t link, uint32_t dir, uint32_t w);

// ---- multicast routes ----

// One multicast connection: dir is the logical neighbor; range holds hop counts at physical slots,
// range = e | w << 8 | n << 16 | s << 24.
struct McastConn {
    Dir dir;
    uint32_t range;
};

// mesh_shape: devices per axis; axis selects the active axis of a 1D collective (nullopt: every axis with more than
// one device). axis_ring: the axis wraps (false for 2 devices). phys_dir maps each logical direction to its physical
// eth_chan_directions slot; identity when the axes run E/W and N/S. Together the connections reach every other device
// of the collective exactly once.
std::vector<McastConn> mcast_routes(
    uint32_t N,
    std::array<uint32_t, 2> mesh_shape,
    std::array<uint32_t, 2> my_coord,
    std::optional<uint32_t> axis,
    std::array<bool, 2> axis_ring,
    std::array<uint32_t, 4> phys_dir = {0, 1, 2, 3});

// ---- choice model (make_plan's internals, exposed for host tests) ----

namespace detail {

// Bytes and packets on the busiest link-direction: B = stripe bytes * (N - 1) / (ring ? 2 : 1) / links,
// ceil(B / packet_cap).
uint64_t busiest_link_bytes(const Geometry& geo, uint32_t N, bool ring, uint32_t links);
uint64_t busiest_link_packets(const Geometry& geo, uint32_t N, bool ring, uint32_t links);

// Modelled time in microseconds for a native algorithm on `in` at in.links: fixed + B / BW, and for unicast at least
// fixed + the time the stripe's banks (in.pages.stripe_banks) need for its writes and relay reads.
double model_us(const PlanIn& in, Algo algo, const Geometry& geo);

// Unicast workers per link-direction the model wants (before placement shrinks it).
uint32_t model_workers(const Tuning& t, bool ring, uint64_t busiest_packets);

}  // namespace detail

}  // namespace ttnn::operations::ccl::all_gather_plan
