// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <tt-metalium/core_coord.hpp>
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
// Data flow: AllGatherParams + tensor specs -> PlanIn (make_plan_in) -> AgPlan (make_plan, or expand for a choice
// already made) -> WorkerSched per worker (schedule).
namespace ttnn::operations::ccl::all_gather_plan {

// ---- structural constants (decided, not swept) ----

// CB slots per worker. A third slot measured within 3% on BH; reads in flight per slot (k) are the lever instead.
inline constexpr uint32_t kCbSlots = 2;
// Read bytes one slot aims to have in flight: two full packets at P = 15168 (k = 2 cut worker time 24% vs k = 1).
// Packets shorter than that (a smaller payload, or a packet cap from short runs) get more per slot.
inline constexpr uint32_t kSlotReadBytes = 2 * 15168;
// Packets per slot at most. Nothing in the kernels grows with k but the slot (records and local addresses live in it,
// not on a stack), so this only bounds the L1 search; slot_bytes is still checked against L1.
inline constexpr uint32_t kMaxPacketsPerSlot = 32;

// Mux V2 buffers per channel. n = 1 needs 4 (2 costs 12-14% on BH); at n >= 2, 2 is within 3% of 4 and keeps the
// mux's L1 map smallest.
constexpr uint32_t mux_buffers(uint32_t n) { return n == 1 ? 4 : 2; }

// Test-only override: "<algo>[:key=value,...]", algo one of auto|unicast|multicast|composite, keys n (unicast workers
// per link-direction, 1-3), links, k (packets per slot, 1-kMaxPacketsPerSlot; more than fit in L1 is a fatal), rdelay
// and wdelay (spin cycles per slot, 0-65535). all_gather() reads it; nothing else in the op reads the environment.
inline constexpr const char* kForceEnvVar = "TTNN_ALL_GATHER_FORCE";

// Strict: any malformed spec is a TT_FATAL naming the problem.
AgForce parse_force(std::string_view spec);

// ---- facts ----

// The op's only HAL / fabric / allocator / sub-device queries. The Mux V2 map ends are memoized, so a program-cache hit
// constructs no FabricMuxV2Config.
AgFacts make_facts(
    const tt::tt_metal::distributed::MeshDevice& mesh_device,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<CoreRangeSet>& sub_core_grid);

// ---- plan input ----

// Logical neighbor directions, numbered as eth_chan_directions: E/W run along axis 1 (E forward), N/S along axis 0
// (S forward).
enum Dir : uint32_t { kEast = 0, kWest = 1, kNorth = 2, kSouth = 3 };

// A unicast link-direction: forward sends to the next device on the axis.
inline constexpr uint32_t kFwd = 0;
inline constexpr uint32_t kBwd = 1;

// Index of the per-algorithm arrays below.
inline constexpr uint32_t kUc = 0;
inline constexpr uint32_t kMc = 1;

// The input's and output's pages, from their specs. Page content c*, aligned page (bank pitch) a* and read alignment
// r*, in bytes; i input, o output. pages_per_stripe_row: input pages this device contributes to one output row.
// L: the output's contiguous_page_stride(). run_pages: output pages contiguous (stride L) from the first page of a
// stripe row (unbounded for interleaved). stripe_banks: the fewest output banks (DRAM banks, L1 banks or shard cores)
// that hold one device's stripe (0: unknown).
struct Pages {
    uint32_t ci = 0, co = 0, ai = 0, ao = 0, ri = 0, ro = 0;
    uint32_t in_pages = 0, pages_per_stripe_row = 0;
    uint32_t L = 1, run_pages = 1;
    uint32_t stripe_banks = 0;
    bool operator==(const Pages&) const = default;
};

struct PlanIn {
    AgFacts facts;
    AgForce force;
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

// A: the cut quantum in bytes; a unit is A bytes. Ui/Uo: input/output page footprint in units. S: units per
// stripe-row, R: rows. E: units per walk element (Uo when a stripe-row holds whole output pages, else 1). P_A: packet
// payload cap, a multiple of A. dst_cross/src_cross: runs may continue across output/input pages. lambda, h,
// cross_row: the walk shape (ag::walk_shape, shared with the kernel). run_bytes: modelled mean bytes per packet
// segment (a destination run, split evenly when longer than P_A); fill: modelled payload per packet (at most 4
// segments). cap: the packet builder's payload cap and slot payload stride (ag::packet_cap). stripe_banks:
// Pages::stripe_banks.
struct Geometry {
    uint32_t A = 0, Ui = 0, Uo = 0, S = 0, R = 0, E = 0, elements = 0, P_A = 0;
    bool dst_cross = false, src_cross = false;
    uint32_t lambda = 1, h = 1;
    bool cross_row = false;
    uint32_t cap = 0;
    uint32_t run_bytes = 0, fill = 0;
    uint32_t stripe_banks = 0;
    bool operator==(const Geometry&) const = default;
};

// nullopt: these pages cannot be cut into aligned units. unicast: the relay also reads its own output, so A covers the
// output's read alignment too.
std::optional<Geometry> make_geometry(const PlanIn& in, bool unicast);

// ---- legality ----

struct Verdict {
    bool ok = false;
    std::string reason;  // why not, when !ok
    bool operator==(const Verdict&) const = default;
};

// [kUc], [kMc] at the least resources (1 worker, 1 link). A legal input never fatals later; the per-n mux-map and
// placement checks run again for the n and links actually used.
std::array<Verdict, 2> legality(const PlanIn& in);

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

struct AgPlan {
    AgChoice choice;
    Geometry geo;  // the chosen algorithm's (empty for composite)
    uint32_t N = 0;
    bool ring = false;
    std::string composite_reason;        // why composite (empty otherwise)
    uint32_t slot_bytes = 0;             // CB page = one slot (ag::slot_offsets); kCbSlots of them per worker
    uint32_t variants = 1;               // multicast route variants: 2 when an active axis is an even ring
    std::vector<Slice> slices;           // unicast: links * n, (link, w) -> link * n + w; multicast: links
    std::vector<Group> groups;           // unicast
    std::vector<CoreCoord> mcast_cores;  // multicast, one per link
    bool operator==(const AgPlan&) const = default;
};

// Legality -> choice (model, then force) -> shrink (n 3 -> 2 -> 1, then links, then the other algorithm, then
// composite, each warned once) -> expand. A forced choice that is illegal is a TT_FATAL with the reason, and so is an
// input composite cannot run either (fewer than 2 devices, or a neighbor-exchange fabric past 2 devices).
AgPlan make_plan(const PlanIn& in);

// The plan for a native choice already made (no decisions, no logging); the factories call this on a program-cache
// miss. TT_FATAL if the choice does not fit `in`.
AgPlan expand(const PlanIn& in, const AgChoice& choice);
AgPlan expand(const AllGatherParams& params, const tt::tt_metal::TensorSpec& in, const tt::tt_metal::TensorSpec& out);

// Unicast: the device has a neighbor in this direction (a mux exists for it).
bool dir_active(const AgPlan& plan, uint32_t dev, uint32_t dir);

// One worker of one device. Counts are in units (A bytes) unless named otherwise. Iteration i sends stripe
// (stripe0 + i * stripe_step) % N, walking positions [0, ne) of the slice, except the last iteration, which walks
// [last_skip, last_skip + last_take). An idle direction has iters == 0 but still opens, inits and closes.
struct WorkerSched {
    uint32_t e0 = 0, ne = 0;  // slice: walk elements [e0, e0 + ne)
    uint32_t iters = 0;
    uint32_t stripe0 = 0, stripe_step = 0;
    uint32_t last_skip = 0, last_take = 0;  // elements
    uint32_t iter_units = 0;                // units of one full iteration (ne * E)
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
WorkerSched schedule(const AgPlan& plan, uint32_t dev, uint32_t link, uint32_t dir, uint32_t w);

// ---- multicast routes ----

// dir: the logical neighbor (a Dir). Range bytes sit at physical slots: range = e | w << 8 | n << 16 | s << 24.
struct McastConn {
    uint32_t dir;
    uint32_t range_v0, range_v1;
};
struct McastRoutes {
    std::vector<McastConn> conns;
    bool alternates;
};

// mesh_shape: devices per axis; axis selects the active axis of a 1D collective (nullopt: every axis with more than
// one device). axis_ring: the axis wraps (false for 2 devices). phys_dir maps each logical direction to its physical
// eth_chan_directions slot; identity when the axes run E/W and N/S. Each route variant reaches every other device of
// the collective exactly once.
McastRoutes mcast_routes(
    uint32_t N,
    std::array<uint32_t, 2> mesh_shape,
    std::array<uint32_t, 2> my_coord,
    std::optional<uint32_t> axis,
    std::array<bool, 2> axis_ring,
    std::array<uint32_t, 4> phys_dir = {kEast, kWest, kNorth, kSouth});

// ---- choice model (exposed for tests and logs) ----

// Bytes and packets on the busiest link-direction: B = stripe bytes * (N - 1) / (ring ? 2 : 1) / links, ceil(B / fill).
uint64_t busiest_link_bytes(const Geometry& geo, uint32_t N, bool ring, uint32_t links);
uint64_t busiest_link_packets(const Geometry& geo, uint32_t N, bool ring, uint32_t links);

// Modelled time in microseconds for algorithm kUc / kMc: F + B / BW, and for unicast at least F + the time the
// stripe's banks need for its writes and relay reads.
double model_us(uint32_t algo, const Geometry& geo, uint32_t N, bool ring, uint32_t links);

// Unicast workers per link-direction the model wants (before placement shrinks it).
uint32_t model_workers(bool ring, uint64_t busiest_packets);

}  // namespace ttnn::operations::ccl::all_gather_plan
