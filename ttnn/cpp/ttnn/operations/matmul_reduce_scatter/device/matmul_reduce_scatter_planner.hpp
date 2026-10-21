// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <compare>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "ttnn/cpp/ttnn/kernel_lib/mcast/host/mcast_host.hpp"

// The host planner of matmul_reduce_scatter (the Python planner, minus its perf-experiment knobs, which collapse to
// their defaults): the transport schedule, the grid, the regime / K-block, the super-block and the output chunks.
// Pure integer arithmetic: no device, no tensors.
namespace ttnn::operations::matmul_reduce_scatter {

// ---- knobs ----------------------------------------------------------------------------------------------------
constexpr uint32_t kDepthIn0 = 3;  // stream regime: A K-block windows
constexpr uint32_t kDepthIn1 = 3;  // stream regime: W K-block windows
// resident regime: depths the streamed operand's window may take, preferred first
constexpr std::array<uint32_t, 2> kResidentStreamDepths = {3, 2};
constexpr uint32_t kSuperBlock = 2;       // K-windows per DEST-resident pack
constexpr uint32_t kSuperBlockExtra = 2;  // extra streamed windows (the loader's look-ahead)
constexpr uint32_t kDepthHold = 2;        // fp32 hold slots when some entry lags (else 1)
constexpr uint32_t kDepthHandoff = 2;     // finished slots per core
constexpr uint32_t kZoneSets = 2;         // chunks' arrivals the zones hold at once (chunked plans)
constexpr uint32_t kMaxAutoChunks = 256;
constexpr uint32_t kKbMaxChunked = 32;
constexpr uint32_t kKbMinResident = 8;
constexpr uint32_t kKbMax = 16;
constexpr uint32_t kL1HeadroomBytes = 64 * 1024;  // bytes kept free per core beyond the footprint
constexpr uint32_t kDestTilesFp32FullSync = 8;    // subblock area cap
constexpr uint32_t kPortGroup = 4;                // chunks per port batch (cb_port_chunks holds 2 batches)
constexpr uint32_t kMailboxStride = 16;           // bytes per matmul core in a port's mailbox
constexpr uint32_t kPrefetchSteps = 1;
// chunk_cost_us() constants (least-squares fit, see the Python planner)
constexpr double kCostFpuEff = 0.93;
constexpr double kCostStepUs = 1.4;
constexpr double kCostStepStreamUs = 2.1;
constexpr double kCostHandoffUs = 0.29;
constexpr double kCostDramGbps = 200;
constexpr double kCostFloorUs = 8.0;
constexpr double kClockMhz = 1350;

// ---- transport schedule (build_schedule / plan_transport / entry_finish_plan / hold_depth) --------------------
enum class ScheduleKind : uint32_t { Line, RingUnbalanced, RingBalanced };

enum class RoleKind : uint32_t { Fwd = 0, Bwd = 1, Fin = 2 };  // Fwd / Bwd index the per-direction arrays below

struct Role {
    RoleKind kind = RoleKind::Fin;
    bool is_start = false;
    uint32_t distance = 0;
};

// One chip position's schedule. Block ids are real blocks (the port runs one wave per block).
struct ChipSchedule {
    std::vector<std::array<Role, 2>> roles;        // roles[b][h]
    std::array<std::vector<uint32_t>, 2> entries;  // entries[h]: the half's block order
    // arrivals[h][d] (d: 0 fwd, 1 bwd): blocks of half h this chip receives from direction d, in entry order
    std::array<std::array<std::vector<uint32_t>, 2>, 2> arrivals;
    // thresholds[h][d][b]: position + 1 of b in the upstream's increment order; produced_at[h][d][b]: the
    // upstream's entry index producing it
    std::array<std::array<std::map<uint32_t, uint32_t>, 2>, 2> thresholds;
    std::array<std::array<std::map<uint32_t, uint32_t>, 2>, 2> produced_at;
};

struct TransportPlan {
    std::vector<ChipSchedule> chips;  // per position in the group
    std::array<uint32_t, 2> slots{};  // zone slots per direction (max over chips and halves, >= 1)
};

// TT_FATALs the Python asserts.
TransportPlan plan_transport(uint32_t group_size, ScheduleKind kind, bool mirror);

struct EntryFinish {
    uint32_t mask = 0;  // bit 0 fwd / bit 1 bwd arrival added in
    bool lag = false;   // finish behind the next entry's matmul
};
std::vector<EntryFinish> entry_finish_plan(const ChipSchedule& schedule, uint32_t half);
uint32_t hold_depth(const TransportPlan& transport);

// ---- grid / regime / chunks ------------------------------------------------------------------------------------
struct Subblock {
    uint32_t h = 1;
    uint32_t w = 1;
};
Subblock choose_subblock(uint32_t core_m, uint32_t core_n);

struct GridPlan {
    bool row_strip = true;  // ports in logical row 0 (matmul grid from y = 1), else in the last column
    bool m_on_x = true;     // m along logical x
    uint32_t grid_m = 0;
    uint32_t grid_n = 0;
    uint32_t core_m = 0;
    uint32_t core_n = 0;
    Subblock subblock;
};

enum class HalfSplit : uint32_t { K, N, M };  // row-major core index | along gn | along gm

// The solved plan; the program depends on the device only through it. A hit of the program cache reuses it.
struct ChunkPlan {
    uint32_t n_m = 1;
    uint32_t n_n = 1;
    uint32_t chunk_m = 0;
    uint32_t chunk_n = 0;
    uint32_t zone_sets = 1;
    GridPlan grid;
    bool resident = false;
    uint32_t kb = 0;
    uint32_t depth_a = 0;
    uint32_t depth_w = 0;
    uint32_t super_block = 1;
    uint32_t depth_hold = 1;
    HalfSplit half_split = HalfSplit::K;
    uint32_t p = 0;                   // slot tiles (core_m * core_n: one wave)
    std::array<uint32_t, 2> slots{};  // zone slots per direction of the schedule the regime runs
    uint64_t footprint = 0;
};

// Everything the planner reads, as integers (the caller queries the device and tensors).
struct PlanInputs {
    uint32_t group_size = 0;  // G
    ScheduleKind kind = ScheduleKind::Line;
    uint32_t num_links = 0;  // L
    uint32_t mt = 0;
    uint32_t kt = 0;
    uint32_t nt = 0;
    uint32_t block_m = 0;
    uint32_t block_n = 0;
    uint32_t grid_x = 0;  // compute_with_storage_grid_size
    uint32_t grid_y = 0;
    uint32_t a_tile_bytes = 0;  // input buffer aligned page size
    uint32_t w_tile_bytes = 0;
    uint32_t hold_tile_bytes = 0;   // fp32 tile
    uint32_t stage_tile_bytes = 0;  // bf16 tile
    int32_t scatter_dim = -2;       // canonical: -1 or -2
    uint64_t budget = 0;            // L1 bank size - kL1HeadroomBytes

    auto operator<=>(const PlanInputs&) const = default;  // the key of the entry's solved-plan cache
};

// The unchunked plan when it fits, else the auto chunk search (cost model). Throws std::invalid_argument where
// the Python raises ValueError and NotImplementedRefusal where it raises NotImplementedError.
ChunkPlan solve_plan(const PlanInputs& inputs);

// Staging tiles per core: cb_finished slots + every zone ring.
inline uint32_t staging_tiles(const ChunkPlan& plan) {
    return (kDepthHandoff + plan.zone_sets * (plan.slots[0] + plan.slots[1])) * plan.p;
}

// Byte offset of direction dir's (0 fwd, 1 bwd) arrival zone ring in the staging tensor: after the cb_finished slots,
// the bwd ring after the fwd one.
inline uint32_t staging_zone_offset(const ChunkPlan& plan, uint32_t dir, uint32_t tile_bytes) {
    return (kDepthHandoff + (dir == 0 ? 0 : plan.zone_sets * plan.slots[0])) * plan.p * tile_bytes;
}

// ---- multicast senders (plan_mcast_senders) -----------------------------------------------------------------------
enum class SenderPlacement : uint32_t { Diagonal, DiagonalDisjoint };  // resident: disjoint, stream: diagonal

struct McastSenderPlan {
    std::string prefix;  // "a", "a2", "b", "b2"
    ttnn::kernel_lib::host::Mcast1DShape shape;
    std::optional<uint32_t> start;  // nullopt: absent family
    ttnn::kernel_lib::host::Mcast1DSenderPlacement placement;
};
std::vector<McastSenderPlan> plan_mcast_senders(
    SenderPlacement placement,
    uint32_t grid_m,
    uint32_t grid_n,
    bool a_two_parts,
    bool w_two_parts,
    ttnn::kernel_lib::host::Mcast1DShape a_shape,
    ttnn::kernel_lib::host::Mcast1DShape w_shape);

}  // namespace ttnn::operations::matmul_reduce_scatter
