// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <vector>

// The host planner of matmul_reduce_scatter (the Python's choose_grid / choose_k_block / super_block_plan /
// chunk search and build_schedule / plan_transport / entry_finish_plan), integers in, integers out. The Python's
// policy knobs that only ever took one value are collapsed: one wave per block (so P = core_m * core_n and a wave
// is the whole per-core block), an arrival increment per unit (inc_unit is the identity), the "auto" lag policy,
// the "model" auto-chunk policy, the "exact" staging policy (no staging floor), ports preferred in the row strip.
namespace ttnn::operations::matmul_reduce_scatter {

enum class ScheduleKind : uint8_t { LINE, RING_UNBALANCED, RING_BALANCED };
// How matmul cores split into the two halves: by core index ("k"), by gn ("n") or by gm ("m").
enum class HalfSplit : uint8_t { K, N, M };

// Everything the planner reads, resolved by the entry from the tensors and the device.
struct PlanInputs {
    uint32_t group_size = 0;  // G
    ScheduleKind kind = ScheduleKind::LINE;
    uint32_t num_links = 0;  // L, resolved
    uint32_t Mt = 0;
    uint32_t Kt = 0;
    uint32_t Nt = 0;
    uint32_t block_m = 0;
    uint32_t block_n = 0;
    uint32_t grid_x = 0;  // compute_with_storage_grid_size
    uint32_t grid_y = 0;
    uint32_t a_tile_bytes = 0;  // input buffer's aligned page size
    uint32_t w_tile_bytes = 0;
    int32_t scatter_dim = -2;  // canonical: -1 or -2
    uint64_t l1_budget = 0;    // l1 bank size - L1_HEADROOM
};

struct GridPlan {
    bool row_strip = true;  // ports in logical row 0 (else the last column)
    bool m_on_x = true;     // orientation: m along x
    uint32_t grid_m = 0;
    uint32_t grid_n = 0;
    uint32_t core_m = 0;
    uint32_t core_n = 0;
    uint32_t sb_h = 0;  // subblock of the (single) wave
    uint32_t sb_w = 0;
};

// The solved plan: what fixes every circular-buffer size and compile-time arg. An attribute of the operation,
// so it is in the program key.
struct Plan {
    uint32_t n_m = 1;
    uint32_t n_n = 1;
    uint32_t chunk_m = 0;
    uint32_t chunk_n = 0;
    uint32_t zone_sets = 1;
    GridPlan grid;
    bool resident = false;
    uint32_t kb = 0;
    uint32_t depth_a = 0;  // window depths in K-blocks (unused for the resident operand)
    uint32_t depth_w = 0;
    uint32_t super_block = 1;
    uint32_t depth_hold = 1;
    HalfSplit half_split = HalfSplit::K;
    uint32_t P = 0;          // slot tiles (core_m * core_n)
    uint32_t slots_fwd = 1;  // zone slots per direction, per chunk
    uint32_t slots_bwd = 1;
    uint64_t footprint = 0;  // bytes per matmul core
};

// Finished slots per core (a port gathers entry e while compute finishes e + 1).
inline constexpr uint32_t kDepthHandoff = 2;

// Staging tiles per core: (kDepthHandoff + zone_sets * (slots_fwd + slots_bwd)) * P.
uint32_t staging_tiles(const Plan& plan);

// Solves the plan: unchunked if it fits, else the auto chunk search (memoized). Throws std::invalid_argument
// where the Python raised ValueError and NotImplementedRefusal where it raised NotImplementedError.
Plan solve_plan(const PlanInputs& in);

// ---- transport schedule -----------------------------------------------------------------------------------------

enum class RoleKind : uint8_t { FWD, BWD, FIN };
struct Role {
    RoleKind kind = RoleKind::FIN;
    bool is_start = false;
    uint32_t distance = 0;
};

// One chip position's schedule. Directions index 0 = fwd, 1 = bwd; halves 0 / 1.
struct ChipSchedule {
    std::vector<std::array<Role, 2>> roles;                                  // [block][half]
    std::array<std::vector<uint32_t>, 2> entries;                            // [half]: block order
    std::array<std::array<std::vector<uint32_t>, 2>, 2> arrivals;            // [half][dir]
    std::array<std::array<std::map<uint32_t, uint32_t>, 2>, 2> thresholds;   // [half][dir]: block -> threshold
    std::array<std::array<std::map<uint32_t, uint32_t>, 2>, 2> produced_at;  // [half][dir]: block -> entry index
};

struct TransportPlan {
    std::vector<ChipSchedule> chips;      // per chip position p
    std::array<uint32_t, 2> slots{1, 1};  // zone slots per direction
};

TransportPlan plan_transport(uint32_t group_size, ScheduleKind kind, bool mirror);

struct EntryFinish {
    uint32_t mask = 0;  // bit 0 fwd, bit 1 bwd arrival added in
    bool lag = false;
};
std::vector<EntryFinish> entry_finish_plan(const ChipSchedule& schedule, uint32_t half);

// The Python's _subblock: largest h x w <= cap dividing (rows, cols), ties to the wider.
std::array<uint32_t, 2> choose_subblock(uint32_t rows, uint32_t cols);

}  // namespace ttnn::operations::matmul_reduce_scatter
