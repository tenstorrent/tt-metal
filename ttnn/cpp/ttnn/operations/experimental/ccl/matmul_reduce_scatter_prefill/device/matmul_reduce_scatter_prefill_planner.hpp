// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

// The host planner of matmul_reduce_scatter_prefill: the transport schedule (build_schedule, plan_transport,
// entry_finish_plan), the matmul grid and blocking (choose_grid, wave_plan, footprint, choose_k_block,
// super_block_plan) and the output-chunk search (chunk_candidates, plan_chunk_geometry, chunk_cost_us). Integers in,
// integers out: no device, no tensors. Fixed policy: lag a relay's finish only when its arrival is produced late,
// model-based chunk choice, staging sized exactly, one entry per block, one arrival increment per wave, mirrored half
// orders, resident stream depths {3, 2}, 2 zone sets, kb <= 16 (32 when chunked), 4 chunks per port batch, the
// folded 1D grid for a one-tile-row block whenever it has more cores than the 2D grid.
namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_prefill::detail {

// The schedule kind: "line", "ring_unbalanced" (odd G >= 3), "ring_balanced" (even G >= 4).
enum class ScheduleKind : uint8_t { Line, RingUnbalanced, RingBalanced };
// A block half's direction at a chip; Fin is the chip's own block.
enum class Direction : uint8_t { Fwd, Bwd, Fin };
// Ports strip (logical row 0 / last logical column) and the grid orientation.
enum class Strip : uint8_t { Row, Col };
enum class Orient : uint8_t { MOnX, MOnY };
// The matmul flow. Grid2D: a grid_m x grid_n rectangle, A multicast along its m-lines and W along its n-lines.
// Folded1D (a block of at most kFoldedMaxM tile rows): one m-line of grid_n cores folded row-major into a
// rows x cols rectangle; every core holds the whole block's rows, A is multicast to the whole rectangle and each
// core reads its own W columns.
enum class Flow : uint8_t { Grid2D, Folded1D };
// How the matmul cores split into halves ("k": row-major core index, "n"/"m": along that grid axis).
enum class HalfSplit : uint8_t { K, N, M };

struct Role {
    Direction dir = Direction::Fin;
    bool is_start = false;
    uint32_t distance = 0;
};

// One chip position's schedule (plan_transport's chips[p]); blocks are virtual (b * waves + w).
struct ChipSchedule {
    std::vector<std::array<Role, 2>> roles;                                  // [block][half]
    std::array<std::vector<uint32_t>, 2> entries;                            // [half]: entry order
    std::array<std::array<std::vector<uint32_t>, 2>, 2> arrivals;            // [half][Fwd/Bwd]
    std::array<std::array<std::map<uint32_t, uint32_t>, 2>, 2> thresholds;   // [half][dir]: block -> threshold
    std::array<std::array<std::map<uint32_t, uint32_t>, 2>, 2> produced_at;  // [half][dir]: block -> entry index
};

struct Transport {
    std::vector<ChipSchedule> chips;  // per chip position p in the group
    std::array<uint32_t, 2> slots{};  // zone slots per direction (Fwd, Bwd)
};

struct EntryFinish {
    uint32_t mask = 0;  // bit 0 fwd / bit 1 bwd arrival added in
    uint32_t lag = 0;
};

struct GridChoice {
    Strip strip = Strip::Row;
    Orient orient = Orient::MOnX;
    uint32_t grid_m = 0;
    uint32_t grid_n = 0;
    uint32_t core_m = 0;
    uint32_t core_n = 0;
    std::array<uint32_t, 2> subblock{};  // (sb_h, sb_w)
    Flow flow = Flow::Grid2D;
    uint32_t fold_rows = 0;  // Folded1D: logical core gn sits at (gn % fold_cols, gn / fold_cols) of the rectangle
    uint32_t fold_cols = 0;
};

struct WaveDesc {
    uint32_t row_off = 0;
    uint32_t col_off = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
};

struct WavePlan {
    uint32_t n = 1;
    uint32_t win_m = 0;
    uint32_t win_n = 0;
    uint32_t P = 0;  // slot tiles (the largest wave)
    std::vector<WaveDesc> descs;
};

// Everything plan_chunk_geometry returns.
struct Plan {
    uint32_t n_m = 1;
    uint32_t n_n = 1;
    uint32_t chunk_m = 0;
    uint32_t chunk_n = 0;
    uint32_t zone_sets = 1;
    GridChoice grid;
    bool resident = false;
    uint32_t kb = 0;
    std::array<uint32_t, 2> depths{};  // (A, W) window depths in K-blocks
    uint32_t super_block = 1;
    uint32_t depth_hold = 1;
    HalfSplit half_split = HalfSplit::K;
    Transport sched;
    WavePlan waves;
    uint32_t P = 0;
    uint64_t footprint = 0;  // bytes per matmul core
};

// What the entry hands the planner, the device values already read.
struct PlanInputs {
    uint32_t G = 0;
    ScheduleKind kind = ScheduleKind::Line;
    uint32_t L = 0;  // links per hop
    uint32_t Mt = 0;
    uint32_t Kt = 0;
    uint32_t Nt = 0;
    uint32_t block_m = 0;
    uint32_t block_n = 0;
    uint32_t gx = 0;  // compute_with_storage_grid_size
    uint32_t gy = 0;
    uint32_t tA = 0;  // A / W buffer aligned page sizes
    uint32_t tW = 0;
    int32_t scatter_dim = -2;  // canonical: -1 or -2
    uint64_t budget = 0;       // L1 bank size - L1_HEADROOM
};

// Planner constants the factory needs too.
constexpr uint32_t kDepthHandoff = 2;        // finished slots per matmul core
constexpr uint32_t kMailboxStride = 16;      // bytes per matmul core in a port's mailbox
constexpr uint32_t kPortGroup = 4;           // chunks per port batch (cb::PORT_CHUNKS holds 2 batches)
constexpr uint32_t kPrefetchSteps = 1;       // K-block windows a sender reads ahead
constexpr uint32_t kL1Headroom = 64 * 1024;  // bytes kept free per core beyond the footprint

// The entry's plan: the cheapest (chunk_cost_us) of the unchunked plan and the chunk_candidates that fit L1; scatter on
// M takes the unchunked plan whenever it fits. TT_THROWs when no plan fits L1 or the geometry cannot be laid out. A
// solved plan is remembered per input, so a repeated call looks it up instead of re-planning. The memo is never erased,
// so the reference stays valid for the process: callers keep the inputs and look the plan up, never copy it.
const Plan& solve_plan(const PlanInputs& in);

// The schedule pieces the factory walks.
Transport plan_transport(uint32_t G, ScheduleKind kind, bool mirror, uint32_t waves);
std::vector<EntryFinish> entry_finish_plan(const ChipSchedule& sc, uint32_t half);
// _subblock: the largest h x w (h | rows, w | cols, h * w <= 8), ties to the wider.
std::array<uint32_t, 2> choose_subblock(uint32_t rows, uint32_t cols);
// staging tiles per matmul core: (kDepthHandoff + zone_sets * (slots fwd + slots bwd)) * P.
uint32_t staging_tiles(const Plan& plan);

}  // namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_prefill::detail
