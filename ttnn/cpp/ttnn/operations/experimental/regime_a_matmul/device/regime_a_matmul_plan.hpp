// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Pure host-side execution planner for the Regime-A DRAM-BW-optimal matmul.
//
// Given (Mt, Kt, Nt) tile dims, a manual (Ns, Pk, Sm, kb, nsb) config, the compute grid, and the
// per-NoC bank-adjacent worker assignments, this produces ONE canonical ExecutionPlan describing:
//   - actual + padded tile dims,
//   - per-core M/K/N ownership ranges (padded) and valid (unpadded) extents,
//   - worker coordinate + bank + NoC assignment,
//   - in0 ring membership + ordering (next/prev),
//   - split-K reduction links (next/prev, is_bottom/is_top),
//   - CB sizes + L1 accounting,
//   - the compile-time + per-core runtime kernel-argument values.
//
// It is deliberately FREE of tt_metal/device dependencies (uses a plain PlanXY coord and injected
// grid/bank data) so it can be unit-tested exhaustively WITHOUT hardware. The device operation maps
// PlanXY <-> tt::tt_metal::CoreCoord and fetches the bank assignments from the device.
//
// Reference: tools/mm_sweep/UNIFIED_EXECUTION_PLAN_SPEC.md (the frozen prototype blueprint) and
// tools/mm_sweep/GOLDEN_PARITY_SUITE.md (the parity oracle).

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace ttnn::operations::experimental::regime_a_matmul::plan {

// ------------------------------------------------------------------------------------------------
// Shared constants (SINGLE SOURCE OF TRUTH)
// ------------------------------------------------------------------------------------------------
// Hardware / layout constants shared by the pure planner (this header), the device auto-selector +
// weight-memory-config helper (regime_a_matmul_config.cpp), and the program factory
// (regime_a_matmul_program_factory.cpp) — the latter two reach these via the `plan::` alias. Keep values
// EXACTLY as-is: tile-byte sizes feed compile-time kernel args and the L1 budget / bank count / core-count
// window drive config feasibility, so any change here changes codegen or picker behaviour.
constexpr uint32_t kTileBytesBf16 = 2048u;          // bf16 tile bytes
constexpr uint32_t kTileBytesFp32 = 4096u;          // fp32 tile bytes
constexpr uint32_t kNumBanks = 8u;                  // Regime-A fixes the in1 DRAM width-shard to 8 banks (== G)
constexpr uint32_t kL1BudgetBytes = 1440u * 1024u;  // BH usable L1 per core
constexpr uint32_t kMinCores = 16u;                 // auto-picker feasibility: core-count window [kMin, kMax]
constexpr uint32_t kMaxCores = 104u;

// ------------------------------------------------------------------------------------------------
// Inputs
// ------------------------------------------------------------------------------------------------

struct PlanXY {
    uint32_t x{};
    uint32_t y{};
    bool operator==(const PlanXY& o) const { return x == o.x && y == o.y; }
    bool operator<(const PlanXY& o) const { return (y != o.y) ? (y < o.y) : (x < o.x); }
};

// Manual config knobs. Field names mirror the public ttnn.RegimeAMatmulConfig.
struct RegimeAConfig {
    uint32_t k_slices{1};          // Pk : split-K depth (>=1). Reduction only when >1.
    uint32_t n_slices{1};          // Ns : N-slices per bank-band.
    uint32_t m_slices{1};          // Sm : M-split factor.
    uint32_t k_block_tiles{1};     // kb : K-block depth fed to compute (tiles).
    uint32_t n_subblock_tiles{0};  // nsb: N-subblock width (tiles). 0 => full N_own.
};

struct PlanInputs {
    // Logical (unpadded) tile counts.
    uint32_t Mt{};
    uint32_t Kt{};
    uint32_t Nt{};

    RegimeAConfig cfg{};

    // Compute grid (BH = 11x10).
    uint32_t grid_x{};
    uint32_t grid_y{};

    // Bank-adjacent logical worker cores, one per DRAM bank (8), for each NoC.
    // device->get_optimal_dram_bank_to_logical_worker_assignment(NOC_0 / NOC_1).
    std::vector<PlanXY> opt0;  // size 8
    std::vector<PlanXY> opt1;  // size 8

    // Logical cores that are unavailable (grid holes / reserved). find_near skips these.
    std::set<PlanXY> holes;

    // L1 budget per core (bytes). BH usable ~1440 KB.
    uint32_t l1_budget_bytes{kL1BudgetBytes};

    // Tile byte sizes.
    uint32_t tb{kTileBytesBf16};  // bf16 tile
    uint32_t tf{kTileBytesFp32};  // fp32 tile
};

// ------------------------------------------------------------------------------------------------
// Outputs
// ------------------------------------------------------------------------------------------------

// Geometry with THREE cleanly separated concepts (balanced-tail design):
//   (a) PHYSICAL STRIDES — derived from the tensor layouts; the ONLY values used as address strides.
//   (b) SCHEDULE CAPACITIES — fixed uniform kernel block sizes (may round up to kb*8 / nsb); the loop
//       trip-counts and CB sizes. Invalid block positions are locally zero-filled, never DRAM-read.
//   (c) LOGICAL OWNERSHIP — per-core balanced (start, valid_extent) ranges over the logical tiles.
// Schedule capacities must NEVER be used as a tensor address stride.
struct Geometry {
    uint32_t Mt{}, Kt{}, Nt{};  // logical tile counts

    // (a) physical strides (tiles), from tensor layouts:
    uint32_t in0_stride_k{};        // in0 [Mt,Kt] row stride = Kt
    uint32_t in1_shard_stride_n{};  // in1 per-bank shard width = ceil(Nt/8) (K-row stride within a bank)
    uint32_t out_stride_n{};        // out [Mt,Nt] row stride = Nt

    // (b) schedule capacities (uniform kernel blocks):
    uint32_t K_slice_capacity{};  // K tiles processed per k-slice = rup(ceil(Kt/Pk), kb*8) (== G*W*kb)
    uint32_t M_block_capacity{};  // M tiles per m-block = ceil(Mt/Sm)
    uint32_t N_slice_capacity{};  // N tiles per core = N_bpc * N_sub (rounded to nsb)
    uint32_t N_band{};            // per-bank physical width = ceil(Nt/8) (also == in1_shard_stride_n)
    uint32_t N_sub{};             // N-subblock width (nsb, or N_own when nsb==0)
    uint32_t N_bpc{};             // N-subblocks per core
    uint32_t K_num_blocks_eff{};  // K_slice_capacity / kb  (== G*W, multiple of 8)
    uint32_t W{};                 // ring shard width = K_num_blocks_eff / 8
    uint32_t G{8};                // ring size (fixed 8 banks)

    uint32_t preaders{};   // Pk*Ns*Sm
    uint32_t mfac{};       // Ns*Sm (reduction stride in core index)
    uint32_t num_cores{};  // 8*preaders

    // Reporting only (effective-vs-delivered): schedule zero-fill fraction over K and N.
    double waste_k{}, waste_n{};
};

// Circular-buffer sizing (spec §5).
struct CbSizes {
    uint32_t cb0_tiles{};  // in0 k-slice resident (bf16)
    uint32_t cb1_tiles{};  // in1 (bf16), depth 4
    uint32_t cb2_tiles{};  // out (bf16), depth 2
    uint32_t cb3_tiles{};  // fp32 intermediate accumulator
    uint32_t cb7_tiles{};  // reduce running sum (bf16), depth 2 — 0 when Pk==1
    uint32_t l1_bytes{};   // total
};

// Per-core plan. Ownership is LOGICAL + BALANCED: (start, valid) ranges over the logical tiles, with
// no schedule padding folded into the address offsets. The kernel processes the fixed schedule
// capacities and zero-fills positions >= valid.
struct CorePlan {
    PlanXY coord{};    // assigned logical worker core
    uint32_t bank{};   // DRAM bank id 0..7
    uint32_t noc{};    // 0 => reader on NOC0 / writer on NOC1 ; 1 => swapped
    uint32_t kk{};     // k-slice index 0..Pk-1
    uint32_t nn{};     // n-slice index 0..Ns-1
    uint32_t mm{};     // m-block index 0..Sm-1
    uint32_t slice{};  // p = kk*mfac + nn*Sm + mm  (slice index within a bank)

    // Balanced logical ownership (tiles). start = floor(i*extent/parts); valid = next_start - start.
    uint32_t k_start{};  // first logical K tile of this k-slice (balanced over Pk)
    uint32_t valid_k{};  // K tiles this slice owns (<= K_slice_capacity; tail beyond it is zero)
    uint32_t m_start{};  // first logical M tile of this m-block (balanced over Sm)
    uint32_t valid_m{};  // M tiles this m-block owns (<= M_block_capacity)
    uint32_t n_start{};  // first logical (global) N tile this core owns (for OUTPUT addressing)
    uint32_t valid_n{};  // N tiles this core owns (<= N_slice_capacity)
    uint32_t n_local{};  // n_start - bank*N_band: column offset WITHIN this core's DRAM-bank shard (in1 addr)

    // in0 ring (spec §3): position and cyclic neighbours (physical coords filled by the device op;
    // here we store the ring member core INDICES so the op can translate to physical coords).
    uint32_t ring_pos{};
    uint32_t ring_next_idx{};  // core index of (pos+1)%8 in this ring
    uint32_t ring_prev_idx{};  // core index of (pos+7)%8 in this ring

    // split-K reduction chain (spec §4).
    bool is_bottom{};         // kk == 0
    bool is_top{};            // kk == Pk-1
    uint32_t red_next_idx{};  // core index of next k-slice (i+mfac); self if is_top
    uint32_t red_prev_idx{};  // core index of prev k-slice (i-mfac); self if is_bottom
};

struct ExecutionPlan {
    Geometry geo{};
    CbSizes cb{};
    std::vector<CorePlan> cores;  // size num_cores, index i = bank*preaders + slice

    // Physical in1 DRAM width-shard requirement, per bank, in tiles: [rows, cols]. rows must cover the
    // logical Kt; cols = ceil(Nt/8) (== geo.in1_shard_stride_n). Depends only on (Kt, Nt), not on the
    // schedule config.
    uint32_t in1_shard_rows{};  // Kt
    uint32_t in1_shard_cols{};  // ceil(Nt/8)
};

struct PlanResult {
    std::optional<ExecutionPlan> plan;  // set iff ok
    std::string error;                  // non-empty iff rejected
    bool ok() const { return plan.has_value(); }
};

// ------------------------------------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------------------------------------

inline uint32_t rap_cdiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
inline uint32_t rap_rup(uint32_t x, uint32_t y) { return rap_cdiv(x, y) * y; }

// Width-sharding in1 across 8 banks requires every bank's physical N-interval to intersect logical Nt
// (else trailing banks are wholly padded/empty). N_band = ceil(Nt/8); infeasible iff 7*N_band >= Nt.
// SHARED by the planner (build_plan) and the auto-picker (pick_plan) so config=None never selects a shape
// the planner will later reject — keep this the single source of truth for the bank-interval constraint.
inline bool nt_width_shard_feasible(uint32_t Nt) { return 7u * rap_cdiv(Nt, 8u) < Nt; }

// Balanced prefix range for owner `i` of `parts` over `[0, total)`:
//   start = floor(i*total/parts), end = floor((i+1)*total/parts). Disjoint, exact cover, sizes differ
//   by <= 1. Degenerates to uniform i*(total/parts) when total % parts == 0.
struct BalRange {
    uint32_t start{};
    uint32_t extent{};
};
inline BalRange rap_balanced(uint32_t i, uint32_t total, uint32_t parts) {
    const uint32_t s = static_cast<uint32_t>(static_cast<uint64_t>(i) * total / parts);
    const uint32_t e = static_cast<uint32_t>(static_cast<uint64_t>(i + 1) * total / parts);
    return BalRange{s, e - s};
}

// ------------------------------------------------------------------------------------------------
// Planner
// ------------------------------------------------------------------------------------------------

inline PlanResult build_plan(const PlanInputs& in) {
    PlanResult res;
    const auto& c = in.cfg;
    const uint32_t Pk = c.k_slices ? c.k_slices : 1u;
    const uint32_t Ns = c.n_slices ? c.n_slices : 1u;
    const uint32_t Sm = c.m_slices ? c.m_slices : 1u;
    const uint32_t kb = c.k_block_tiles ? c.k_block_tiles : 1u;

    // --- Basic validation ---
    if (in.Mt == 0 || in.Kt == 0 || in.Nt == 0) {
        res.error = "Mt/Kt/Nt must be > 0";
        return res;
    }
    if (Sm > in.Mt) {
        res.error = "m_slices (Sm) must be <= Mt";
        return res;
    }
    if (Pk > in.Kt) {
        res.error = "k_slices (Pk) must be <= Kt (empty k-slice otherwise)";
        return res;
    }
    if (in.opt0.size() != 8 || in.opt1.size() != 8) {
        res.error = "opt0/opt1 must each have 8 bank-adjacent cores";
        return res;
    }
    // Width-sharding across 8 banks requires every bank's physical N-interval to intersect logical Nt
    // (else the last banks are wholly padded / empty). N_band = ceil(Nt/8); bank b owns [b*N_band, ...).
    const uint32_t N_band = rap_cdiv(in.Nt, 8u);
    if (!nt_width_shard_feasible(in.Nt)) {
        res.error = "Nt too small to width-shard across 8 banks without empty banks (need Nt > 7*ceil(Nt/8))";
        return res;
    }

    // --- Geometry: (a) physical strides, (b) schedule capacities ---
    Geometry g;
    g.Mt = in.Mt;
    g.Kt = in.Kt;
    g.Nt = in.Nt;
    // (a) physical strides — from tensor layouts, the ONLY address strides.
    g.in0_stride_k = in.Kt;
    g.N_band = N_band;
    g.in1_shard_stride_n = N_band;
    g.out_stride_n = in.Nt;
    // (b) schedule capacities — uniform kernel blocks (round to kb*8 / nsb); tail positions zero-filled.
    g.K_slice_capacity = rap_rup(rap_cdiv(in.Kt, Pk), kb * 8u);
    g.M_block_capacity = rap_cdiv(in.Mt, Sm);
    const uint32_t N_own = rap_cdiv(N_band, Ns);  // max N tiles a core owns (balanced subdivision bound)
    g.N_sub = c.n_subblock_tiles ? c.n_subblock_tiles : N_own;
    if (g.N_sub > N_own) {
        res.error = "n_subblock_tiles (nsb) must be <= N_own";
        return res;
    }
    g.N_bpc = rap_cdiv(N_own, g.N_sub);
    g.N_slice_capacity = g.N_bpc * g.N_sub;
    g.K_num_blocks_eff = g.K_slice_capacity / kb;
    if (g.K_num_blocks_eff % 8u != 0u) {
        res.error = "internal: K_num_blocks_eff not a multiple of 8";
        return res;
    }
    g.W = g.K_num_blocks_eff / 8u;
    g.G = 8u;
    g.preaders = Pk * Ns * Sm;
    g.mfac = Ns * Sm;
    g.num_cores = 8u * g.preaders;
    // Schedule zero-fill fraction (compute/NoC-internal overhead; DRAM reads only valid tiles).
    g.waste_k = static_cast<double>(Pk * g.K_slice_capacity) / in.Kt - 1.0;
    g.waste_n = static_cast<double>(8u * Ns * g.N_slice_capacity) / in.Nt - 1.0;

    // --- Core-count feasibility (8 * Pk * Ns * Sm <= available workers) ---
    const uint32_t grid_cells = in.grid_x * in.grid_y;
    const uint32_t available = (grid_cells > in.holes.size()) ? (grid_cells - (uint32_t)in.holes.size()) : 0u;
    if (g.num_cores > available) {
        res.error = "config needs " + std::to_string(g.num_cores) + " cores but only " + std::to_string(available) +
                    " are available on the grid";
        return res;
    }

    // --- CB sizing + L1 check (spec §5; cb7 only when Pk>1) ---
    CbSizes cb;
    cb.cb0_tiles = g.M_block_capacity * g.K_slice_capacity;  // == K_num_blocks_eff * M_block * kb
    cb.cb1_tiles = 4u * kb * g.N_sub;
    cb.cb2_tiles = 2u * g.M_block_capacity * g.N_sub;
    cb.cb3_tiles = g.M_block_capacity * g.N_sub;
    cb.cb7_tiles = (Pk > 1u) ? (2u * g.M_block_capacity * g.N_sub) : 0u;
    cb.l1_bytes = (cb.cb0_tiles + cb.cb1_tiles + cb.cb2_tiles + cb.cb7_tiles) * in.tb + cb.cb3_tiles * in.tf;
    if (cb.l1_bytes > in.l1_budget_bytes) {
        res.error = "L1 over budget: needs " + std::to_string(cb.l1_bytes) + " B > " +
                    std::to_string(in.l1_budget_bytes) + " B";
        return res;
    }

    // --- Placement (spec §2): find_near spiral over the injected grid ---
    std::set<PlanXY> used = in.holes;
    auto in_grid = [&](int x, int y) { return x >= 0 && y >= 0 && (uint32_t)x < in.grid_x && (uint32_t)y < in.grid_y; };
    auto find_near = [&](PlanXY t) -> std::optional<PlanXY> {
        for (int d = 0; d < (int)(in.grid_x + in.grid_y); ++d) {
            for (int dx = -d; dx <= d; ++dx) {
                int rem = d - (dx < 0 ? -dx : dx);
                for (int sgn = 0; sgn <= 1; ++sgn) {
                    int dy = sgn ? -rem : rem;
                    int x = (int)t.x + dx, y = (int)t.y + dy;
                    if (!in_grid(x, y)) {
                        continue;
                    }
                    PlanXY cand{(uint32_t)x, (uint32_t)y};
                    if (used.count(cand)) {
                        continue;
                    }
                    used.insert(cand);
                    return cand;
                }
            }
        }
        return std::nullopt;
    };

    ExecutionPlan plan;
    plan.geo = g;
    plan.cb = cb;
    plan.in1_shard_rows = g.Kt;      // physical shard must cover the logical K tiles
    plan.in1_shard_cols = g.N_band;  // ceil(Nt/8) per bank
    plan.cores.resize(g.num_cores);

    for (uint32_t b = 0; b < 8u; ++b) {
        for (uint32_t p = 0; p < g.preaders; ++p) {
            const uint32_t i = b * g.preaders + p;
            CorePlan cp;
            cp.bank = b;
            cp.slice = p;
            cp.noc = (Sm > 1u) ? ((p / Sm) & 1u) : (p & 1u);
            auto tgt = cp.noc ? in.opt1[b] : in.opt0[b];
            auto placed = find_near(tgt);
            if (!placed) {
                res.error = "collision-free placement failed: no free core near bank " + std::to_string(b);
                return res;
            }
            cp.coord = *placed;

            // decode (spec §2)
            const uint32_t kk = p / g.mfac;
            const uint32_t sub = p % g.mfac;
            cp.kk = kk;
            cp.mm = sub % Sm;
            cp.nn = sub / Sm;

            // --- Balanced logical ownership (no schedule padding in the offsets) ---
            // K over Pk, M over Sm: balanced prefix ranges.
            const BalRange rk = rap_balanced(kk, in.Kt, Pk);
            cp.k_start = rk.start;
            cp.valid_k = rk.extent;
            const BalRange rm = rap_balanced(cp.mm, in.Mt, Sm);
            cp.m_start = rm.start;
            cp.valid_m = rm.extent;
            // N: intersect this bank's physical interval [b*N_band, (b+1)*N_band) with logical [0,Nt),
            // then subdivide that VALID interval across Ns (avoids internal N_own_s holes).
            const uint32_t b_start = b * g.N_band;
            const uint32_t b_end = std::min((b + 1u) * g.N_band, in.Nt);
            const uint32_t b_valid = (b_start < in.Nt) ? (b_end - b_start) : 0u;
            const BalRange rn = rap_balanced(cp.nn, b_valid, Ns);
            cp.n_start = b_start + rn.start;  // global N tile (output addressing)
            cp.valid_n = rn.extent;
            cp.n_local = rn.start;  // column within this bank's shard (in1 addressing)
            if (cp.valid_k == 0u || cp.valid_m == 0u || cp.valid_n == 0u) {
                res.error = "empty ownership for a core (k/m/n) — reduce Pk/Ns/Sm for this shape";
                return res;
            }

            // reduction chain links (spec §4)
            cp.is_bottom = (kk == 0u);
            cp.is_top = (kk == Pk - 1u);
            cp.red_next_idx = cp.is_top ? i : (i + g.mfac);
            cp.red_prev_idx = cp.is_bottom ? i : (i - g.mfac);

            plan.cores[i] = cp;
        }
    }

    // --- Ring membership + ordering (spec §3) ---
    // For each slice index j, the ring is the 8 cores {pos*preaders + j} in BANK ORDER [0..7]. This is the
    // canonical plan output (unit-tested offline); the program factory re-derives the physical PARETO ring
    // order at runtime (optimize_in0_ring_order) from the device's NoC hop distances.
    for (uint32_t j = 0; j < g.preaders; ++j) {
        std::vector<uint32_t> order(8);
        for (uint32_t k = 0; k < 8u; ++k) {
            order[k] = k;
        }
        for (uint32_t pos = 0; pos < 8u; ++pos) {
            const uint32_t ci = order[pos] * g.preaders + j;
            plan.cores[ci].ring_pos = pos;
            plan.cores[ci].ring_next_idx = order[(pos + 1u) % 8u] * g.preaders + j;
            plan.cores[ci].ring_prev_idx = order[(pos + 7u) % 8u] * g.preaders + j;
        }
    }

    res.plan = std::move(plan);
    return res;
}

}  // namespace ttnn::operations::experimental::regime_a_matmul::plan
