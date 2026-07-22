// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Pure, device-free host plan for the fused all-gather + regime_a_matmul op
// (REGIME_A_AGMM_EXECUTION_PLAN.md, Task 2). This header has NO ttnn/tt-metal device dependencies so it
// can be unit-tested offline (and exposed to Python) for D=1/2/4/8, topology, links, K-block ownership,
// core collisions, and L1 sizing — before any fabric/streaming kernels exist (that is Task 3).
//
// Model: each device owns a contiguous K-shard of in0[M, K_global]; the op all-gathers K, then runs
// regime_a_matmul(full in0, in1) per device. This plan represents the multi-device geometry and, crucially,
// the GLOBAL K-BLOCK IDENTITY explicitly (a per-device list of global K-block ids), so ownership is never
// baked into kernel arithmetic (Task 7 will experiment with cyclic/balanced assignment).

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ttnn::operations::experimental::agmm::plan {

// Shared with regime_a (kept local so this header stays dependency-free).
constexpr uint32_t kTileHW = 32u;             // tile height/width (elements)
constexpr uint32_t kTileBytesBf16 = 2048u;    // bf16 tile bytes
constexpr uint32_t kNumBanks = 8u;            // regime_a in1 DRAM width-shard banks
constexpr uint32_t kL1BudgetBytes = 1440u * 1024u;  // BH usable L1 per core
constexpr uint32_t kRegimeCoreWindow = 104u;  // regime_a places compute cores within [16, 104)
constexpr uint32_t kRegimeCoreBase = 16u;

enum class Topology : uint8_t { Linear = 0, Ring = 1 };

struct AgmmPlanConfig {
    // Problem (element dims).
    uint32_t M = 0, K = 0, N = 0;
    // Multi-device geometry.
    uint32_t D = 1;                    // devices along the gather (cluster) axis
    Topology topology = Topology::Ring;
    uint32_t num_links = 1;
    uint32_t num_workers_per_link = 1;
    // Per-device regime_a config (Ns,Pk,Sm,kb,nsb).
    uint32_t Ns = 1, Pk = 1, Sm = 1, kb = 1, nsb = 0;
    // Transport chunk = C * kb K-blocks (Task 3+ knob; default 1). Slot depth for ingress L1.
    uint32_t C = 1;
    uint32_t transport_slots = 2;
    uint32_t packet_bytes = 4096u;     // default 4 KiB fabric packet payload
    // Device worker grid (offline: passed explicitly; BH galaxy usable grid is 12x10=120).
    uint32_t grid_x = 12, grid_y = 10;
};

struct DevicePlan {
    uint32_t device_index = 0;
    std::vector<uint32_t> local_k_blocks;  // explicit GLOBAL K-block ids owned initially by this device
};

struct AgmmPlan {
    // ---- geometry ----
    uint32_t Mt = 0, Kt = 0, Nt = 0;
    uint32_t kb = 1;
    uint32_t global_k_blocks = 0;      // total kb-tile K-blocks over the full (gathered) K
    uint32_t k_blocks_per_device = 0;
    std::vector<DevicePlan> devices;   // size D; union of local_k_blocks == [0, global_k_blocks) exactly once

    // ---- core reservation (fabric reserved BEFORE regime_a compute cores) ----
    uint32_t regime_a_cores = 0;       // 8 * Pk * Ns * Sm
    uint32_t mux_cores = 0;            // num_links * (Ring ? 2 : 1)
    uint32_t fabric_worker_cores = 0;  // num_links * num_workers_per_link
    uint32_t reserved_fabric_cores = 0;
    uint32_t usable_cores = 0;         // grid_x * grid_y
    uint32_t total_cores = 0;          // reserved_fabric + regime_a
    bool core_fit = false;             // total_cores <= usable_cores
    bool core_collision = false;       // reserved fabric region overlaps regime_a compute window

    // ---- L1 sizing (ingress transport buffering) ----
    uint32_t transport_chunk_tiles = 0;  // C * kb * Mt  (in0 tiles moved per transport chunk)
    uint32_t transport_chunk_bytes = 0;  // * bf16 tile bytes
    uint32_t transport_l1_bytes = 0;     // transport_slots * transport_chunk_bytes (per ingress)
    uint32_t l1_budget_bytes = kL1BudgetBytes;
    bool l1_fit = false;

    // ---- overall ----
    bool valid = false;
    std::vector<std::string> errors;
};

inline uint32_t cdiv(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

// Build (and validate) the plan. Never throws; on constraint violation sets valid=false and populates
// errors (so callers can validate cleanly and tests can assert on specific messages).
inline AgmmPlan build_plan(const AgmmPlanConfig& c) {
    AgmmPlan p;
    p.kb = c.kb;
    p.Mt = cdiv(c.M, kTileHW);
    p.Kt = cdiv(c.K, kTileHW);
    p.Nt = cdiv(c.N, kTileHW);
    p.usable_cores = c.grid_x * c.grid_y;
    p.l1_budget_bytes = kL1BudgetBytes;

    auto err = [&](std::string m) { p.errors.push_back(std::move(m)); };

    // ---- constraint validation (BF16 is checked at the op/tensor level; here: shapes/divisibility) ----
    if (c.D == 0) err("D must be >= 1");
    if (c.kb == 0) err("kb must be >= 1");
    if (c.Ns == 0 || c.Pk == 0 || c.Sm == 0) err("Ns/Pk/Sm must be >= 1");
    if (c.C == 0) err("C must be >= 1");
    if (c.transport_slots == 0) err("transport_slots must be >= 1");
    // tile-aligned K sharding: full K is tile-aligned, splits evenly across D, and into whole kb-blocks.
    if (c.K % kTileHW != 0) err("K must be tile-aligned (multiple of 32)");
    if (c.D > 0 && p.Kt % c.D != 0) err("Kt (" + std::to_string(p.Kt) + ") must be divisible by D (" + std::to_string(c.D) + ") for tile-aligned K sharding");
    if (c.kb > 0 && p.Kt % c.kb != 0) err("Kt must be divisible by kb for whole K-blocks");

    // ---- global K-block identity (explicit per-device ownership) ----
    if (c.kb > 0) {
        p.global_k_blocks = p.Kt / c.kb;
        if (c.D > 0 && p.global_k_blocks % c.D == 0) {
            p.k_blocks_per_device = p.global_k_blocks / c.D;
            p.devices.reserve(c.D);
            for (uint32_t d = 0; d < c.D; ++d) {
                DevicePlan dp;
                dp.device_index = d;
                // Contiguous shard for now, but stored as EXPLICIT global block ids (not d*B+i arithmetic
                // in the kernel). Task 7 may replace this with cyclic/balanced assignment.
                for (uint32_t i = 0; i < p.k_blocks_per_device; ++i) {
                    dp.local_k_blocks.push_back(d * p.k_blocks_per_device + i);
                }
                p.devices.push_back(std::move(dp));
            }
        } else if (c.D > 0) {
            err("global_k_blocks (" + std::to_string(p.global_k_blocks) + ") must be divisible by D (" + std::to_string(c.D) + ")");
        }
    }

    // ---- core reservation: reserve fabric mux/worker cores BEFORE regime_a compute cores ----
    p.regime_a_cores = 8u * c.Pk * c.Ns * c.Sm;
    if (c.D == 1) {
        // D=1 delegates to single-chip regime_a: no fabric cores.
        p.mux_cores = 0;
        p.fabric_worker_cores = 0;
    } else {
        p.mux_cores = c.num_links * (c.topology == Topology::Ring ? 2u : 1u);
        p.fabric_worker_cores = c.num_links * c.num_workers_per_link;
    }
    p.reserved_fabric_cores = p.mux_cores + p.fabric_worker_cores;
    p.total_cores = p.reserved_fabric_cores + p.regime_a_cores;
    p.core_fit = p.total_cores <= p.usable_cores;
    if (!p.core_fit) {
        err("core over-subscription: reserved_fabric(" + std::to_string(p.reserved_fabric_cores) +
            ") + regime_a(" + std::to_string(p.regime_a_cores) + ") > usable(" + std::to_string(p.usable_cores) + ")");
    }
    // Collision: regime_a compute cores live in [kRegimeCoreBase, kRegimeCoreWindow); fabric cores are
    // reserved from the grid tail (>= kRegimeCoreWindow) so they do not overlap the compute window.
    // Collision iff regime_a needs more than the window OR fabric reservation reaches into the window.
    const uint32_t window = kRegimeCoreWindow - kRegimeCoreBase;  // 88 compute-core slots
    const bool regime_overflow = p.regime_a_cores > window;
    const bool fabric_into_window = p.reserved_fabric_cores > (p.usable_cores - kRegimeCoreWindow);
    p.core_collision = regime_overflow || (p.reserved_fabric_cores > 0 && fabric_into_window);
    if (p.core_collision) {
        err("core placement collision: regime_a(" + std::to_string(p.regime_a_cores) + " vs window " +
            std::to_string(window) + ") / fabric(" + std::to_string(p.reserved_fabric_cores) + " vs tail " +
            std::to_string(p.usable_cores - kRegimeCoreWindow) + ")");
    }

    // ---- L1 sizing (ingress transport buffering; independent of regime_a's own CBs) ----
    p.transport_chunk_tiles = c.C * c.kb * p.Mt;
    p.transport_chunk_bytes = p.transport_chunk_tiles * kTileBytesBf16;
    p.transport_l1_bytes = c.transport_slots * p.transport_chunk_bytes;
    // Ingress transport buffers must leave room for regime_a's CBs; cap at ~1/3 of L1 for the plan gate.
    const uint32_t transport_l1_cap = p.l1_budget_bytes / 3u;
    p.l1_fit = (c.D == 1) ? true : (p.transport_l1_bytes <= transport_l1_cap);
    if (!p.l1_fit) {
        err("transport L1 over-budget: " + std::to_string(p.transport_l1_bytes) + " > cap " +
            std::to_string(transport_l1_cap) + " (slots=" + std::to_string(c.transport_slots) +
            ", chunk_tiles=" + std::to_string(p.transport_chunk_tiles) + ")");
    }

    p.valid = p.errors.empty();
    return p;
}

}  // namespace ttnn::operations::experimental::agmm::plan
