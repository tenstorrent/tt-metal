// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_regime_a_matmul_async_program_factory.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/experimental/device.hpp>         // get_worker_noc_hop_distance (M-split placement + ring order)
#include <tt-metalium/mesh_device.hpp>                 // MeshDevice (resolve a unit device for the NoC queries)
#include <tt-metalium/experimental/fabric/fabric.hpp>  // FabricMuxV2Config, add_fabric_mux_v2_to_program
#include "ttnn/operations/ccl/ccl_common.hpp"          // rank / neighbour resolution along a cluster axis

#include "ttnn/operations/experimental/regime_a_matmul/device/regime_a_matmul_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::experimental::prim {

namespace {

constexpr const char* kIn1ReaderKernel =
    "ttnn/cpp/ttnn/operations/experimental/all_gather_regime_a_matmul_async/device/kernels/in1_reader.cpp";
constexpr const char* kWriterKernel =
    "ttnn/cpp/ttnn/operations/experimental/all_gather_regime_a_matmul_async/device/kernels/in0_ring_reduce_writer.cpp";
constexpr const char* kComputeKernel =
    "ttnn/cpp/ttnn/operations/experimental/all_gather_regime_a_matmul_async/device/kernels/compute.cpp";

// Tile-byte sizes are defined once in ttnn/operations/experimental/regime_a_matmul/device/regime_a_matmul_plan.hpp
// (single source of truth), reached via `plan::`.
using plan::kTileBytesBf16;
using plan::kTileBytesFp32;

// Largest divisor of v that is <= cap (always >= 1).
uint32_t largest_div(uint32_t v, uint32_t cap) {
    if (v == 0) {
        return 1u;
    }
    for (uint32_t d = std::min(cap, v); d >= 1; --d) {
        if (v % d == 0) {
            return d;
        }
    }
    return 1u;
}

// mkcb: single-format circular buffer over a core range set (matches the harness form).
void mkcb(Program& program, const CoreRangeSet& crs, uint32_t idx, uint32_t ntiles, tt::DataFormat df, uint32_t tsz) {
    CircularBufferConfig c(ntiles * tsz, {{idx, df}});
    c.set_page_size(idx, tsz);
    CreateCircularBuffer(program, crs, c);
}

// M-split (Sm>1) worker PLACEMENT (IN1_NEAR). Overrides ONLY P.cores[i].coord; logical core indices,
// ownership, and the factory's reader->i+s / slave->i-mm runtime-arg math are unchanged. MUST run BEFORE the
// ring reorder so the ring order recomputes on the new coords. Pass 1 places every mm==0 DRAM reader around
// its bank target (a logical-Manhattan spiral mirroring the planner's find_near) so slaves can't displace
// later readers from bank-adjacent cores; pass 2 places each slave at the free worker minimizing the directed
// reader->slave hop on the group's in1-reader NoC. No-op / never called at Sm==1.
void place_m_split_workers(plan::ExecutionPlan& P, IDevice* device, const plan::Geometry& geo) {
    namespace expd = tt::tt_metal::experimental::Device;
    const uint32_t preaders = geo.num_cores / 8u;
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const auto opt0 = device->get_optimal_dram_bank_to_logical_worker_assignment(NOC::NOC_0);
    const auto opt1 = device->get_optimal_dram_bank_to_logical_worker_assignment(NOC::NOC_1);
    std::set<std::pair<uint32_t, uint32_t>> used;
    auto bank_tgt = [&](uint32_t b, uint32_t noc) { return noc ? opt1[b] : opt0[b]; };
    // logical-Manhattan spiral over the compute grid (mirrors the planner's find_near).
    auto find_near = [&](CoreCoord t) -> CoreCoord {
        for (int d = 0; d < (int)(grid.x + grid.y); ++d) {
            for (int dx = -d; dx <= d; ++dx) {
                const int rem = d - (dx < 0 ? -dx : dx);
                for (int sgn = 0; sgn <= 1; ++sgn) {
                    const int dy = sgn ? -rem : rem;
                    const int x = (int)t.x + dx, y = (int)t.y + dy;
                    if (x < 0 || y < 0 || (uint32_t)x >= grid.x || (uint32_t)y >= grid.y) {
                        continue;
                    }
                    const auto key = std::make_pair((uint32_t)x, (uint32_t)y);
                    if (used.count(key)) {
                        continue;
                    }
                    used.insert(key);
                    return CoreCoord{(uint32_t)x, (uint32_t)y};
                }
            }
        }
        return CoreCoord{t.x, t.y};
    };
    auto set_coord = [&](uint32_t i, CoreCoord c) {
        P.cores[i].coord.x = c.x;
        P.cores[i].coord.y = c.y;
    };
    // pass 1: every mm==0 reader around its bank target (readers-first).
    for (uint32_t b = 0; b < 8u; ++b) {
        for (uint32_t p = 0; p < preaders; ++p) {
            const uint32_t i = b * preaders + p;
            if (P.cores[i].mm == 0u) {
                set_coord(i, find_near(bank_tgt(b, P.cores[i].noc)));
            }
        }
    }
    // pass 2: slaves — IN1_NEAR minimizes the directed reader->slave hop on the reader NoC.
    for (uint32_t b = 0; b < 8u; ++b) {
        for (uint32_t p = 0; p < preaders; ++p) {
            const uint32_t i = b * preaders + p;
            if (P.cores[i].mm == 0u) {
                continue;
            }
            const uint32_t ri = i - P.cores[i].mm;  // this group's reader (contiguous index)
            const CoreCoord rc{P.cores[ri].coord.x, P.cores[ri].coord.y};
            const NOC rnoc = P.cores[i].noc ? NOC::NOC_1 : NOC::NOC_0;
            CoreCoord best{};
            uint32_t bestd = 0xffffffffu;
            bool found = false;
            for (uint32_t y = 0; y < grid.y; ++y) {
                for (uint32_t x = 0; x < grid.x; ++x) {
                    if (used.count(std::make_pair(x, y))) {
                        continue;
                    }
                    const uint32_t dd = expd::get_worker_noc_hop_distance(device, rc, CoreCoord{x, y}, rnoc);
                    if (!found || dd < bestd) {
                        bestd = dd;
                        best = CoreCoord{x, y};
                        found = true;
                    }
                }
            }
            used.insert(std::make_pair(best.x, best.y));
            set_coord(i, best);
        }
    }
}

// Physical-topology-aware in0 ring ordering (PARETO). Overrides ring_pos/ring_next_idx/ring_prev_idx per ring
// group using the group's WRITER NoC authoritative hop distance (get_worker_noc_hop_distance; logical->physical
// + directed torus routing w/ wraparound). Placement / work / reduction are unchanged; only the ring visiting
// order (which core seeds which in0 shard, the forward route, the in1 rotated read) changes — correct for ANY
// permutation.
//
// M-split (Sm>1): slices differing only in mm form a (kk,nn) group of Sm CONTIGUOUS slice indices [base,
// base+Sm), all sharing the same writer NoC. Their in1 slaves receive in1 in the mm==0 READER's shard order
// while their in0 rings are separate physical cores, so the WHOLE group MUST use the SAME permutation
// (reader/slave ring_pos must agree per bank) or the in0/in1 pairing corrupts.
//
// PARETO objective (aggregated over the Sm physical mm-rings; aggmax = worst directed edge over all rings,
// aggtot = summed hops over all rings): min aggmax subject to aggtot <= the MM0 order's aggtot, then aggtot.
// MM0 (score only the mm==0 ring: min ring0.max then ring0.total) establishes the aggtot budget and seeds the
// search, so PARETO route-dominates MM0 by construction (never a worse total) — it keeps the Sm=2 win and
// stays within noise of MM0 on Sm=1.
void optimize_in0_ring_order(plan::ExecutionPlan& P, IDevice* device, const plan::Geometry& geo, uint32_t Sm) {
    namespace expdev = tt::tt_metal::experimental::Device;
    const uint32_t preaders = geo.num_cores / 8u;
    // directed route cost of one 8-core cycle over a single ring's hop matrix: (max edge, total hops).
    auto ring_cost = [](const std::array<uint32_t, 8>& ord,
                        const std::array<std::array<uint32_t, 8>, 8>& d) -> std::pair<uint32_t, uint32_t> {
        uint32_t mx = 0, tot = 0;
        for (uint32_t p = 0; p < 8u; ++p) {
            const uint32_t e = d[ord[p]][ord[(p + 1u) % 8u]];
            tot += e;
            mx = std::max(mx, e);
        }
        return {mx, tot};
    };
    for (uint32_t base = 0; base < preaders; base += Sm) {
        // shared writer NoC (opposite the reader's): noc==0 -> writer NOC1; noc==1 -> writer NOC0.
        const NOC wnoc = (P.cores[base].noc == 0u) ? NOC::NOC_1 : NOC::NOC_0;
        // one 8x8 hop matrix per mm-ring (same wnoc, different physical cores).
        std::vector<std::array<std::array<uint32_t, 8>, 8>> dm(Sm);
        for (uint32_t mm = 0; mm < Sm; ++mm) {
            auto lc = [&](uint32_t b) {
                const auto& c = P.cores[b * preaders + base + mm].coord;
                return CoreCoord{c.x, c.y};
            };
            for (uint32_t a = 0; a < 8u; ++a) {
                for (uint32_t b = 0; b < 8u; ++b) {
                    dm[mm][a][b] = (a == b) ? 0u : expdev::get_worker_noc_hop_distance(device, lc(a), lc(b), wnoc);
                }
            }
        }
        // per-candidate metrics across all Sm mm-rings: ring0 (mm==0) max/total; aggmax = worst edge over
        // rings; aggtot = summed hops over all rings.
        struct Metrics {
            uint32_t r0max, r0tot, aggmax, aggtot;
        };
        auto metrics = [&](const std::array<uint32_t, 8>& ord) -> Metrics {
            Metrics m{0, 0, 0, 0};
            for (uint32_t mm = 0; mm < Sm; ++mm) {
                const auto [rm, rt] = ring_cost(ord, dm[mm]);
                if (mm == 0) {
                    m.r0max = rm;
                    m.r0tot = rt;
                }
                m.aggmax = std::max(m.aggmax, rm);
                m.aggtot += rt;
            }
            return m;
        };
        auto lt2 = [](uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) { return a0 < b0 || (a0 == b0 && a1 < b1); };
        auto cand_of = [](const std::array<uint32_t, 7>& t) {
            std::array<uint32_t, 8> c{};
            c[0] = 0;
            for (uint32_t i = 0; i < 7u; ++i) {
                c[i + 1u] = t[i];
            }
            return c;
        };
        const std::array<uint32_t, 8> bank = {0, 1, 2, 3, 4, 5, 6, 7};
        // exhaustive: fix bank 0 at pos 0, permute the other 7 (5040 cycles; directed => both orientations).
        // Pass 1 — MM0 objective (establishes the PARETO aggtot budget).
        std::array<uint32_t, 8> opt_mm0 = bank;
        Metrics b_mm0{~0u, ~0u, ~0u, ~0u};
        std::array<uint32_t, 7> tail = {1, 2, 3, 4, 5, 6, 7};
        do {
            const std::array<uint32_t, 8> cand = cand_of(tail);
            const Metrics m = metrics(cand);
            if (lt2(m.r0max, m.r0tot, b_mm0.r0max, b_mm0.r0tot)) {
                b_mm0 = m;
                opt_mm0 = cand;
            }
        } while (std::next_permutation(tail.begin(), tail.end()));
        // Pass 2 — PARETO: min aggmax (then aggtot) subject to aggtot <= MM0's aggtot. Seeded with MM0 itself
        // (satisfies the constraint by construction).
        std::array<uint32_t, 8> opt_pareto = opt_mm0;
        Metrics b_pa = b_mm0;
        const uint32_t budget = b_mm0.aggtot;
        std::array<uint32_t, 7> tail2 = {1, 2, 3, 4, 5, 6, 7};
        do {
            const std::array<uint32_t, 8> cand = cand_of(tail2);
            const Metrics m = metrics(cand);
            if (m.aggtot <= budget && lt2(m.aggmax, m.aggtot, b_pa.aggmax, b_pa.aggtot)) {
                b_pa = m;
                opt_pareto = cand;
            }
        } while (std::next_permutation(tail2.begin(), tail2.end()));
        // apply the PARETO order to ALL Sm slices of this group (same permutation => reader/slave ring_pos
        // agree per bank, preserving in0/in1 pairing under M-split).
        for (uint32_t mm = 0; mm < Sm; ++mm) {
            const uint32_t jj = base + mm;
            for (uint32_t pos = 0; pos < 8u; ++pos) {
                const uint32_t ci = opt_pareto[pos] * preaders + jj;
                P.cores[ci].ring_pos = pos;
                P.cores[ci].ring_next_idx = opt_pareto[(pos + 1u) % 8u] * preaders + jj;
                P.cores[ci].ring_prev_idx = opt_pareto[(pos + 7u) % 8u] * preaders + jj;
            }
        }
    }
}

// TEST-ONLY (diag bit13 PLACE_MESH): 2D (bank x slice) MESH placement, aimed at in0 ring traffic.
// Host-only and correctness-preserving - writes only P.cores[i].coord.
//
// The structural point: two traffic classes want opposite groupings of the same 8 x preaders core array.
// The in0 RING connects the 8 cores of one SLICE (one per bank), so it wants slice-compact clusters. The
// split-K REDUCTION chain connects the Pk cores of one BANK, so it wants bank-compact clusters. Those are
// orthogonal partitions, so no clustering makes both short - production picks bank-compact blobs (short
// reduction, long ring). A 2D EMBEDDING escapes the tension: put banks along x and slices along y, and then a
// ring step (bank -> bank) and a reduction step (kk -> kk+1) are each ONE hop, in different dimensions.
//
// Offline (in0_ring_place_search.py, exact route model): ring hops -70% AND reduction hops -19..-40%
// simultaneously, whole-op peak link load -11..-15%, total link traffic -20..-36%. in1 read distance is
// ~unchanged (+3%), which is acceptable because the in1 read was measured to be DRAM-bound (76-98% of peak
// in isolation) and insensitive to distance - see IN1_PLACEMENT_AB.md.
//
// Layout: cores (bank b, slice p) -> (x=b, y=p) for p < grid.y; overflow slices each take their own column at
// x >= 8 with the 8 banks down rows 0..7. Collision-free by construction. mm-siblings are consecutive in p,
// so M-split slaves land adjacent to their reader, which keeps the in1 forward short without a separate pass.
void place_mesh(plan::ExecutionPlan& P, const plan::Geometry& geo, const CoreCoord& grid) {
    const uint32_t preaders = geo.num_cores / 8u;
    const uint32_t overflow_cols = (grid.x > 8u) ? (grid.x - 8u) : 0u;
    TT_FATAL(
        grid.x >= 8u && preaders <= grid.y + overflow_cols,
        "all_gather_regime_a_matmul_async mesh placement does not fit: {} slices need <= {} (grid {}x{})",
        preaders,
        grid.y + overflow_cols,
        grid.x,
        grid.y);
    for (uint32_t b = 0; b < 8u; ++b) {
        for (uint32_t p = 0; p < preaders; ++p) {
            const uint32_t i = b * preaders + p;
            if (p < grid.y) {
                P.cores[i].coord.x = b;  // banks along x, slices along y
                P.cores[i].coord.y = p;
            } else {
                P.cores[i].coord.x = 8u + (p - grid.y);  // one spare column per overflow slice
                P.cores[i].coord.y = b;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------
// PHASE 1 fused fabric all-gather: per-device host context.
// ---------------------------------------------------------------------------------------------------
// Built once per mesh coordinate inside create_at, only when tp > 1. Resolves this rank's position in the
// TP group and its two neighbours, then stands up ONE mux v2 per direction. Only the master in0 ring
// (ring group p == 0, i.e. cores i = b*preaders for b in 0..7) connects to the muxes -- per the design
// discussion, forwarding work is not yet spread across the other rings.
struct FusedGatherContext {
    bool enabled = false;
    uint32_t rank = 0;  // this device's index in the TP group
    uint32_t tp = 1;
    // A direction is absent at the ends of a LINE; on a ring both are always present.
    std::optional<CoreCoord> mux_virtual_core_fwd;
    std::optional<CoreCoord> mux_virtual_core_bwd;
    std::unique_ptr<tt::tt_fabric::FabricMuxV2Config> mux_cfg_fwd;
    std::unique_ptr<tt::tt_fabric::FabricMuxV2Config> mux_cfg_bwd;
    uint8_t next_channel_fwd = 0;
    uint8_t next_channel_bwd = 0;
    // Readiness: one semaphore slot per (source rank, chunk). Receivers block on these; senders
    // atomic-inc AFTER the payload is flushed.
    uint32_t chunk_ready_sem_id = 0;   // VALID/INVALID go-ahead flag, on EVERY core
    uint32_t gather_count_sem_id = 0;  // masters-done counter, meaningful on master 0
    // Barrier geometry, in VIRTUAL (translated) coords.
    CoreCoord master0_virtual{};
    std::vector<CoreCoord> release_list;  // every core master 0 releases once the gather is complete
    uint32_t num_masters = 8;
    // Counts forward-stream shard arrivals from the backward neighbour. This is a caller-owned GLOBAL
    // semaphore address, not a program semaphore id: see the note in the operation types header for why a
    // cross-chip credit cannot land in a program semaphore.
    uint32_t fwd_recv_sem_addr = 0;
};

// Offsets WITHIN the fused-gather writer arg block, i.e. relative to fused_rt_base. The kernel reads this
// block sequentially, create_at pushes it in this order, and override_runtime_arguments patches the three
// entries that can change between invocations. All three places must agree, so they name these constants
// rather than counting; kFusedArgCount is checked against the actual push count at the end of the block.
enum FusedGatherArg : uint32_t {
    kFgIsClient = 0,
    kFgRank = 1,
    kFgTp = 2,
    kFgKShardTiles = 3,
    kFgStageAddr = 4,  // patched on replay (caller ping-pongs the staging buffer)
    kFgChunkReadySem = 5,
    kFgHasFwd = 6,
    kFgHasBwd = 7,
    kFgMtTotal = 8,
    kFgKtGlobal = 9,
    kFgShardAddr = 10,   // patched on replay (in0 may be a fresh allocation)
    kFgBankId = 11,      //
    kFgFwdRecvSem = 12,  // patched on replay (caller ping-pongs the global semaphore)
    // On-chip gather barrier: every master reports to master 0, which multicasts one go-ahead to the grid.
    kFgIsMaster0 = 13,
    kFgGatherCountSem = 14,
    kFgNumMasters = 15,
    kFgMaster0X = 16,
    kFgMaster0Y = 17,
    kFgNumAllCores = 18,
    // Followed by kFgNumAllCores (x, y) VIRTUAL coord pairs -- the release list master 0 unicasts to.
    // A multicast would be cheaper but the regime-A placement is bank-adjacent and deliberately NOT a
    // filled rectangle, so a bounding-box mcast would also hit cores running no kernel of ours.
    kFusedArgCount = 19,  // + 2 * num_all_cores coord words, then the mux client block
};

// Mux sizing. Kept deliberately small for bring-up: the design spec says to optimise for the default
// 4 KiB fabric packet, and one channel per direction is enough while a single ring does the forwarding.
constexpr uint8_t kMuxNumChannels = 8;  // one logical channel per master-ring core
constexpr uint8_t kMuxBuffersPerChannel = 8;
constexpr size_t kMuxChannelBufferBytes = 4096;  // 4 KiB packet

FusedGatherContext build_fused_gather_context(
    Program& program,
    const AllGatherRegimeAMatmulAsyncParams& attrs,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const Tensor& in0,
    const std::vector<CoreCoord>& master_ring_cores,
    const CoreRangeSet& all_cores,
    IDevice* device) {
    FusedGatherContext ctx;
    if (attrs.tp <= 1) {
        return ctx;  // single-chip path: no fabric, nothing to build
    }
    ctx.enabled = true;
    ctx.tp = attrs.tp;

    // Forward-only store-and-forward needs a CLOSED ring. On a line, rank 0 has no backward neighbour so
    // its credit never arrives, and rank tp-1 can never pass shards onward. Refuse rather than hang.
    TT_FATAL(
        attrs.topology_is_ring,
        "the fused gather is ring-only today: a forward-only store-and-forward gather deadlocks on a line "
        "(rank 0 is never credited, and the last rank cannot forward). Use Topology::Ring, or the Phase-0 "
        "composition for line topologies");

    const auto topology = attrs.topology_is_ring ? ttnn::ccl::Topology::Ring : ttnn::ccl::Topology::Linear;
    ctx.rank = ttnn::ccl::get_linearized_index_from_physical_coord(in0, mesh_coordinate, attrs.cluster_axis);

    auto* mesh_device = dynamic_cast<tt::tt_metal::distributed::MeshDevice*>(device);
    TT_FATAL(mesh_device != nullptr, "fused gather requires a MeshDevice");
    const auto src_node = mesh_device->get_fabric_node_id(mesh_coordinate);

    // A ring always has both neighbours; a line returns nullopt at its two ends, which is exactly the
    // "no such direction" gate the bidirectional schedule needs.
    const auto fwd_coord =
        ttnn::ccl::get_physical_neighbor_from_physical_coord(in0, mesh_coordinate, 1, topology, attrs.cluster_axis);
    const auto bwd_coord =
        ttnn::ccl::get_physical_neighbor_from_physical_coord(in0, mesh_coordinate, -1, topology, attrs.cluster_axis);

    // Readiness semaphore, on the matmul's own cores (receivers wait on it there).
    const CoreRangeSet ring_crs(std::vector<CoreRange>([&] {
        std::vector<CoreRange> v;
        v.reserve(master_ring_cores.size());
        for (const auto& c : master_ring_cores) {
            v.emplace_back(c, c);
        }
        return v;
    }()));
    // These two stay PROGRAM semaphores on purpose: both sides of the handshake are on this chip and in the
    // same program launch, so the cross-chip early-credit race that forced fwd_recv to be global does not
    // apply -- and dispatch re-zeroes them on every enqueue, which is exactly the re-arm we want.
    // Both live on ALL cores: every core waits on the go-ahead flag, not just the master ring.
    ctx.chunk_ready_sem_id = CreateSemaphore(program, all_cores, 0);  // 0 == INVALID
    ctx.gather_count_sem_id = CreateSemaphore(program, all_cores, 0);
    ctx.num_masters = static_cast<uint32_t>(master_ring_cores.size());
    TT_FATAL(
        !attrs.gather_semaphores.empty(),
        "the fused gather (tp={}) needs at least one caller-supplied global semaphore for the cross-chip "
        "arrival credit; a program semaphore cannot be used here because a peer can credit it before this "
        "device's program has launched and zeroed it",
        attrs.tp);
    ctx.fwd_recv_sem_addr = attrs.gather_semaphores[0].address();

    const size_t mux_base_l1 = device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);

    // The mux cores must not collide with the matmul's compute cores. The matmul occupies the low part of
    // the grid (banks x ring groups), so take mux cores from the TOP row downward.
    const CoreCoord grid = device->compute_with_storage_grid_size();
    auto mux_core_for = [&](uint32_t slot) { return CoreCoord{grid.x - 1u - slot, grid.y - 1u}; };

    auto deploy = [&](const std::optional<ttnn::MeshCoordinate>& dst_coord,
                      uint32_t slot,
                      std::unique_ptr<tt::tt_fabric::FabricMuxV2Config>& cfg_out,
                      std::optional<CoreCoord>& vcore_out) {
        if (!dst_coord.has_value()) {
            return;  // line end: this direction does not exist
        }
        const CoreCoord mux_logical = mux_core_for(slot);
        cfg_out = std::make_unique<tt::tt_fabric::FabricMuxV2Config>(
            kMuxNumChannels, kMuxBuffersPerChannel, kMuxChannelBufferBytes, mux_base_l1);
        tt::tt_fabric::add_fabric_mux_v2_to_program(
            program,
            *cfg_out,
            mux_logical,
            src_node,
            mesh_device->get_fabric_node_id(*dst_coord),
            /*link_idx=*/0,
            tt::tt_metal::NOC::RISCV_0_default);
        vcore_out = device->worker_core_from_logical_core(mux_logical);
    };

    deploy(fwd_coord, 0, ctx.mux_cfg_fwd, ctx.mux_virtual_core_fwd);
    // The backward mux is NOT deployed yet. Mux v2 self-terminates by counting close() calls against its
    // compile-time channel count and has no host-side termination signal (that existed only in v1), so a
    // mux whose 8 registered clients never open/close leaves its forwarder RISC spinning forever and the
    // program never completes. Deploy it in the same commit that makes the kernel drive it, not before.
    (void)bwd_coord;
    return ctx;
}

}  // namespace

ttnn::device_operation::CachedProgram<AllGatherRegimeAMatmulAsyncProgramFactory::shared_variables_t>
AllGatherRegimeAMatmulAsyncProgramFactory::create_at(
    const AllGatherRegimeAMatmulAsyncParams& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const AllGatherRegimeAMatmulAsyncInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    Program program = CreateProgram();

    const auto& in0 = tensor_args.input_tensor;
    const auto& in1 = tensor_args.weight_tensor;
    Tensor& out = tensor_return_value[0];  // chunk 0 (or the sole output when chunks==1)
    IDevice* device = in0.device();

    // get_worker_noc_hop_distance() hard-asserts a unit MeshDevice, so resolve a single representative
    // device for the placement queries below. Those queries only drive core PLACEMENT heuristics (M-split
    // worker siting and in0 ring order) — never the math — so a multi-device mesh can use device 0's
    // physical layout without affecting correctness.
    //
    // CAVEAT: on this Galaxy every chip harvests a DIFFERENT Tensix column, so logical->physical layouts
    // are not identical across the mesh and the placement chosen from device 0 may be mildly suboptimal
    // elsewhere. Acceptable for Phase 0 (correctness first); revisit when tuning fused performance.
    IDevice* noc_ref_device = device;
    if (auto* mesh = dynamic_cast<tt::tt_metal::distributed::MeshDevice*>(device);
        mesh != nullptr && mesh->num_devices() > 1) {
        noc_ref_device = mesh->get_devices().front();
    }

    // Resolve config=None via the auto-selector (deterministic in the tile dims, program-cache-safe).
    // K for PLANNING is always the GLOBAL K, taken from in1 (which is full-K on every device). With the
    // fused gather in0 only carries this rank's [M, K/tp] shard, so planning off in0 would size the whole
    // matmul to a fraction of the real K. in1 is the authority; validate has already checked
    // in1.K == in0.K * tp.
    const uint32_t Mt_r = (static_cast<uint32_t>(in0.logical_shape()[-2]) + 31u) / 32u;
    const uint32_t Kt_r = (static_cast<uint32_t>(in1.logical_shape()[-2]) + 31u) / 32u;
    const uint32_t Nt_r = (static_cast<uint32_t>(in1.logical_shape()[-1]) + 31u) / 32u;
    const RegimeAMatmulConfig cfg = operation_attributes.config.value_or(auto_select_config(Mt_r, Kt_r, Nt_r));

    // ---- Fused epilogue detection (all off => byte-identical no-fusion path). ----
    // Detected BEFORE planning: the fused operand CBs (c_4..c_6) and the reduce-scatter epilogue scratch
    // (c_10) are real L1, so the planner's feasibility check needs them to be authoritative.
    const bool has_bias = tensor_args.bias_tensor.has_value();
    const bool has_ternary = operation_attributes.fused_ternary_scalar.has_value();
    const bool has_activation = operation_attributes.fused_activation.has_value();
    const bool gate_is_fp32 = has_ternary && tensor_args.fused_ternary_input_b->dtype() == DataType::FLOAT32;
    // gate broadcast [1,N] vs full [M,N]. Decide from LOGICAL M, not padded: a full per-row gate with
    // M_logical in 2..32 pads to a single tile row, so padded_shape()/TILE_HEIGHT==1 cannot tell it apart
    // from a real [1,N] broadcast and would silently broadcast row 0 across all M rows. logical M==1 is the
    // only broadcast case, matching validate()'s tb_l[-2]==1 || tb_l[-2]==M check.
    const uint32_t broadcast_gate =
        has_ternary ? (tensor_args.fused_ternary_input_b->logical_shape()[-2] == 1u ? 1u : 0u) : 1u;
    const plan::FusionInputs fusion{
        .has_bias = has_bias,
        .has_ternary = has_ternary,
        .has_activation = has_activation,
        // broadcast_gate defaults to 1 when there is no ternary; only meaningful when has_ternary.
        .broadcast_gate = has_ternary && broadcast_gate != 0u,
        .gate_is_fp32 = gate_is_fp32};

    // ---- Run the pure host planner ----
    // Plan against the GATHERED activation, not this device's shard. make_and_build_plan takes Mt/Kt from
    // in0's logical shape, so handing it the shard plans the whole matmul for K/tp: geo.Kt (and with it the
    // in0 row stride, the k-slice capacity, and the fused block's k_shard_tiles) all come out tp times too
    // small, and every page index the gather computes is wrong. The staging buffer already has exactly the
    // [M, K_global] shape the matmul will actually read, so it is the correct planning input.
    const Tensor& in0_for_plan = (operation_attributes.tp > 1) ? *tensor_args.gather_staging_buffer : in0;
    auto planres = make_and_build_plan(device, in0_for_plan, in1, cfg, fusion);
    TT_FATAL(planres.ok(), "all_gather_regime_a_matmul_async planner rejected config: {}", planres.error);
    plan::ExecutionPlan& P = *planres.plan;  // mutable: the ring-order diag overrides ring_pos/next/prev below
    const plan::Geometry& geo = P.geo;
    const plan::CbSizes& cb = P.cb;

    const uint32_t Pk = cfg.k_slices ? cfg.k_slices : 1u;
    const uint32_t Sm = cfg.m_slices ? cfg.m_slices : 1u;
    const uint32_t kb = cfg.k_block_tiles ? cfg.k_block_tiles : 1u;
    // `use_reduce` doubles as the reduction-CB DEPTH: 0 when Pk==1 (no chain), else the number of cb7 slots.
    // The kernel takes its slot modulus from this same value, so the CB size and the remote write offset can
    // never disagree - the failure mode that produced a PCC 0.38 bug in earlier reduction work.
    // It is deliberately NOT derived from cb.cb7_tiles: that is 0 under reduce-scatter (which allocates
    // c_8/c_9 instead), and a 0 here reads to the kernels as "Pk == 1, no reduction at all", so the writer
    // takes the no-reduce exit while compute still waits for ring partials -> hang.
    const uint32_t use_reduce = (Pk > 1u) ? plan::kCb7Depth : 0u;

    // ---- Output-split detection ----
    const int32_t chunks = operation_attributes.chunks < 1 ? 1 : operation_attributes.chunks;
    const uint32_t n_chunks = static_cast<uint32_t>(chunks);
    const uint32_t out_ntc = Nt_r / n_chunks;  // per-chunk N tiles (validated divisible + tile-aligned)

    // ---- Kernel compile defines. wdefs = writer (in0 ring/reduce + fused output); fdefs_compute (below) =
    // compute fusion defines merged into cdefs. The in1 reader takes NO defines. Empty maps => the
    // byte-identical no-fusion compile. ----
    std::map<std::string, std::string> wdefs;
    // Fused fabric all-gather. A PREPROCESSOR define, not just a compile-time arg: the prologue declares a
    // TensorAccessorArgs that only exists when tp > 1, and `if constexpr` does NOT discard an ill-formed
    // branch in a non-template function -- it would still be compiled and fail deduction on the tp == 1 build.
    if (operation_attributes.tp > 1) {
        wdefs["FUSED_GATHER"] = "1";
    }
    // extra COMPUTE defines beyond fusion; currently only the reduction strategy (RSCATTER).
    std::map<std::string, std::string> cdefs_extra;

    // ---- INTERNAL REDUCTION STRATEGY: linear chain (default) vs ring REDUCE-SCATTER. ----
    // The chain sends each of the Pk-1 non-root bands' FULL output block one hop up, so the last partial only
    // starts moving after Pk-2 earlier hops and the root alone writes all the output. Reduce-scatter instead
    // tile-partitions the block into Pk chunks and rotates them around the Pk cores: the same total number of
    // adds and the same total bytes, but every core sends concurrently every round, and each core ends up
    // owning + writing ONE fully-reduced chunk, so the output write is spread over Pk cores instead of 1.
    //
    // Adopted only on the regime where it was MEASURED to win 5-9% with zero regressions (five corpus shapes:
    // 64/128/256 x 2048 x 1024/2048): Pk>=4, shallow K (the deep-K shapes are in1-read-bound, so the reduction
    // is not on the critical path), wide enough N, N_sub>=2, and a block that partitions evenly over Pk.
    // Fusion and chunked output ARE supported: each owner applies the epilogue exactly once to its own fully
    // reduced slice, and the writer feeds that owner only its slice's operands, in slice order, so compute
    // indexes them 0..nt-1. The ring's send/recv CBs live at c_8/c_9 so they cannot collide with the fusion
    // operand CBs c_4/c_5/c_6.
    //
    // NOTE: reduce-scatter re-associates the K-sum (each owner adds the Pk partials in ring order instead of
    // bottom-to-top), so it is PCC-preserving but NOT bit-identical to the chain. Every add stays in FP32 and
    // no operand narrows, so this is a different summation ORDER, not a precision reduction.
    //
    // bit22 = FORCE_CHAIN (A/B the chain baseline), bit23 = FORCE_RSCATTER (test it outside the gate). Any
    // kernel-behaviour diagnostic (ablations, meet) keeps the chain so the ablation floors stay comparable.
    // The partition does NOT need to divide evenly: chunk sizes differ by at most one tile, so the only
    // structural requirement is rs_T >= Pk (every core must own at least one tile to write). Requiring
    // divisibility locked out 41 of the 62 corpus shapes for no reason.
    const uint32_t rs_T = geo.M_block_capacity * geo.N_sub;  // tiles per output sub-block
    // No N-WIDTH requirement. The original gate also demanded Nt>=32; measuring the declined shapes with bit23
    // showed that was wrong - 128x2048x512 (Nt=16) is 8.9% FASTER with reduce-scatter. Every shallow-K Pk>=4
    // shape with N_sub>=2 won (6/6, 5.5-14.7%) regardless of N width, and N_sub==1 was neutral (+0.4%).
    //
    // DEEP K is admitted only under two extra conditions, because reduce-scatter trades data movement for
    // per-round compute SETUP cost (each round pays an add_tiles_init + data-format reconfig no matter how few
    // tiles it touches). It therefore needs FEW rounds and ENOUGH WORK per round:
    //   Pk <= 6        -> at most 5 rounds per sub-block. At Pk=12 the 11 rounds cost far more than the
    //                     shortened critical path saves: 512x6144x4608 +33.4%, 512x6144x2304 +19.5%,
    //                     128x15360x1536 +3.0%.
    //   max_chunk >= 2 -> a round that moves a single 2 KB tile is almost pure overhead: 64x15360x1536
    //                     (Pk=6, 1-tile chunks) is +2.7% despite satisfying the round bound.
    // Together these separate all 13 measured deep-K shapes exactly: the 6 satisfying both won (-1.3% to -3.8%)
    // and all 5 losses are excluded. Shallow K still wins with single-tile chunks (compute has far more slack
    // there - 30-40% of the wall vs 47-77%), so the chunk-size floor is deep-K only.
    // The gate itself lives in plan::rscatter_selected() so that the L1 accounting charges for exactly the
    // buffers this decision allocates. cb.rscatter is that same call, made during planning; assert they agree
    // rather than trusting it, since a divergence would silently under-charge L1.
    const bool rscatter = plan::rscatter_selected(Pk, Kt_r, geo.M_block_capacity, geo.N_sub);
    TT_FATAL(
        rscatter == cb.rscatter,
        "internal: reduction strategy disagrees with the L1 accounting (factory={}, planner={})",
        rscatter,
        cb.rscatter);
    if (rscatter) {
        wdefs["RSCATTER"] = "1";
        cdefs_extra["RSCATTER"] = "1";
    }

    // ---- M-split worker PLACEMENT (Sm>1): IN1_NEAR. Overrides only P.cores[i].coord; MUST run BEFORE the ring
    // reorder so the ring order recomputes on the placed coords. No-op at Sm==1. ----
    // PRODUCTION GATE for the 2D mesh placement. Adopt only when the mesh FILLS the grid: preaders >= 10
    // puts at least one slice in every grid row, so cores spread over the whole 11x10 array. Below that the
    // mesh packs all 8*preaders cores into rows 0..preaders-1 and concentrates every DRAM path into a corner -
    // measured -48% to -89% at preaders<=5 and -6% to -23% at preaders 6..7. Sm>1 is excluded because M-split
    // slaves then lose the IN1_NEAR pass and the measured results are mixed (+8.7% to -22.2%), and Pk>=4 is
    // required for the Ns>1 case (Pk=3/Ns=4 measured neutral-to-negative).
    // Fitted on the 63-shape corpus at deployed configs: adopts 24/63 shapes, mean +5.06%, best +14.98%,
    // worst -1.27%; every declined shape stays byte-identical to the previous production placement.
    // Second clause: adopt regardless of the above when the in0 RING simply carries more traffic than the in1
    // read - at ring >= 2x in1 the ring savings dominate whatever the placement costs the read. On the corpus
    // exactly 3 shapes clear 2x and all 3 win (+14.64%, +9.09%, +8.70%); the highest-ratio loser is at 1.31x.
    // This is what lets the two Sm>1 ring-heavy shapes in (256x15360x768, 256x2048x512).
    const uint32_t Ns_gate = cfg.n_slices ? cfg.n_slices : 1u;
    const uint64_t ring_bytes =
        static_cast<uint64_t>(geo.num_cores) * 7u * geo.W * geo.M_block_capacity * kb * kTileBytesBf16;
    const uint64_t in1_bytes = static_cast<uint64_t>(geo.Kt) * geo.Nt * kTileBytesBf16;
    // Mt >= 8 is REQUIRED. The mesh trades in1 read locality (cores leave their own DRAM bank) for a ~70% cut
    // in in0 ring traffic, and ring traffic per shard scales with M_block = Mt/Sm. At Mt <= 4 the shards are
    // so small that there is almost no ring traffic to save while the read penalty is paid in full: measured
    // 9.6-13.1% SLOWER on 32x6080x4640, 128x6144x4608, 64x6144x768, 128x6144x2304, 64x15360x768 and
    // 128x15360x1536 (all Pk=12, i.e. inside the old gate). At Mt >= 8 it is 16-24% FASTER on the same
    // topology (512x6144x2304 +16.4%, 256x6144x768 +23.9%).
    const bool mesh_gate = geo.Mt >= 8u && (((Pk * Ns_gate >= 10u) && (Sm == 1u) && (Ns_gate == 1u || Pk >= 4u)) ||
                                            (ring_bytes >= 2u * in1_bytes));
    const bool use_mesh = mesh_gate;
    // OBSERVABILITY ONLY (TT_REGIME_A_LOG_CFG): report what the picker and the internal gates actually chose.
    // Runs once per program-cache miss and changes NOTHING about behaviour -- there is no way to read the
    // auto-selected config from Python otherwise, and reporting a host-side mirror of the picker risks silently
    // misreporting if the mirror drifts from auto_select_config.
    if (std::getenv("TT_REGIME_A_LOG_CFG") != nullptr) {
        log_info(
            tt::LogOp,
            "regime_a_cfg M={} K={} N={} pick=({},{},{},{},{}) cores={} reduction={} placement={}",
            Mt_r * 32u,
            Kt_r * 32u,
            Nt_r * 32u,
            Pk,
            cfg.n_slices ? cfg.n_slices : 1u,
            Sm,
            kb,
            cfg.n_subblock_tiles,
            geo.num_cores,
            rscatter ? "reduce-scatter" : "chain",
            use_mesh ? "mesh" : (Sm > 1u ? "in1-near" : "bank-local"));
    }
    if (use_mesh) {
        place_mesh(P, geo, device->compute_with_storage_grid_size());
    } else if (Sm > 1u) {
        place_m_split_workers(P, noc_ref_device, geo);
    }

    // ---- Physical-topology-aware in0 ring ordering (PARETO) over each (kk,nn) group's Sm mm-rings. ----
    optimize_in0_ring_order(P, noc_ref_device, geo, Sm);

    // ---- REDUCE-SCATTER cyclic order over each group's Pk cores (runs AFTER placement + ring ordering, so it
    // sees the final coords). For each (bank b, within-bank sub) group, order the Pk k-slice cores into a
    // Hamiltonian CYCLE minimizing the worst DIRECTED NoC hop: one bad wraparound edge would serialize every
    // round of the ring, so the max edge matters more than the total. The edge cost a->c is measured on the
    // SENDER's writer NoC (writer runs opposite the core's in1-reader NoC), which is asymmetric on the torus
    // and therefore cannot be approximated by a coordinate distance. Pk==4 searches all 3! orders exactly;
    // larger Pk uses greedy nearest-neighbour (P! is infeasible). Mutates only the rs_* fields. ----
    if (rscatter) {
        namespace expdev = tt::tt_metal::experimental::Device;
        for (uint32_t b = 0; b < 8u; ++b) {
            for (uint32_t sub = 0; sub < geo.mfac; ++sub) {
                std::vector<uint32_t> idx(Pk);
                for (uint32_t kk = 0; kk < Pk; ++kk) {
                    idx[kk] = b * geo.preaders + kk * geo.mfac + sub;
                }
                auto lc = [&](uint32_t li) {
                    const auto& c = P.cores[idx[li]].coord;
                    return CoreCoord{c.x, c.y};
                };
                auto dist = [&](uint32_t a, uint32_t c) -> uint32_t {
                    if (a == c) {
                        return 0u;
                    }
                    const NOC wnoc = (P.cores[idx[a]].noc == 0u) ? NOC::NOC_1 : NOC::NOC_0;
                    return expdev::get_worker_noc_hop_distance(device, lc(a), lc(c), wnoc);
                };
                std::vector<uint32_t> ord(Pk);
                if (Pk == 4u) {
                    // exact min-(maxedge, total) cycle over the 3! orderings of {1,2,3} (position 0 fixed)
                    const uint32_t perms[6][3] = {{1, 2, 3}, {1, 3, 2}, {2, 1, 3}, {2, 3, 1}, {3, 1, 2}, {3, 2, 1}};
                    uint32_t best_max = ~0u, best_tot = ~0u;
                    for (const auto& pm : perms) {
                        const uint32_t o[4] = {0u, pm[0], pm[1], pm[2]};
                        uint32_t mx = 0, tot = 0;
                        for (uint32_t p = 0; p < 4u; ++p) {
                            const uint32_t e = dist(o[p], o[(p + 1u) % 4u]);
                            mx = std::max(mx, e);
                            tot += e;
                        }
                        if (mx < best_max || (mx == best_max && tot < best_tot)) {
                            best_max = mx;
                            best_tot = tot;
                            for (uint32_t p = 0; p < 4u; ++p) {
                                ord[p] = o[p];
                            }
                        }
                    }
                } else {
                    std::vector<bool> vis(Pk, false);
                    ord[0] = 0u;
                    vis[0] = true;
                    for (uint32_t p = 1; p < Pk; ++p) {
                        uint32_t best = 0, bestd = ~0u;
                        for (uint32_t cand = 0; cand < Pk; ++cand) {
                            if (vis[cand]) {
                                continue;
                            }
                            const uint32_t dd = dist(ord[p - 1], cand);
                            if (dd < bestd) {
                                bestd = dd;
                                best = cand;
                            }
                        }
                        ord[p] = best;
                        vis[best] = true;
                    }
                }
                for (uint32_t p = 0; p < Pk; ++p) {
                    auto& cp = P.cores[idx[ord[p]]];
                    cp.rs_pos = p;
                    cp.rs_next_idx = idx[ord[(p + 1u) % Pk]];
                    cp.rs_prev_idx = idx[ord[(p + Pk - 1u) % Pk]];
                    cp.rs_own_chunk = (p + 1u) % Pk;
                }
            }
        }
    }

    // ---- Fused-epilogue / output-split kernel defines (empty => byte-identical no-fusion compile). ----
    // Compute-only fusion defines are collected here and merged into cdefs at compute-kernel creation.
    std::map<std::string, std::string> fdefs_compute;
    if (has_bias) {
        wdefs["FUSE_BIAS"] = "1";
        fdefs_compute["FUSE_BIAS"] = "1";
    }
    if (has_ternary) {
        wdefs["FUSE_TERNARY"] = "1";
        fdefs_compute["FUSE_TERNARY"] = "1";
        if (gate_is_fp32) {
            wdefs["TERNARY_B_IS_FLOAT32"] = "1";
            fdefs_compute["TERNARY_B_IS_FLOAT32"] = "1";
        }
    }
    if (n_chunks > 1u) {
        wdefs["OUT_CHUNKS"] = "1";
    }
    if (has_activation) {
        auto act = ttnn::operations::unary::utils::get_defines(
            operation_attributes.fused_activation->op_type,
            operation_attributes.fused_activation->params,
            "ACTIVATION",
            "fused_act_dst_id",
            out.dtype());
        fdefs_compute.insert(act.begin(), act.end());
    }

    // ---- Core range sets: all cores + split-NoC groups (g0 = noc 0, g1 = noc 1) ----
    std::set<CoreRange> all_set, g0_set, g1_set;
    std::vector<CoreCoord> cores;
    std::vector<uint32_t> core_noc;
    cores.reserve(geo.num_cores);
    core_noc.reserve(geo.num_cores);
    for (const auto& cp : P.cores) {
        CoreCoord c{cp.coord.x, cp.coord.y};
        cores.push_back(c);
        core_noc.push_back(cp.noc);
        all_set.insert(CoreRange(c, c));
        (cp.noc ? g1_set : g0_set).insert(CoreRange(c, c));
    }
    CoreRangeSet all_cores(all_set);
    CoreRangeSet g0(g0_set);
    CoreRangeSet g1(g1_set);

    // ---- PHASE 1 fused fabric all-gather: per-device host context ----
    // The MASTER in0 ring is ring group p == 0, i.e. core indices i = b*preaders for b in 0..7. Those 8
    // cores are the only fabric clients for now (load-balancing forwarding across the other rings is a
    // deliberate follow-up). Built here, after `cores` exists and before runtime args are emitted.
    const uint32_t preaders_pf = geo.num_cores / 8u;
    std::vector<CoreCoord> master_ring_cores;
    master_ring_cores.reserve(8);
    for (uint32_t b = 0; b < 8u; ++b) {
        master_ring_cores.push_back(cores[b * preaders_pf]);
    }
    FusedGatherContext fused_gather = build_fused_gather_context(
        program, operation_attributes, mesh_coordinate, in0, master_ring_cores, all_cores, device);

    // ---- On-chip gather barrier geometry ----
    // A master core's fwd_recv count only proves ITS OWN M slice arrived: the gather splits M by bank_id,
    // while the matmul splits M by the planner's m_start. Those partitions differ, so every core -- master
    // or not -- can read rows that a DIFFERENT core gathered. Each master therefore reports completion to
    // master 0, which multicasts a single go-ahead to the whole grid.
    if (fused_gather.enabled) {
        // worker_core_from_logical_core returns TRANSLATED (virtual) coords, in which the Blackhole grid is
        // dense whatever the harvesting mask. Raw physical coords would straddle harvested columns here.
        fused_gather.master0_virtual = device->worker_core_from_logical_core(master_ring_cores[0]);
        fused_gather.release_list.reserve(cores.size());
        for (const auto& c : cores) {
            fused_gather.release_list.push_back(device->worker_core_from_logical_core(c));
        }
        // Two coord words per core ride in every core's arg block. Runtime args are a bounded resource, so
        // cap this rather than silently overflowing; past this size the barrier wants a per-CoreRange
        // multicast instead of a unicast fan-out.
        TT_FATAL(
            fused_gather.release_list.size() <= 64u,
            "fused gather barrier currently unicasts the release to each of {} cores, which exceeds the "
            "64-core runtime-arg budget; switch to a per-CoreRange multicast for grids this large",
            fused_gather.release_list.size());
    }

    // NOTE: packet headers come from PacketHeaderPool (a per-RISC L1 region), NOT from a circular buffer.
    // The CB carve-out is the older pattern; the pool is what current fabric kernels use.

    // ---- Circular buffers (spec §5) on all cores ----
    mkcb(program, all_cores, 0, cb.cb0_tiles, tt::DataFormat::Float16_b, kTileBytesBf16);  // in0 k-slice resident
    mkcb(program, all_cores, 1, cb.cb1_tiles, tt::DataFormat::Float16_b, kTileBytesBf16);  // in1 (depth 4)
    mkcb(program, all_cores, 2, cb.cb2_tiles, tt::DataFormat::Float16_b, kTileBytesBf16);  // out
    mkcb(program, all_cores, 3, cb.cb3_tiles, tt::DataFormat::Float32, kTileBytesFp32);    // fp32 intermediate
    // cb7 is the CHAIN's running-sum buffer. Reduce-scatter never touches it (its partials travel through the
    // c_4/c_5 chunk CBs instead), so don't spend the L1 on it there.
    // cb7_tiles is already 0 under reduce-scatter (the sizer decides that), so this one test covers both
    // "Pk == 1, no chain" and "reduce-scatter, partials travel via c_8/c_9 instead".
    if (cb.cb7_tiles > 0u) {
        mkcb(program, all_cores, 7, cb.cb7_tiles, tt::DataFormat::Float16_b, kTileBytesBf16);  // reduce (Pk>1)
    }
    // Fused-epilogue operand CBs (only when the matching fusion is active). c_4 bias [1,N_sub], c_5 residual
    // [M,N] block, c_6 gate [1,N_sub] (broadcast) or [M,N] block. Sized to hold a full sub-block so the
    // writer can stream all M rows while compute consumes them (matches minimal_matmul's ternary CB sizing).
    // Sizes come from the planner's CbSizes (plan::compute_cb_sizes), never recomputed here: what feasibility
    // charged and what we allocate are then the same numbers by construction.
    if (has_bias) {
        mkcb(program, all_cores, 4, cb.cb4_tiles, tt::DataFormat::Float16_b, kTileBytesBf16);
    }
    if (has_ternary) {
        mkcb(program, all_cores, 5, cb.cb5_tiles, tt::DataFormat::Float16_b, kTileBytesBf16);
        const tt::DataFormat gfmt = gate_is_fp32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
        const uint32_t gtsz = gate_is_fp32 ? kTileBytesFp32 : kTileBytesBf16;
        mkcb(program, all_cores, 6, cb.cb6_tiles, gfmt, gtsz);
    }
    // REDUCE-SCATTER ring CBs at c_8/c_9 -- deliberately NOT c_4/c_5, which are fusion operands, so the ring
    // and the fused epilogue coexist. EXACTLY 2 slots each, sized to the LARGEST slice
    // (chunks differ by at most one tile when Pk does not divide the sub-block): the kernel's slot index is the
    // global epoch mod 2 and every CB operation moves a whole max-size slot, so the FIFO period and the remote
    // write stride are the same value by construction.
    if (rscatter) {
        mkcb(program, all_cores, 8, cb.cb8_tiles, tt::DataFormat::Float16_b, kTileBytesBf16);
        mkcb(program, all_cores, 9, cb.cb9_tiles, tt::DataFormat::Float16_b, kTileBytesBf16);
        if (cb.cb10_tiles > 0u) {
            // c_10: scratch for the reduced slice while the fused epilogue is applied to it in place.
            mkcb(program, all_cores, 10, cb.cb10_tiles, tt::DataFormat::Float16_b, kTileBytesBf16);
        }
    }

    // ---- Semaphores ----
    const uint32_t fwd_sem = CreateSemaphore(program, all_cores, 0u);      // in0 ring recv
    const uint32_t red_sem = CreateSemaphore(program, all_cores, 0u);      // reduction recv (shared counter)
    const uint32_t redfree_sem = CreateSemaphore(program, all_cores, 0u);  // cb_reduce reverse credit
    uint32_t in1valid_sem = 0u, in1ready_sem = 0u;                         // M-split reader<->slaves
    if (Sm > 1u) {
        in1valid_sem = CreateSemaphore(program, all_cores, 0u);
        in1ready_sem = CreateSemaphore(program, all_cores, 0u);
    }

    // ---- Kernels ----
    // in1 reader == consumer. compile args (in1_reader.cpp order). No TensorAccessorArgs.
    std::vector<uint32_t> rct = {
        kb,                      // 0 K_block
        geo.N_sub,               // 1 N_block
        geo.W,                   // 2 W
        geo.G,                   // 3 G (=8)
        kTileBytesBf16,          // 4 tile_bytes
        geo.N_bpc,               // 5 N_bpc
        geo.in1_shard_stride_n,  // 6 in1_shard_stride_n (physical per-bank width)
        in1valid_sem,            // 7
        in1ready_sem};           // 8

    auto mk = [&](const char* src,
                  const CoreRangeSet& g,
                  DataMovementProcessor proc,
                  NOC noc,
                  const std::vector<uint32_t>& ct,
                  const std::map<std::string, std::string>& defs) -> KernelHandle {
        if (g.num_cores() == 0) {
            return 0;
        }
        return CreateKernel(
            program, src, g, DataMovementConfig{.processor = proc, .noc = noc, .compile_args = ct, .defines = defs});
    };

    // writer compile args (in0_ring_reduce_writer.cpp order). TensorAccessorArgs(in0) then (out).
    // Index in the writer's runtime args where the fused-gather block begins. MUST mirror the wa push
    // order below exactly: 17 fixed, then bias, ternary, chunk, and reduce-scatter args. Asserted against
    // the real wa.size() at emission time so the two can never silently drift apart.
    const uint32_t fused_rt_base = 17u + (has_bias ? 1u : 0u) + (has_ternary ? 3u : 0u) +
                                   ((n_chunks > 1u) ? (2u + (n_chunks - 1u)) : 0u) + (rscatter ? 7u : 0u);

    std::vector<uint32_t> wct = {
        geo.M_block_capacity,  // 0
        kb,                    // 1 K_block
        geo.N_sub,             // 2 N_block
        geo.K_num_blocks_eff,  // 3 K_num_blocks
        kTileBytesBf16,        // 4 tile_bytes
        geo.in0_stride_k,      // 5 in0 row stride (physical = Kt)
        geo.out_stride_n,      // 6 out row stride (physical = Nt)
        geo.W,                 // 7 W
        geo.G,                 // 8 G
        fwd_sem,               // 9
        red_sem,               // 10
        geo.N_bpc,             // 11 N_bpc
        redfree_sem,           // 12
        use_reduce,            // 13
        // ---- fused fabric all-gather (Phase 1) ----
        // 14: 0 => the whole gather prologue compiles out, so tp==1 is byte-identical.
        // 15: index into the writer's runtime args where the fused block starts. Passed explicitly because
        //     the block sits after a variable number of optional bias / ternary / chunk / rscatter args.
        (operation_attributes.tp > 1) ? 1u : 0u,  // 14 fused_gather_enabled
        fused_rt_base};                           // 15 fused_rt_base
    // in0 ACCESSOR: on the fused path the matmul reads the GATHERED activation out of the staging buffer,
    // not the local shard. The staging buffer is [M, K_global] so the existing (m*Kt + k) addressing and the
    // geo.in0_stride_k row stride are already correct for it -- the matmul body needs no change at all, the
    // accessor just points somewhere else.
    const Tensor& in0_for_matmul = (operation_attributes.tp > 1) ? *tensor_args.gather_staging_buffer : in0;
    TensorAccessorArgs(*in0_for_matmul.buffer()).append_to(wct);
    TensorAccessorArgs(*out.buffer()).append_to(wct);
    // Fused-operand accessors, in the order the writer kernel expects: bias, then residual/gate.
    if (has_bias) {
        TensorAccessorArgs(*tensor_args.bias_tensor->buffer()).append_to(wct);
    }
    if (has_ternary) {
        TensorAccessorArgs(*tensor_args.fused_ternary_input_a->buffer()).append_to(wct);
        TensorAccessorArgs(*tensor_args.fused_ternary_input_b->buffer()).append_to(wct);
    }
    // LOCAL-SHARD accessor for the fused gather. Appended LAST, after every existing accessor, so adding it
    // cannot shift the indices the kernel already resolves by chaining. Only present/read when tp > 1.
    if (operation_attributes.tp > 1) {
        TensorAccessorArgs(*in0.buffer()).append_to(wct);
    }

    // Split-NOC: reader on the core's in1 NoC, writer on the OTHER NoC.
    //   g0 (noc==0): reader RISCV_0/NOC0, writer RISCV_1/NOC1
    //   g1 (noc==1): reader RISCV_1/NOC1, writer RISCV_0/NOC0
    // in1 reader takes no compile defines.
    const std::map<std::string, std::string> rdefs;
    KernelHandle readerA = mk(kIn1ReaderKernel, g0, DataMovementProcessor::RISCV_0, NOC::RISCV_0_default, rct, rdefs);
    KernelHandle readerB = mk(kIn1ReaderKernel, g1, DataMovementProcessor::RISCV_1, NOC::RISCV_1_default, rct, rdefs);
    KernelHandle writerA = mk(kWriterKernel, g0, DataMovementProcessor::RISCV_1, NOC::RISCV_1_default, wct, wdefs);
    KernelHandle writerB = mk(kWriterKernel, g1, DataMovementProcessor::RISCV_0, NOC::RISCV_0_default, wct, wdefs);

    // compute (spec §6c). fp32 DST limit: subblock_h * subblock_w <= 4.
    // Subblock geometry. fp32_dest_acc_en gives 4 DST tiles, so the hard limit is sbh*sbw <= 4 (exceeding it
    // silently corrupts output - there is a known precedent, hence the assert below). The historical sizer
    // caps subblock_h at 2, so when N_sub == 1 it yields 2x1 = 2 tiles and leaves HALF the DST idle. We now
    // ENLARGE such subblocks to the biggest area that still fits (4x1 when N_sub==1 and M_block%4==0), which
    // halves the matmul_block call count for identical math - verified BIT-EXACT. Measured on the 63-shape
    // corpus: +2.56% on 512x6144x4608, +2.25% on 512x6144x2304, and within noise everywhere else.
    // diag bit16 restores the legacy sizer for A/B.
    uint32_t sbh = largest_div(geo.M_block_capacity, 2u);
    uint32_t sbw = largest_div(geo.N_sub, 4u / sbh);
    // Only ever ENLARGE the subblock: where the historical sizer already reaches the 4-tile limit, keep its
    // exact shape. Re-shaping an already-maximal subblock (e.g. 2x2 -> 1x4) is not free - it measured -2.22%
    // on 256x2048x1024 - so the rule is a strict Pareto improvement by construction.
    if (sbh * sbw < 4u) {
        uint32_t bh = sbh, bw = sbw;
        for (uint32_t h = 1u; h <= 4u; ++h) {
            if (geo.M_block_capacity % h != 0u) {
                continue;
            }
            for (uint32_t w = 1u; h * w <= 4u; ++w) {
                if (geo.N_sub % w != 0u) {
                    continue;
                }
                if (h * w > bh * bw || (h * w == bh * bw && w > bw)) {
                    bh = h;
                    bw = w;
                }
            }
        }
        sbh = bh;
        sbw = bw;
    }
    TT_FATAL(
        sbh * sbw <= 4u, "all_gather_regime_a_matmul_async subblock {}x{} exceeds the 4-tile fp32 DST limit", sbh, sbw);
    std::vector<uint32_t> cct = {
        geo.K_num_blocks_eff,  // 0 K_num_blocks
        geo.M_block_capacity,  // 1 M_block_tiles
        kb,                    // 2 K_block_tiles
        geo.N_sub,             // 3 N_block_tiles
        1u,                    // 4 M_blocks_per_core
        geo.N_bpc,             // 5 N_blocks_per_core
        sbh,                   // 6 subblock_h
        sbw};                  // 7 subblock_w
    std::map<std::string, std::string> cdefs = {{"REDUCE_K", "1"}, {"IN0_KSLICE_RESIDENT", "1"}};
    cdefs.insert(fdefs_compute.begin(), fdefs_compute.end());  // fusion defines (empty for the no-fusion path)
    cdefs.insert(cdefs_extra.begin(), cdefs_extra.end());      // reduction strategy (RSCATTER when gated in)
    KernelHandle compute = CreateKernel(
        program,
        kComputeKernel,
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi2,
            .fp32_dest_acc_en = true,
            .dst_full_sync_en = false,
            .math_approx_mode = false,
            .compile_args = cct,
            .defines = cdefs});

    // ---- Runtime args ----
    const uint32_t in0_addr = in0_for_matmul.buffer()->address();  // staging buffer when tp>1
    const uint32_t in1_addr = in1.buffer()->address();
    const uint32_t out_addr = out.buffer()->address();

    auto phys = [&](uint32_t core_idx) {
        const auto& c = P.cores[core_idx].coord;
        return device->worker_core_from_logical_core(CoreCoord{c.x, c.y});
    };

    for (uint32_t i = 0; i < geo.num_cores; ++i) {
        const plan::CorePlan& cp = P.cores[i];
        const KernelHandle rh = cp.noc ? readerB : readerA;
        const KernelHandle wh = cp.noc ? writerB : writerA;

        // in1 reader runtime args.
        std::vector<uint32_t> ra = {
            in1_addr,     // 0
            cp.bank,      // 1
            cp.ring_pos,  // 2
            cp.k_start,   // 3 first logical K tile (balanced)
            cp.n_local,   // 4 within-bank column offset
            cp.valid_k,   // 5 valid K tiles (rest of capacity zero-filled)
            cp.valid_n};  // 6 valid N tiles this core owns
        if (Sm == 1u) {
            ra.push_back(2u);  // 5 mrole = solo
            ra.push_back(0u);  // 6 mpeers
        } else if (cp.mm == 0u) {
            // reader (mm==0 of this (bank,kk,nn) group): read from DRAM + forward to the Sm-1 slaves.
            ra.push_back(1u);       // mrole = reader
            ra.push_back(Sm - 1u);  // mpeers
            for (uint32_t s = 1; s < Sm; ++s) {
                auto p = phys(i + s);  // slaves are the next Sm-1 contiguous core indices (mm innermost)
                ra.push_back(p.x);
                ra.push_back(p.y);
            }
        } else {
            // slave: receive from the group's reader (core i - mm).
            ra.push_back(0u);  // mrole = slave
            ra.push_back(1u);  // mpeers
            auto p = phys(i - cp.mm);
            ra.push_back(p.x);
            ra.push_back(p.y);
        }
        SetRuntimeArgs(program, rh, cores[i], ra);

        // writer runtime args.
        auto fwd_next = phys(cp.ring_next_idx);
        auto red_next = phys(cp.red_next_idx);
        auto red_prev = phys(cp.red_prev_idx);
        std::vector<uint32_t> wa = {
            in0_addr,                // 0
            out_addr,                // 1
            cp.m_start,              // 2 first logical M tile (balanced)
            cp.n_start,              // 3 first logical (global) N tile (output addressing)
            cp.k_start,              // 4 first logical K tile (balanced)
            cp.ring_pos,             // 5
            fwd_next.x,              // 6
            fwd_next.y,              // 7
            red_next.x,              // 8
            red_next.y,              // 9
            cp.is_bottom ? 1u : 0u,  // 10
            cp.is_top ? 1u : 0u,     // 11
            red_prev.x,              // 12
            red_prev.y,              // 13
            cp.valid_k,              // 14 valid K tiles (rest of capacity zero)
            cp.valid_m,              // 15 valid M tiles (rest zero / not written)
            cp.valid_n};             // 16 valid N tiles (rest zero / not written)
        // Fused-epilogue / output-split writer args (index 17+). Order MUST match the writer kernel's fidx
        // reads: bias, then residual/gate/broadcast, then chunk count/width/addresses.
        if (has_bias) {
            wa.push_back(tensor_args.bias_tensor->buffer()->address());
        }
        if (has_ternary) {
            wa.push_back(tensor_args.fused_ternary_input_a->buffer()->address());
            wa.push_back(tensor_args.fused_ternary_input_b->buffer()->address());
            wa.push_back(broadcast_gate);
        }
        if (n_chunks > 1u) {
            wa.push_back(n_chunks);
            wa.push_back(out_ntc);
            for (uint32_t c = 1; c < n_chunks; ++c) {
                wa.push_back(tensor_return_value[c].buffer()->address());
            }
        }
        // REDUCE-SCATTER writer args (index 17+; unfused + single-chunk only, so index 17 is free here).
        if (rscatter) {
            const auto rn = phys(cp.rs_next_idx);
            const auto rp = phys(cp.rs_prev_idx);
            wa.push_back(rn.x);             // 17 next core in the Pk cycle (I send to it)
            wa.push_back(rn.y);             // 18
            wa.push_back(rp.x);             // 19 prev core (it sends to me)
            wa.push_back(rp.y);             // 20
            wa.push_back(cp.rs_own_chunk);  // 21 tile-chunk index this core owns + writes
            wa.push_back(Pk);               // 22 cycle size
            wa.push_back(rs_T);             // 23 sub-block tiles (kernel derives the chunk sizes)
        }

        // ---- PHASE 1 fused fabric all-gather args (appended last; only on the MASTER ring) ----
        // Emitted for the 8 cores of ring group p == 0 -- the only fabric clients. Every other core reads
        // the staged shards but never sends, so it needs no mux connection and gets none of these.
        //
        // Layout: [rank, tp, k_shard_tiles, stage_addr, chunk_ready_sem_id, has_fwd, has_bwd]
        //         then, per present direction, the mux client-connection block appended by
        //         append_client_connection_rt_args (opaque to us -- the kernel hands it to FabricMuxV2Sender).
        if (fused_gather.enabled) {
            TT_FATAL(
                wa.size() == fused_rt_base,
                "fused_rt_base ({}) must match the actual writer arg count ({}); the CT arg and the wa push "
                "order have drifted",
                fused_rt_base,
                wa.size());
            // EVERY core gets the common block at the same index (`fused_rt_base`, a compile-time arg), so
            // the kernel can locate it without knowing which optional fusion args preceded it. The first
            // word says whether this core is a fabric client; only master-ring cores get the mux block.
            const bool is_master_ring = (i % preaders_pf) == 0u;
            // Kt must divide evenly by tp, and each shard must be tile-aligned. Two distinct silent
            // corruptions otherwise: the kernel strides the local shard as m * k_shard_tiles, which is
            // only the true row stride when k_local is tile-aligned; and when Kt % tp != 0 the staging
            // columns in [tp*k_shard_tiles, Kt) are never written but ARE read by the matmul.
            TT_FATAL(
                geo.Kt % fused_gather.tp == 0,
                "fused gather needs the global K tile count ({}) to divide by tp ({}); the remainder "
                "columns would be read by the matmul but never staged",
                geo.Kt,
                fused_gather.tp);
            const uint32_t k_local_elems = tensor_args.input_tensor.logical_shape()[-1];
            TT_FATAL(
                k_local_elems % tt::constants::TILE_WIDTH == 0,
                "fused gather needs each K shard tile-aligned, got a {}-element shard (TILE_WIDTH={}); the "
                "kernel would stride the local shard by the wrong row pitch",
                k_local_elems,
                tt::constants::TILE_WIDTH);
            const uint32_t k_shard_tiles = geo.Kt / fused_gather.tp;  // this rank's K tiles
            wa.push_back(is_master_ring ? 1u : 0u);
            wa.push_back(fused_gather.rank);
            wa.push_back(fused_gather.tp);
            wa.push_back(k_shard_tiles);
            wa.push_back(tensor_args.gather_staging_buffer->buffer()->address());
            wa.push_back(fused_gather.chunk_ready_sem_id);
            wa.push_back(fused_gather.mux_cfg_fwd ? 1u : 0u);
            wa.push_back(fused_gather.mux_cfg_bwd ? 1u : 0u);
            wa.push_back(Mt_r);                          // global M tiles (staging row count)
            wa.push_back(geo.Kt);                        // global K tiles (staging row stride)
            wa.push_back(in0.buffer()->address());       // LOCAL shard base (in0_addr now points at staging)
            wa.push_back(i / preaders_pf);               // bank id 0..7 -> which M-slice this core stages
            wa.push_back(fused_gather.fwd_recv_sem_addr);  // incremented by my BACKWARD neighbour
            wa.push_back((is_master_ring && (i == 0u)) ? 1u : 0u);
            wa.push_back(fused_gather.gather_count_sem_id);
            wa.push_back(fused_gather.num_masters);
            wa.push_back(static_cast<uint32_t>(fused_gather.master0_virtual.x));
            wa.push_back(static_cast<uint32_t>(fused_gather.master0_virtual.y));
            wa.push_back(static_cast<uint32_t>(fused_gather.release_list.size()));
            for (const auto& rc : fused_gather.release_list) {
                wa.push_back(static_cast<uint32_t>(rc.x));
                wa.push_back(static_cast<uint32_t>(rc.y));
            }
            TT_FATAL(
                wa.size() == fused_rt_base + kFusedArgCount + 2u * fused_gather.release_list.size(),
                "fused-gather arg block is {} words but FusedGatherArg says {} + 2*{}; the push order and "
                "the offsets used by override_runtime_arguments have drifted",
                wa.size() - fused_rt_base,
                static_cast<uint32_t>(kFusedArgCount),
                fused_gather.release_list.size());
            if (is_master_ring) {
                if (fused_gather.mux_cfg_fwd) {
                    const auto fc = CreateSemaphore(program, CoreRangeSet(CoreRange(cores[i], cores[i])), 0);
                    const auto tc = CreateSemaphore(program, CoreRangeSet(CoreRange(cores[i], cores[i])), 0);
                    fused_gather.mux_cfg_fwd->append_client_connection_rt_args(
                        *fused_gather.mux_virtual_core_fwd,
                        fused_gather.next_channel_fwd++,
                        tt::tt_fabric::FabricMuxV2Config::ClientSemaphores{fc, tc},
                        wa);
                }
                if (fused_gather.mux_cfg_bwd) {
                    const auto fc = CreateSemaphore(program, CoreRangeSet(CoreRange(cores[i], cores[i])), 0);
                    const auto tc = CreateSemaphore(program, CoreRangeSet(CoreRange(cores[i], cores[i])), 0);
                    fused_gather.mux_cfg_bwd->append_client_connection_rt_args(
                        *fused_gather.mux_virtual_core_bwd,
                        fused_gather.next_channel_bwd++,
                        tt::tt_fabric::FabricMuxV2Config::ClientSemaphores{fc, tc},
                        wa);
                }
            }
        }
        SetRuntimeArgs(program, wh, cores[i], wa);

        // compute runtime args: fixed rectangular block over the schedule capacities. N_end spans ALL
        // N_bpc sub-blocks (spec §7); zero-filled tail positions contribute zero. When a fusion is active the
        // reduction-root flag (is_top) follows, then the addcmul scalar bits + gate-broadcast flag.
        std::vector<uint32_t> ca = {0u, geo.M_block_capacity, 0u, geo.N_bpc * geo.N_sub, cp.is_bottom ? 1u : 0u};
        if (rscatter) {
            ca.push_back(cp.rs_pos);  // my position in the Pk cycle
            ca.push_back(Pk);         // cycle size
            ca.push_back(rs_T);       // sub-block tiles (kernel derives the chunk sizes)
        }
        if (has_bias || has_ternary || has_activation) {
            ca.push_back(cp.is_top ? 1u : 0u);
        }
        if (has_ternary) {
            const float sc = *operation_attributes.fused_ternary_scalar;
            ca.push_back(*reinterpret_cast<const uint32_t*>(&sc));
            ca.push_back(broadcast_gate);
        }
        SetRuntimeArgs(program, compute, cores[i], ca);
    }

    return ttnn::device_operation::CachedProgram<shared_variables_t>{
        std::move(program),
        shared_variables_t{
            .num_cores = geo.num_cores,
            .cores = std::move(cores),
            .core_noc = std::move(core_noc),
            .readerA = readerA,
            .readerB = readerB,
            .writerA = writerA,
            .writerB = writerB,
            .compute = compute,
            .has_bias = has_bias,
            .has_ternary = has_ternary,
            .n_chunks = n_chunks,
            .fused_gather = fused_gather.enabled,
            .fused_rt_base = fused_rt_base,
            .preaders = preaders_pf}};
}

AllGatherRegimeAMatmulAsyncProgramFactory::cached_mesh_workload_t
AllGatherRegimeAMatmulAsyncProgramFactory::create_mesh_workload(
    const AllGatherRegimeAMatmulAsyncParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const AllGatherRegimeAMatmulAsyncInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, tensor_return_value);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

void AllGatherRegimeAMatmulAsyncProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const AllGatherRegimeAMatmulAsyncParams& operation_attributes,
    const AllGatherRegimeAMatmulAsyncInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    for (auto& [_range, program] : cached_workload.workload.get_programs()) {
        auto& sv = cached_workload.shared_variables.at(_range);

        // Must track the SAME tensor create_at bound: the staging buffer on the fused path.
        const uint32_t in0_addr = (operation_attributes.tp > 1) ? tensor_args.gather_staging_buffer->buffer()->address()
                                                                : tensor_args.input_tensor.buffer()->address();
        const uint32_t in1_addr = tensor_args.weight_tensor.buffer()->address();
        const uint32_t out_addr = tensor_return_value[0].buffer()->address();

        // Fresh fused-operand / chunk addresses (cache replay with new buffers). Layout mirrors create()'s
        // appended writer args (index 17+): [bias] [residual gate bcast] [n_chunks out_ntc chunk1..].
        const uint32_t bias_addr = sv.has_bias ? tensor_args.bias_tensor->buffer()->address() : 0u;
        const uint32_t ta_addr = sv.has_ternary ? tensor_args.fused_ternary_input_a->buffer()->address() : 0u;
        const uint32_t tb_addr = sv.has_ternary ? tensor_args.fused_ternary_input_b->buffer()->address() : 0u;

        // Some configs place every core on a single NoC group (e.g. preaders==1 => all noc 0), leaving the
        // other group's kernel handles unset. Only fetch runtime-arg maps for groups that actually exist.
        bool has_g0 = false, has_g1 = false;
        for (const auto n : sv.core_noc) {
            (n ? has_g1 : has_g0) = true;
        }
        auto* readerA_args = has_g0 ? &GetRuntimeArgs(program, sv.readerA) : nullptr;
        auto* readerB_args = has_g1 ? &GetRuntimeArgs(program, sv.readerB) : nullptr;
        auto* writerA_args = has_g0 ? &GetRuntimeArgs(program, sv.writerA) : nullptr;
        auto* writerB_args = has_g1 ? &GetRuntimeArgs(program, sv.writerB) : nullptr;

        for (uint32_t i = 0; i < sv.num_cores; ++i) {
            const CoreCoord& core = sv.cores[i];
            const bool b = sv.core_noc[i] != 0u;

            // reader arg 0 = in1_addr.
            auto& ra = (*(b ? readerB_args : readerA_args))[core.x][core.y];
            ra[0] = in1_addr;

            // writer arg 0 = in0_addr, arg 1 = out_addr (chunk 0).
            auto& wa = (*(b ? writerB_args : writerA_args))[core.x][core.y];
            wa[0] = in0_addr;
            wa[1] = out_addr;
            uint32_t fidx = 17u;
            if (sv.has_bias) {
                wa[fidx++] = bias_addr;
            }
            if (sv.has_ternary) {
                wa[fidx++] = ta_addr;
                wa[fidx++] = tb_addr;
                fidx++;  // broadcast_gate flag is shape-derived, unchanged across replays
            }
            if (sv.n_chunks > 1u) {
                fidx += 2u;  // n_chunks, out_ntc (unchanged)
                for (uint32_t c = 1; c < sv.n_chunks; ++c) {
                    wa[fidx++] = tensor_return_value[c].buffer()->address();
                }
            }

            // ---- fused-gather block ----
            // Everything the caller ping-pongs lives here, so it MUST be refreshed on every replay. The
            // staging buffer and the global semaphore rotate between invocations by design (that is what
            // makes repeated CCL safe), which means a cached program that kept the first invocation's
            // addresses would gather into a buffer nobody reads and wait on a credit nobody sends.
            // Offsets mirror the push order in create_at; fused_rt_base is captured there.
            if (sv.fused_gather) {
                const uint32_t g = sv.fused_rt_base;
                wa[g + kFgStageAddr] = tensor_args.gather_staging_buffer->buffer()->address();
                wa[g + kFgShardAddr] = tensor_args.input_tensor.buffer()->address();
                wa[g + kFgFwdRecvSem] = operation_attributes.gather_semaphores[0].address();
            }
        }
    }  // per-coordinate program
}

}  // namespace ttnn::experimental::prim
