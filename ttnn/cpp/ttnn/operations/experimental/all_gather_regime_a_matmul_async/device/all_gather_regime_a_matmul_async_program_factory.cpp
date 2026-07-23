// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Phase A (DRAM-staged) fused all-gather + regime_a_matmul, ONE device's program.
//
// Milestone (this file): D=2 FULL-GATHER-BARRIER correctness (REGIME_A_AGMM_TASK3_BLUEPRINT.md §"Bring-up").
//   1. Allocate a per-device DRAM gather buffer [M, K_global] (output slot 1, a mesh tensor => same address on
//      every device, so the injector can address remote gather slots with this device's TensorAccessor).
//   2. Fabric injector (mux v2, one link toward the forward neighbour): read this device's in0 shard, write it
//      into the LOCAL gather slice, fabric-unicast it into the neighbour's gather buffer at the same global-K
//      offset, then atomic-inc the neighbour's gather_progress GlobalSemaphore once the shard has flushed.
//   3. Full-gather barrier: the injector waits gather_progress >= D-1 (all remote shards landed locally), then
//      fans out a local gather_ready semaphore to every regime_a compute core.
//   4. Replicated regime_a compute engine (DEFAULT placement / bank-order rings — no diagnostics, no fabric
//      hop-distance) reads the fully-populated gather buffer as its in0 and computes [M, N]. Its in0 reader is
//      the COPIED in0_ring_reduce_writer.cpp, gated with AGMM_FULL_GATHER_BARRIER to wait on gather_ready.
//
// Streaming (per-transport readiness, overlap), D>2, and bidirectional ring are Task 3 milestone #18.

#include "all_gather_regime_a_matmul_async_program_factory.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/experimental/device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/experimental/regime_a_matmul/device/regime_a_matmul_config.hpp"
#include "ttnn/operations/experimental/regime_a_matmul/device/regime_a_matmul_plan.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::experimental::prim {

namespace {

namespace ra = ttnn::operations::experimental::regime_a_matmul::plan;

constexpr const char* kIn1ReaderKernel =
    "ttnn/cpp/ttnn/operations/experimental/regime_a_matmul/device/kernels/in1_reader.cpp";
constexpr const char* kComputeKernel =
    "ttnn/cpp/ttnn/operations/experimental/regime_a_matmul/device/kernels/compute.cpp";
// COPY of regime_a's in0 ring/reduce writer, gated with AGMM_FULL_GATHER_BARRIER (gather-buffer in0 + barrier).
constexpr const char* kWriterKernel =
    "ttnn/cpp/ttnn/operations/experimental/all_gather_regime_a_matmul_async/device/kernels/in0_ring_reduce_writer.cpp";
constexpr const char* kInjectorKernel =
    "ttnn/cpp/ttnn/operations/experimental/all_gather_regime_a_matmul_async/device/kernels/dm_in0_injector.cpp";

constexpr uint32_t kTileBytesBf16 = 2048u;
constexpr uint32_t kTileBytesFp32 = 4096u;

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

void mkcb(Program& program, const CoreRangeSet& crs, uint32_t idx, uint32_t ntiles, tt::DataFormat df, uint32_t tsz) {
    CircularBufferConfig c(ntiles * tsz, {{idx, df}});
    c.set_page_size(idx, tsz);
    CreateCircularBuffer(program, crs, c);
}

uint32_t align_up(uint32_t v, uint32_t a) { return ((v + a - 1u) / a) * a; }

}  // namespace

ttnn::device_operation::CachedProgram<AllGatherRegimeAMatmulAsyncProgramFactory::shared_variables_t>
AllGatherRegimeAMatmulAsyncProgramFactory::create_at(
    const AllGatherRegimeAMatmulAsyncParams& op,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const AllGatherRegimeAMatmulAsyncInputs& tensor_args,
    std::vector<Tensor>& output_tensors) {
    Program program = CreateProgram();

    const auto& in0_shard = tensor_args.input_tensor;   // [M, K_local] on THIS device
    const auto& in1 = tensor_args.weight_tensor;         // [K_global, N] (replicated, 8-bank width-shard)
    Tensor& out = output_tensors.at(0);                  // [M, N]
    Tensor& gather = output_tensors.at(1);               // [M, K_global] DRAM interleaved (mesh: same addr all dev)

    auto* mesh_device = in0_shard.device();  // Tensor::device() returns MeshDevice*
    TT_FATAL(mesh_device != nullptr, "all_gather_regime_a_matmul_async requires a MeshDevice");

    const uint32_t D = op.d;
    TT_FATAL(D >= 2, "all_gather_regime_a_matmul_async fused path requires D>=2 (got D={})", D);
    // Milestone #17/#18 (full-gather barrier): the injector unicasts this device's shard to all D-1 other
    // devices through a SINGLE forward mux, reaching the device h hops away for h=1..D-1. For D>2 this only
    // covers every device under a RING (the forward direction wraps around); D=2 also works on a line.
    TT_FATAL(D == 2 || op.topology == ttnn::ccl::Topology::Ring, "all_gather_regime_a_matmul_async D>2 requires ring topology");

    const uint32_t device_index = ttnn::ccl::get_linearized_index_from_physical_coord(
        in0_shard, mesh_coordinate, op.cluster_axis);
    const std::optional<ttnn::MeshCoordinate> forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        in0_shard, mesh_coordinate, 1, op.topology, op.cluster_axis);
    const std::optional<ttnn::MeshCoordinate> backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        in0_shard, mesh_coordinate, -1, op.topology, op.cluster_axis);
    // Mux destination = the +1 forward neighbour (always present on a ring; for a D=2 line's wrap/end device
    // there is no forward neighbour, so fall back to the backward one — still the single "other" device).
    const std::optional<ttnn::MeshCoordinate>& other_coord = forward_coord.has_value() ? forward_coord : backward_coord;
    TT_FATAL(other_coord.has_value(), "fused all-gather requires a neighbour along the cluster axis");

    // ============================ regime_a compute engine (replicated default path) =========================
    // Build the regime_a plan for the FULL matmul [M, K_global] x [K_global, N] using the GATHER buffer's shape
    // as in0 (the planner reads only logical shapes). make_and_build_plan does NOT call get_worker_noc_hop_distance
    // (only the factory's ring/placement diagnostics do — which we deliberately skip here), so it is mesh-safe.
    auto planres = make_and_build_plan(mesh_device, gather, in1, op.regime_a_config);
    TT_FATAL(planres.ok(), "regime_a planner rejected config: {}", planres.error);
    ra::ExecutionPlan& P = *planres.plan;  // mutable: production IN1_NEAR placement + PARETO ring reorder below
    const ra::Geometry& geo = P.geo;
    const ra::CbSizes& cb = P.cb;

    // Resolve the SAME config make_and_build_plan used (auto-select when config=None), so Pk/Ns/Sm/kb match.
    const uint32_t Mt_r = (static_cast<uint32_t>(gather.logical_shape()[-2]) + 31u) / 32u;
    const uint32_t Kt_r = (static_cast<uint32_t>(gather.logical_shape()[-1]) + 31u) / 32u;
    const uint32_t Nt_r = (static_cast<uint32_t>(in1.logical_shape()[-1]) + 31u) / 32u;
    const RegimeAMatmulConfig cfg = op.regime_a_config.value_or(auto_select_config(Mt_r, Kt_r, Nt_r));
    const uint32_t Pk = cfg.k_slices ? cfg.k_slices : 1u;
    const uint32_t Sm = cfg.m_slices ? cfg.m_slices : 1u;
    const uint32_t kb = cfg.k_block_tiles ? cfg.k_block_tiles : 1u;
    const uint32_t use_reduce = (Pk > 1u) ? 1u : 0u;

    // ==== Restore production regime_a placement/ring optimizations (multi-device hop-distance overload) ========
    // The single-chip factory re-places M-split slaves IN1_NEAR and reorders the in0 ring by PARETO before
    // emitting kernels; both materially improve DRAM-BW utilization. Port the DEFAULT (non-diagnostic) paths
    // here so the fused compute engine is placed identically to production and the fused-vs-single-chip gap is
    // comparable. The only change vs regime_a is the hop-distance overload: use the MeshDevice+MeshCoordinate
    // variant (per-chip harvesting-aware) instead of the unit-mesh one (which FATALs on a multi-device mesh).
    namespace expd = tt::tt_metal::experimental::Device;
    const uint32_t preaders = geo.num_cores / 8u;
    auto hop = [&](const CoreCoord& s, const CoreCoord& d, NOC noc) {
        return expd::get_worker_noc_hop_distance(mesh_device, mesh_coordinate, s, d, noc);
    };
    // ---- IN1_NEAR M-slave placement (Sm>1): mm==0 readers near their bank target, slaves at the free worker
    // minimizing the directed reader->slave hop on the group's in1 NoC (mirrors regime_a lines 139-215). ----
    if (Sm > 1u) {
        const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
        const auto opt0 = mesh_device->get_optimal_dram_bank_to_logical_worker_assignment(NOC::NOC_0);
        const auto opt1 = mesh_device->get_optimal_dram_bank_to_logical_worker_assignment(NOC::NOC_1);
        std::set<std::pair<uint32_t, uint32_t>> used;
        auto bank_tgt = [&](uint32_t b, uint32_t noc) { return noc ? opt1[b] : opt0[b]; };
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
        for (uint32_t b = 0; b < 8u; ++b) {
            for (uint32_t p = 0; p < preaders; ++p) {
                const uint32_t i = b * preaders + p;
                if (P.cores[i].mm == 0u) {
                    set_coord(i, find_near(bank_tgt(b, P.cores[i].noc)));
                }
            }
        }
        for (uint32_t b = 0; b < 8u; ++b) {
            for (uint32_t p = 0; p < preaders; ++p) {
                const uint32_t i = b * preaders + p;
                if (P.cores[i].mm == 0u) {
                    continue;
                }
                const uint32_t ri = i - P.cores[i].mm;
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
                        const uint32_t dd = hop(rc, CoreCoord{x, y}, rnoc);
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
    // ---- PARETO in0 ring ordering (mirrors regime_a lines 262-384, opt_pareto only). Per (kk,nn) group of Sm
    // contiguous slices sharing a writer NoC: pick the ring permutation minimizing (worst directed edge, then
    // summed hops) over all Sm mm-rings, subject to summed-hops <= the mm0-optimal budget. ----
    {
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
            const NOC wnoc = (P.cores[base].noc == 0u) ? NOC::NOC_1 : NOC::NOC_0;
            std::vector<std::array<std::array<uint32_t, 8>, 8>> dm(Sm);
            for (uint32_t mm = 0; mm < Sm; ++mm) {
                auto lc = [&](uint32_t b) {
                    const auto& c = P.cores[b * preaders + base + mm].coord;
                    return CoreCoord{c.x, c.y};
                };
                for (uint32_t a = 0; a < 8u; ++a) {
                    for (uint32_t b = 0; b < 8u; ++b) {
                        dm[mm][a][b] = (a == b) ? 0u : hop(lc(a), lc(b), wnoc);
                    }
                }
            }
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
            auto lt2 = [](uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
                return a0 < b0 || (a0 == b0 && a1 < b1);
            };
            auto cand_of = [](const std::array<uint32_t, 7>& t) {
                std::array<uint32_t, 8> c{};
                c[0] = 0;
                for (uint32_t i = 0; i < 7u; ++i) {
                    c[i + 1u] = t[i];
                }
                return c;
            };
            const std::array<uint32_t, 8> bank = {0, 1, 2, 3, 4, 5, 6, 7};
            std::array<uint32_t, 8> opt_mm0 = bank, opt_pareto = bank;
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
            opt_pareto = opt_mm0;
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

    // Core range sets: all compute cores + split-NoC groups (g0 = noc 0, g1 = noc 1).
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

    // Circular buffers (regime_a spec §5), on all compute cores.
    mkcb(program, all_cores, 0, cb.cb0_tiles, tt::DataFormat::Float16_b, kTileBytesBf16);
    mkcb(program, all_cores, 1, cb.cb1_tiles, tt::DataFormat::Float16_b, kTileBytesBf16);
    mkcb(program, all_cores, 2, cb.cb2_tiles, tt::DataFormat::Float16_b, kTileBytesBf16);
    mkcb(program, all_cores, 3, cb.cb3_tiles, tt::DataFormat::Float32, kTileBytesFp32);
    if (cb.cb7_tiles > 0u) {
        mkcb(program, all_cores, 7, cb.cb7_tiles, tt::DataFormat::Float16_b, kTileBytesBf16);
    }

    // Semaphores. regime_a's ring/reduce sems + the full-gather-barrier fan-out sem (injector -> readers).
    const uint32_t fwd_sem = CreateSemaphore(program, all_cores, 0u);
    const uint32_t red_sem = CreateSemaphore(program, all_cores, 0u);
    const uint32_t redfree_sem = CreateSemaphore(program, all_cores, 0u);
    uint32_t in1valid_sem = 0u, in1ready_sem = 0u;
    if (Sm > 1u) {
        in1valid_sem = CreateSemaphore(program, all_cores, 0u);
        in1ready_sem = CreateSemaphore(program, all_cores, 0u);
    }
    // gather_ready lives on the compute cores AND the injector core (union), so the injector can address it.

    // ---- in1 reader compile args (in1_reader.cpp order) ----
    std::vector<uint32_t> rct = {
        kb, geo.N_sub, geo.W, geo.G, kTileBytesBf16, geo.N_bpc, geo.in1_shard_stride_n, in1valid_sem, in1ready_sem};

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

    // ---- writer (COPIED gated in0 ring/reduce writer) compile args ----
    // Base layout identical to regime_a; the AGMM gate appends (gather_ready_addr, Kt_local, D) at the end.
    // Default = AGMM_STREAM (per-shard progressive gate → matmul overlaps the gather). The full-gather-before-
    // matmul diagnostic (reader waits for all shards) is selected by the hashed op attribute (from
    // TT_AGMM_FULL_GATHER at invoke time), so the two reader variants never alias in the program cache.
    const bool full_gather_diag = op.full_gather_diagnostic;
    std::map<std::string, std::string> wdefs;
    wdefs[full_gather_diag ? "AGMM_FULL_GATHER_BARRIER" : "AGMM_STREAM"] = "1";
    std::vector<uint32_t> wct = {
        geo.M_block_capacity,   // 0
        kb,                     // 1
        geo.N_sub,              // 2
        geo.K_num_blocks_eff,   // 3
        kTileBytesBf16,         // 4
        geo.in0_stride_k,       // 5  (== Kt_global; regime_a reads in0 with the full-K row stride)
        geo.out_stride_n,       // 6
        geo.W,                  // 7
        geo.G,                  // 8
        fwd_sem,                // 9
        red_sem,                // 10
        geo.N_bpc,              // 11
        redfree_sem,            // 12
        use_reduce};            // 13
    TensorAccessorArgs(*gather.buffer()).append_to(wct);  // in0 accessor -> GATHER buffer
    TensorAccessorArgs(*out.buffer()).append_to(wct);     // out accessor

    KernelHandle readerA = mk(kIn1ReaderKernel, g0, DataMovementProcessor::RISCV_0, NOC::RISCV_0_default, rct, {});
    KernelHandle readerB = mk(kIn1ReaderKernel, g1, DataMovementProcessor::RISCV_1, NOC::RISCV_1_default, rct, {});
    KernelHandle writerA = mk(kWriterKernel, g0, DataMovementProcessor::RISCV_1, NOC::RISCV_1_default, wct, wdefs);
    KernelHandle writerB = mk(kWriterKernel, g1, DataMovementProcessor::RISCV_0, NOC::RISCV_0_default, wct, wdefs);

    // ---- compute compile args ----
    const uint32_t sbh = largest_div(geo.M_block_capacity, 2u);
    const uint32_t sbw = largest_div(geo.N_sub, 4u / sbh);
    std::vector<uint32_t> cct = {
        geo.K_num_blocks_eff, geo.M_block_capacity, kb, geo.N_sub, 1u, geo.N_bpc, sbh, sbw};
    std::map<std::string, std::string> cdefs = {{"REDUCE_K", "1"}, {"IN0_KSLICE_RESIDENT", "1"}};
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

    // ============================ fabric injector (mux v2, forward neighbour) ==============================
    // Reserve one injector worker core + one mux core from the grid, avoiding the regime_a compute cores.
    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    std::set<std::pair<uint32_t, uint32_t>> used;
    for (const auto& c : cores) {
        used.insert({c.x, c.y});
    }
    auto take_free_core = [&]() -> CoreCoord {
        for (int y = (int)grid.y - 1; y >= 0; --y) {
            for (int x = (int)grid.x - 1; x >= 0; --x) {
                if (!used.count({(uint32_t)x, (uint32_t)y})) {
                    used.insert({(uint32_t)x, (uint32_t)y});
                    return CoreCoord{(uint32_t)x, (uint32_t)y};
                }
            }
        }
        TT_THROW("all_gather_regime_a_matmul_async: no free core for fabric mux/injector");
    };
    const CoreCoord mux_logical = take_free_core();
    const CoreCoord inj_logical = take_free_core();
    const CoreRangeSet inj_crs(CoreRange(inj_logical, inj_logical));

    // Coordination uses D+1 GlobalSemaphores (multi_device_global_semaphore[0]=gather_ready, [1+e]=shard_landed
    // for device e) rather than program-local CreateSemaphores: GlobalSemaphores can be reset host-side between
    // launches, so the streaming barrier is correct on program-cache replay (a program-local semaphore keeps its
    // accumulated value across launches and defeats the barrier). All live on the CCL sub-device CRS (all cores),
    // so their address is valid on every compute core AND the injector core.
    TT_FATAL(
        op.multi_device_global_semaphore.size() >= D + 1,
        "all_gather_regime_a_matmul_async needs >= D+1 global semaphores (gather_ready + D shard_landed), got {} "
        "for D={}",
        op.multi_device_global_semaphore.size(),
        D);
    const uint32_t gather_ready_addr = op.multi_device_global_semaphore.at(0).address();

    // Injector L1 scratch: payload tile CB (c_0) + packet-header CB (c_1).
    const uint32_t aligned_hdr = align_up((uint32_t)tt::tt_fabric::get_tt_fabric_packet_header_size_bytes(),
                                          (uint32_t)hal::get_l1_alignment());
    mkcb(program, inj_crs, 0, 1, tt::DataFormat::Float16_b, kTileBytesBf16);  // payload (one tile)
    {
        CircularBufferConfig hc(aligned_hdr, {{1, tt::DataFormat::Float16_b}});
        hc.set_page_size(1, aligned_hdr);
        CreateCircularBuffer(program, inj_crs, hc);
    }

    // mux v2 toward the forward neighbour.
    const auto src_node = mesh_device->get_fabric_node_id(mesh_coordinate);
    const auto dst_node = mesh_device->get_fabric_node_id(other_coord.value());
    const auto link_indices = tt::tt_fabric::get_forwarding_link_indices(src_node, dst_node);
    TT_FATAL(!link_indices.empty(), "no fabric forwarding link from src to forward neighbour");
    const uint32_t channel_buf_bytes = align_up(aligned_hdr + kTileBytesBf16, (uint32_t)hal::get_l1_alignment());
    tt::tt_fabric::FabricMuxV2Config mux_config(
        /*num_channels=*/1,
        /*num_buffers_per_channel=*/(uint8_t)op.num_buffers_per_channel,
        /*channel_buffer_size_bytes=*/channel_buf_bytes,
        /*base_l1_address=*/mesh_device->allocator()->get_base_allocator_addr(HalMemType::L1));
    tt::tt_fabric::add_fabric_mux_v2_to_program(
        program, mux_config, mux_logical, src_node, dst_node, link_indices.front(), NOC::RISCV_0_default);
    const CoreCoord mux_virtual = mesh_device->worker_core_from_logical_core(mux_logical);

    // The injector reaches all D-1 other devices by unicasting forward with num_hops = 1..D-1 (ring wrap).
    const uint32_t num_dests = D - 1u;

    // Injector compile args: TensorAccessorArgs(in0 shard) then TensorAccessorArgs(gather buffer).
    std::vector<uint32_t> ict;
    TensorAccessorArgs(*in0_shard.buffer()).append_to(ict);
    TensorAccessorArgs(*gather.buffer()).append_to(ict);
    KernelHandle injector = CreateKernel(
        program,
        kInjectorKernel,
        inj_crs,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = ict, .defines = {}});

    // ============================ runtime args ============================
    const uint32_t in1_addr = in1.buffer()->address();
    const uint32_t out_addr = out.buffer()->address();
    const uint32_t gather_addr = gather.buffer()->address();

    auto phys = [&](uint32_t core_idx) {
        const auto& c = P.cores[core_idx].coord;
        return mesh_device->worker_core_from_logical_core(CoreCoord{c.x, c.y});
    };

    for (uint32_t i = 0; i < geo.num_cores; ++i) {
        const ra::CorePlan& cp = P.cores[i];
        const KernelHandle rh = cp.noc ? readerB : readerA;
        const KernelHandle wh = cp.noc ? writerB : writerA;

        std::vector<uint32_t> ra_rt = {in1_addr, cp.bank, cp.ring_pos, cp.k_start, cp.n_local, cp.valid_k, cp.valid_n};
        if (Sm == 1u) {
            ra_rt.push_back(2u);
            ra_rt.push_back(0u);
        } else if (cp.mm == 0u) {
            ra_rt.push_back(1u);
            ra_rt.push_back(Sm - 1u);
            for (uint32_t s = 1; s < Sm; ++s) {
                auto p = phys(i + s);
                ra_rt.push_back(p.x);
                ra_rt.push_back(p.y);
            }
        } else {
            ra_rt.push_back(0u);
            ra_rt.push_back(1u);
            auto p = phys(i - cp.mm);
            ra_rt.push_back(p.x);
            ra_rt.push_back(p.y);
        }
        SetRuntimeArgs(program, rh, cores[i], ra_rt);

        auto fwd_next = phys(cp.ring_next_idx);
        auto red_next = phys(cp.red_next_idx);
        auto red_prev = phys(cp.red_prev_idx);
        std::vector<uint32_t> wa = {
            gather_addr,             // 0  in0_addr == gather buffer base
            out_addr,                // 1
            cp.m_start,              // 2
            cp.n_start,              // 3
            cp.k_start,              // 4
            cp.ring_pos,             // 5
            fwd_next.x,              // 6
            fwd_next.y,              // 7
            red_next.x,              // 8
            red_next.y,              // 9
            cp.is_bottom ? 1u : 0u,  // 10
            cp.is_top ? 1u : 0u,     // 11
            red_prev.x,              // 12
            red_prev.y,              // 13
            cp.valid_k,              // 14
            cp.valid_m,              // 15
            cp.valid_n,              // 16
            gather_ready_addr,       // 17 gather_ready GlobalSemaphore L1 address
            geo.Kt / D,              // 18 Kt_local (K tiles per device shard; shard of global tile = kg/Kt_local)
            D};                      // 19 device count (streaming: gate kg on gather_ready>kg/Kt_local; full: ==D)
        SetRuntimeArgs(program, wh, cores[i], wa);

        std::vector<uint32_t> ca = {0u, geo.M_block_capacity, 0u, geo.N_bpc * geo.N_sub, cp.is_bottom ? 1u : 0u};
        SetRuntimeArgs(program, compute, cores[i], ca);
    }

    // ---- injector runtime args ----
    // in0 shard has K_local tiles; this device owns global K-tile offset device_index * Kt_local.
    const uint32_t Kt_global = geo.Kt;
    const uint32_t Kt_local = Kt_global / D;
    const uint32_t k_tile_global_base = device_index * Kt_local;
    // injector core (symmetric across devices): where shard_landed + gather_ready live and where remote
    // shard_landed atomic-incs are targeted.
    const CoreCoord inj_virtual = mesh_device->worker_core_from_logical_core(inj_logical);

    std::vector<uint32_t> inj_rt = {
        in0_shard.buffer()->address(),  // a0 in0 shard base
        gather_addr,                    // a1 local gather base
        geo.Mt,                         // a2 Mt
        Kt_local,                       // a3 Kt_local
        Kt_global,                      // a4 Kt_global (gather row stride)
        k_tile_global_base,             // a5 global K-tile base of this shard
        kTileBytesBf16,                 // a6 tile bytes
        num_dests,                      // a7 number of remote devices to reach (hops 1..num_dests)
        D,                              // a8 device count
        device_index,                  // a9 this device's shard index
        (uint32_t)inj_virtual.x,        // a10 injector core x
        (uint32_t)inj_virtual.y,        // a11 injector core y
        gather_ready_addr,              // a12 gather_ready GlobalSemaphore L1 address (monotonic fan-out target)
        geo.num_cores};                 // a13 number of compute cores to fan out to
    // a14..a14+D-1: shard_landed[0..D-1] GlobalSemaphore L1 addresses.
    for (uint32_t s = 0; s < D; ++s) {
        inj_rt.push_back(op.multi_device_global_semaphore.at(1u + s).address());
    }
    // followed by each compute core's (x,y) for the gather_ready fan-out.
    for (uint32_t i = 0; i < geo.num_cores; ++i) {
        auto p = phys(i);
        inj_rt.push_back(p.x);
        inj_rt.push_back(p.y);
    }
    // then the mux v2 client connection args (packet-header/payload come from CBs c_1/c_0 in-kernel).
    const uint32_t flow_control_sem = CreateSemaphore(program, inj_logical, 0u);
    const uint32_t teardown_sem = CreateSemaphore(program, inj_logical, 0u);
    mux_config.append_client_connection_rt_args(
        mux_virtual,
        /*logical_channel_id=*/0,
        tt::tt_fabric::FabricMuxV2Config::ClientSemaphores{flow_control_sem, teardown_sem},
        inj_rt);
    SetRuntimeArgs(program, injector, inj_logical, inj_rt);

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
            .injector_cores = {inj_logical},
            .injector = injector}};
}

void AllGatherRegimeAMatmulAsyncProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const AllGatherRegimeAMatmulAsyncParams& /*operation_attributes*/,
    const AllGatherRegimeAMatmulAsyncInputs& tensor_args,
    std::vector<Tensor>& output_tensors) {
    const uint32_t in1_addr = tensor_args.weight_tensor.buffer()->address();
    const uint32_t out_addr = output_tensors.at(0).buffer()->address();
    const uint32_t gather_addr = output_tensors.at(1).buffer()->address();
    const uint32_t in0_addr = tensor_args.input_tensor.buffer()->address();

    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        auto& sv = cached_workload.shared_variables.at(range);
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
            (*(b ? readerB_args : readerA_args))[core.x][core.y][0] = in1_addr;
            auto& wa = (*(b ? writerB_args : writerA_args))[core.x][core.y];
            wa[0] = gather_addr;
            wa[1] = out_addr;
        }
        // injector: a0 in0 shard, a1 local gather base.
        auto& inj_args = GetRuntimeArgs(program, sv.injector);
        const CoreCoord& ic = sv.injector_cores.at(0);
        inj_args[ic.x][ic.y][0] = in0_addr;
        inj_args[ic.x][ic.y][1] = gather_addr;
    }
}

}  // namespace ttnn::experimental::prim
