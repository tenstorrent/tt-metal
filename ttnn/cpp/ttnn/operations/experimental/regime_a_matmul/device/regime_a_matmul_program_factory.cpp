// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "regime_a_matmul_program_factory.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <fmt/base.h>  // RINGCOST diagnostic line (physical ring-order experiment)

#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/experimental/device.hpp>  // get_worker_noc_hop_distance (physical ring-order diag)

#include "regime_a_matmul_config.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::experimental::prim {

namespace {

constexpr const char* kIn1ReaderKernel =
    "ttnn/cpp/ttnn/operations/experimental/regime_a_matmul/device/kernels/in1_reader.cpp";
constexpr const char* kWriterKernel =
    "ttnn/cpp/ttnn/operations/experimental/regime_a_matmul/device/kernels/in0_ring_reduce_writer.cpp";
constexpr const char* kComputeKernel =
    "ttnn/cpp/ttnn/operations/experimental/regime_a_matmul/device/kernels/compute.cpp";

constexpr uint32_t kTileBytesBf16 = 2048u;
constexpr uint32_t kTileBytesFp32 = 4096u;

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

}  // namespace

RegimeAMatmulProgramFactory::cached_program_t RegimeAMatmulProgramFactory::create(
    const RegimeAMatmulParams& operation_attributes,
    const RegimeAMatmulInputs& tensor_args,
    Tensor& tensor_return_value) {
    Program program = CreateProgram();

    const auto& in0 = tensor_args.input_tensor;
    const auto& in1 = tensor_args.weight_tensor;
    Tensor& out = tensor_return_value;
    IDevice* device = in0.device();

    // Resolve config=None via the auto-selector (deterministic in the tile dims, program-cache-safe).
    const uint32_t Mt_r = (static_cast<uint32_t>(in0.logical_shape()[-2]) + 31u) / 32u;
    const uint32_t Kt_r = (static_cast<uint32_t>(in0.logical_shape()[-1]) + 31u) / 32u;
    const uint32_t Nt_r = (static_cast<uint32_t>(in1.logical_shape()[-1]) + 31u) / 32u;
    const RegimeAMatmulConfig cfg = operation_attributes.config.value_or(auto_select_config(Mt_r, Kt_r, Nt_r));

    // ---- Run the pure host planner ----
    auto planres = make_and_build_plan(device, in0, in1, cfg);
    TT_FATAL(planres.ok(), "regime_a_matmul planner rejected config: {}", planres.error);
    plan::ExecutionPlan& P = *planres.plan;  // mutable: the ring-order diag overrides ring_pos/next/prev below
    const plan::Geometry& geo = P.geo;
    const plan::CbSizes& cb = P.cb;

    const uint32_t Pk = cfg.k_slices ? cfg.k_slices : 1u;
    const uint32_t Sm = cfg.m_slices ? cfg.m_slices : 1u;
    const uint32_t kb = cfg.k_block_tiles ? cfg.k_block_tiles : 1u;
    const uint32_t use_reduce = (Pk > 1u) ? 1u : 0u;

    // Test-only diagnostic ablations: mask 0 (public path) => all three define maps are EMPTY, so the
    // compile is byte-identical to production. Each DIAG_* define is scoped to the kernel(s) that #ifdef it.
    const uint32_t diag = operation_attributes.diag_mask;
    std::map<std::string, std::string> rdefs;  // in1 reader
    std::map<std::string, std::string> wdefs;  // in0 ring/reduce writer
    std::map<std::string, std::string> ddefs;  // compute (added to cdefs below)
    if (diag & RegimeADiag::DIAG_SKIP_IN1_READ) {
        rdefs["DIAG_SKIP_IN1_READ"] = "1";
    }
    if (diag & RegimeADiag::DIAG_SKIP_IN0_READ) {
        wdefs["DIAG_SKIP_IN0_READ"] = "1";
    }
    if (diag & RegimeADiag::DIAG_SKIP_IN0_FORWARD) {
        wdefs["DIAG_SKIP_IN0_FORWARD"] = "1";
    }
    if (diag & RegimeADiag::DIAG_NO_REDUCE) {
        wdefs["DIAG_NO_REDUCE"] = "1";
        ddefs["DIAG_NO_REDUCE"] = "1";
    }
    if (diag & RegimeADiag::DIAG_BARRIER_DRAIN) {
        wdefs["DIAG_BARRIER_DRAIN"] = "1";  // A/B baseline: OLD per-block phase-2 completion barrier
    }

    // ---- Physical-topology-aware in0 ring ordering (test-only; DEFAULT = bank order [0..7]) ----
    // Overrides ring_pos/ring_next_idx/ring_prev_idx per ring group using the group's WRITER NoC authoritative
    // hop distance (get_worker_noc_hop_distance, logical->physical + directed torus routing w/ wraparound).
    // Placement/work/reduction are unchanged; only the ring visiting order (which core seeds which in0 shard,
    // the forward route, and the in1 rotated read) changes — correct for ANY permutation. Emits a RINGCOST
    // line per group (bank/greedy/opt max+total edge cost) for the report. No effect on the public path.
    const bool ring_bank = (diag & RegimeADiag::DIAG_RING_BANK) != 0u;        // diagnostic: bank order [0..7]
    const bool ring_greedy = (diag & RegimeADiag::DIAG_RING_GREEDY) != 0u;    // diagnostic: greedy (mm==0)
    const bool ring_opt_mm0 = (diag & RegimeADiag::DIAG_RING_OPT_MM0) != 0u;  // diagnostic: opt scored on mm==0 only
    if (!ring_bank) {  // DEFAULT = opt scored across ALL Sm mm-rings; greedy / mm0-opt if selected
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
        // M-split (Sm>1): slices differing only in mm form a (kk,nn) group of Sm CONTIGUOUS slice indices
        // [base, base+Sm), all sharing the same writer NoC. Their in1 slaves receive in1 in the mm==0 READER's
        // shard order while their in0 rings are separate physical cores, so the WHOLE group MUST use the SAME
        // permutation (reader/slave ring_pos must agree per bank) or the in0/in1 pairing corrupts.
        // OBJECTIVE (default): lexicographic over the Sm physical mm-rings — (1) minimize the worst directed
        // edge across ALL rings, (2) then the summed hops across ALL edges AND rings. This accounts for the
        // slaves' routes, not just the reader's. DIAG_RING_OPT_MM0 reverts to scoring only the mm==0 ring.
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
            // aggregate route cost of a candidate across all Sm mm-rings: (worst edge over rings, summed hops).
            auto agg_cost = [&](const std::array<uint32_t, 8>& ord) -> std::pair<uint32_t, uint32_t> {
                uint32_t amax = 0, atot = 0;
                for (uint32_t mm = 0; mm < Sm; ++mm) {
                    const auto [m, t] = ring_cost(ord, dm[mm]);
                    amax = std::max(amax, m);
                    atot += t;
                }
                return {amax, atot};
            };

            const std::array<uint32_t, 8> bank = {0, 1, 2, 3, 4, 5, 6, 7};
            // greedy nearest-neighbour on the mm==0 ring (heuristic diagnostic), start at bank 0.
            std::array<uint32_t, 8> greedy{};
            {
                std::array<bool, 8> vis{};
                greedy[0] = 0;
                vis[0] = true;
                for (uint32_t pos = 1; pos < 8u; ++pos) {
                    uint32_t best = 0, bestd = 0xffffffffu;
                    for (uint32_t c = 0; c < 8u; ++c) {
                        if (!vis[c] && dm[0][greedy[pos - 1]][c] < bestd) {
                            bestd = dm[0][greedy[pos - 1]][c];
                            best = c;
                        }
                    }
                    greedy[pos] = best;
                    vis[best] = true;
                }
            }
            // exhaustive: fix bank 0 at pos 0, permute the other 7 (5040 cycles; directed => both orientations).
            // Track BOTH objectives in one pass: opt_mm0 = min(max,total) on the mm==0 ring; opt_agg =
            // min(worst-edge-over-rings, summed-hops-over-rings).
            std::array<uint32_t, 8> opt_mm0 = bank, opt_agg = bank;
            {
                std::array<uint32_t, 7> tail = {1, 2, 3, 4, 5, 6, 7};
                uint32_t bm0 = 0xffffffffu, bt0 = 0xffffffffu, bma = 0xffffffffu, bta = 0xffffffffu;
                do {
                    std::array<uint32_t, 8> cand{};
                    cand[0] = 0;
                    for (uint32_t t = 0; t < 7u; ++t) {
                        cand[t + 1u] = tail[t];
                    }
                    const auto [m0, t0] = ring_cost(cand, dm[0]);
                    if (m0 < bm0 || (m0 == bm0 && t0 < bt0)) {
                        bm0 = m0;
                        bt0 = t0;
                        opt_mm0 = cand;
                    }
                    const auto [ma, ta] = agg_cost(cand);
                    if (ma < bma || (ma == bma && ta < bta)) {
                        bma = ma;
                        bta = ta;
                        opt_agg = cand;
                    }
                } while (std::next_permutation(tail.begin(), tail.end()));
            }
            const std::array<uint32_t, 8>& sel = ring_greedy ? greedy : (ring_opt_mm0 ? opt_mm0 : opt_agg);
            // apply the selected order to ALL Sm slices of this group (same permutation => reader/slave ring_pos
            // agree per bank, preserving in0/in1 pairing under M-split).
            for (uint32_t mm = 0; mm < Sm; ++mm) {
                const uint32_t jj = base + mm;
                for (uint32_t pos = 0; pos < 8u; ++pos) {
                    const uint32_t ci = sel[pos] * preaders + jj;
                    P.cores[ci].ring_pos = pos;
                    P.cores[ci].ring_next_idx = sel[(pos + 1u) % 8u] * preaders + jj;
                    P.cores[ci].ring_prev_idx = sel[(pos + 7u) % 8u] * preaders + jj;
                }
            }
            // RINGCOST diagnostic (route costs, gated behind TT_MM_RINGCOST so the production path is silent on
            // compile). Reports each candidate's GROUP-AGGREGATE cost (worst edge over the Sm rings; summed hops
            // over all rings) PLUS the selected order's PER-RING (max,total) breakdown so the report can
            // distinguish per-ring from aggregated-across-rings costs. The harness further aggregates across
            // the (kk,nn) groups of the whole op.
            if (std::getenv("TT_MM_RINGCOST") != nullptr) {
                auto join = [](const std::array<uint32_t, 8>& o) {
                    std::string s;
                    for (uint32_t p = 0; p < 8u; ++p) {
                        s += std::to_string(o[p]) + (p + 1u < 8u ? "," : "");
                    }
                    return s;
                };
                const auto ab = agg_cost(bank), ag = agg_cost(greedy), a0 = agg_cost(opt_mm0), aa = agg_cost(opt_agg);
                std::string perring;
                for (uint32_t mm = 0; mm < Sm; ++mm) {
                    const auto [m, t] = ring_cost(sel, dm[mm]);
                    perring += "(" + std::to_string(m) + ":" + std::to_string(t) + ")";
                }
                fmt::print(
                    "RINGCOST group={} Sm={} wnoc={} sel={} bank[{}]aggmax={}aggtot={} greedy[{}]aggmax={}aggtot={} "
                    "mm0[{}]aggmax={}aggtot={} agg[{}]aggmax={}aggtot={} sel_perring={}\n",
                    base,
                    Sm,
                    (wnoc == NOC::NOC_0 ? 0 : 1),
                    (ring_greedy ? "greedy" : (ring_opt_mm0 ? "mm0" : "agg")),
                    join(bank),
                    ab.first,
                    ab.second,
                    join(greedy),
                    ag.first,
                    ag.second,
                    join(opt_mm0),
                    a0.first,
                    a0.second,
                    join(opt_agg),
                    aa.first,
                    aa.second,
                    perring);
            }
        }
    }
    if (diag & RegimeADiag::DIAG_LOCAL_FEED) {
        rdefs["DIAG_LOCAL_FEED"] = "1";
        wdefs["DIAG_LOCAL_FEED"] = "1";
        ddefs["DIAG_LOCAL_FEED"] = "1";
    }
    if (diag & RegimeADiag::DIAG_FULL_IN0_WAIT) {
        ddefs["DIAG_FULL_IN0_WAIT"] = "1";  // A/B baseline: old full-slice startup barrier (compute-only)
    }
    const bool scatter = (diag & RegimeADiag::DIAG_IN0_SCATTER) != 0u;
    if (scatter) {
        wdefs["DIAG_IN0_SCATTER"] = "1";  // writer phase-1 uses direct scatter; needs the G-1 ahead peers (below)
    }
    // Replicated shorter ring: IN0_REPL (2 or 4) goes to BOTH the writer (seed reads + R-bundle rotation) and
    // the in1 reader (matching shard order). No runtime-arg change (nearest-neighbor forward reuses fwd_next).
    if (diag & RegimeADiag::DIAG_IN0_REPL2) {
        wdefs["IN0_REPL"] = "2";
        rdefs["IN0_REPL"] = "2";
    } else if (diag & RegimeADiag::DIAG_IN0_REPL4) {
        wdefs["IN0_REPL"] = "4";
        rdefs["IN0_REPL"] = "4";
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

    // ---- Circular buffers (spec §5) on all cores ----
    mkcb(program, all_cores, 0, cb.cb0_tiles, tt::DataFormat::Float16_b, kTileBytesBf16);  // in0 k-slice resident
    mkcb(program, all_cores, 1, cb.cb1_tiles, tt::DataFormat::Float16_b, kTileBytesBf16);  // in1
    mkcb(program, all_cores, 2, cb.cb2_tiles, tt::DataFormat::Float16_b, kTileBytesBf16);  // out
    mkcb(program, all_cores, 3, cb.cb3_tiles, tt::DataFormat::Float32, kTileBytesFp32);    // fp32 intermediate
    if (cb.cb7_tiles > 0u) {
        mkcb(program, all_cores, 7, cb.cb7_tiles, tt::DataFormat::Float16_b, kTileBytesBf16);  // reduce (Pk>1 only)
    }

    // ---- Semaphores ----
    const uint32_t fwd_sem = CreateSemaphore(program, all_cores, 0u);      // in0 ring recv
    const uint32_t red_sem = CreateSemaphore(program, all_cores, 0u);      // reduction recv
    const uint32_t redfree_sem = CreateSemaphore(program, all_cores, 0u);  // cb_reduce reverse credit
    uint32_t in1valid_sem = 0u, in1ready_sem = 0u;                         // M-split reader<->slaves
    if (Sm > 1u) {
        in1valid_sem = CreateSemaphore(program, all_cores, 0u);
        in1ready_sem = CreateSemaphore(program, all_cores, 0u);
    }
    // DIAG_IN0_XCHG: G-1 per-slot readiness semaphores so the writer pushes each in0 slot the moment its
    // direct-exchange write lands (incremental overlap), rather than one ring counter. Created only for the
    // xchg program so the public path's semaphore layout is unchanged.
    const bool xchg = (diag & RegimeADiag::DIAG_IN0_XCHG) != 0u;
    const bool xchgrr = (diag & RegimeADiag::DIAG_IN0_XCHGRR) != 0u;
    std::vector<uint32_t> xchg_slotsem;
    if (xchg || xchgrr) {  // both direct-exchange schedules use G-1 per-slot readiness semaphores
        for (uint32_t d = 1; d < geo.G; ++d) {
            xchg_slotsem.push_back(CreateSemaphore(program, all_cores, 0u));
        }
        wdefs[xchg ? "DIAG_IN0_XCHG" : "DIAG_IN0_XCHGRR"] = "1";
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
        in1ready_sem,            // 8
        0u};                     // 9 in1_mcast (0 for v1: unicast forward)

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
        use_reduce};           // 13
    TensorAccessorArgs(*in0.buffer()).append_to(wct);
    TensorAccessorArgs(*out.buffer()).append_to(wct);

    // Split-NOC: reader on the core's in1 NoC, writer on the OTHER NoC.
    //   g0 (noc==0): reader RISCV_0/NOC0, writer RISCV_1/NOC1
    //   g1 (noc==1): reader RISCV_1/NOC1, writer RISCV_0/NOC0
    KernelHandle readerA = mk(kIn1ReaderKernel, g0, DataMovementProcessor::RISCV_0, NOC::RISCV_0_default, rct, rdefs);
    KernelHandle readerB = mk(kIn1ReaderKernel, g1, DataMovementProcessor::RISCV_1, NOC::RISCV_1_default, rct, rdefs);
    KernelHandle writerA = mk(kWriterKernel, g0, DataMovementProcessor::RISCV_1, NOC::RISCV_1_default, wct, wdefs);
    KernelHandle writerB = mk(kWriterKernel, g1, DataMovementProcessor::RISCV_0, NOC::RISCV_0_default, wct, wdefs);

    // compute (spec §6c). fp32 DST limit: subblock_h * subblock_w <= 4.
    const uint32_t sbh = largest_div(geo.M_block_capacity, 2u);
    const uint32_t sbw = largest_div(geo.N_sub, 4u / sbh);
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
    cdefs.insert(ddefs.begin(), ddefs.end());  // test-only diagnostic defines (empty for mask 0)
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
    const uint32_t in0_addr = in0.buffer()->address();
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
        // DIAG_IN0_SCATTER (test-only variant): append the G-1 ring peers AHEAD of this core (args 17..),
        // in d-ahead order (ring_next^1..ring_next^{G-1}). Core scatters its own shard to peer d's cb0 slot
        // d; each peer receives from the core d-behind, reproducing the ring's cb0 layout. Appended only for
        // the scatter program (distinct hash), so the public path's arg layout is unchanged.
        if (scatter || xchg || xchgrr) {
            // XCHG/XCHGRR: prepend the G-1 per-slot semaphore IDs (args 17..17+G-2). Then (scatter/xchg/
            // xchgrr) the G-1 ahead peers in d-ahead order (ring_next^1..). Core writes its own shard to peer
            // d's slot d and signals that peer's slot-d sem. Appended only for these programs; public-path
            // arg layout unchanged.
            if (xchg || xchgrr) {
                for (uint32_t d = 1; d < geo.G; ++d) {
                    wa.push_back(xchg_slotsem[d - 1]);
                }
            }
            uint32_t cur = i;
            for (uint32_t d = 1; d < geo.G; ++d) {
                cur = P.cores[cur].ring_next_idx;
                auto pc = phys(cur);
                wa.push_back(pc.x);
                wa.push_back(pc.y);
            }
        }
        SetRuntimeArgs(program, wh, cores[i], wa);

        // compute runtime args: fixed rectangular block over the schedule capacities. N_end spans ALL
        // N_bpc sub-blocks (spec §7); zero-filled tail positions contribute zero.
        SetRuntimeArgs(
            program, compute, cores[i], {0u, geo.M_block_capacity, 0u, geo.N_bpc * geo.N_sub, cp.is_bottom ? 1u : 0u});
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .num_cores = geo.num_cores,
            .cores = std::move(cores),
            .core_noc = std::move(core_noc),
            .readerA = readerA,
            .readerB = readerB,
            .writerA = writerA,
            .writerB = writerB,
            .compute = compute}};
}

void RegimeAMatmulProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const RegimeAMatmulParams& /*operation_attributes*/,
    const RegimeAMatmulInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto& program = cached_program.program;
    auto& sv = cached_program.shared_variables;

    const uint32_t in0_addr = tensor_args.input_tensor.buffer()->address();
    const uint32_t in1_addr = tensor_args.weight_tensor.buffer()->address();
    const uint32_t out_addr = tensor_return_value.buffer()->address();

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

        // writer arg 0 = in0_addr, arg 1 = out_addr.
        auto& wa = (*(b ? writerB_args : writerA_args))[core.x][core.y];
        wa[0] = in0_addr;
        wa[1] = out_addr;
    }
}

}  // namespace ttnn::experimental::prim
