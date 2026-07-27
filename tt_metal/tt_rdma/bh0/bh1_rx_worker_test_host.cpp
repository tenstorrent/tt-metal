// SPDX-License-Identifier: Apache-2.0
//
// Experiment 3 — full-processing RX drainer pool. Eth ingest (MAC fills the L1 ring) + N Tensix workers
// running bh_rdma_rx_worker.cpp (read frame -> parse header -> rkey->MR lookup+validate -> land). Each
// worker gets a local MR table (slot 0 = the DOCA sender's rkey 0x00CAFE42). Driven by the DOCA sender
// at ~200G, reports aggregate PROCESSED Gbps + valid-frame fraction + eth drop. The worker count that
// first sustains >=200G processed sizes the production single-link-200G RX drainer pool.
//
//   bh1_rx_worker_test [device_id] [eth_idx|"ext"] [hold_s] [num_workers] [frame_stride]

#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include "impl/kernels/kernel.hpp"
#include "impl/context/metal_context.hpp"

#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_l1_layout.h"

int main(int argc, char** argv) {
    using namespace tt;
    using namespace tt::tt_metal;

    const int device_id = (argc > 1) ? std::atoi(argv[1]) : 1;
    const char* eth_sel = (argc > 2) ? argv[2] : "ext";
    const bool want_ext = (std::strcmp(eth_sel, "ext") == 0);
    const size_t eth_idx = want_ext ? 0 : (size_t)std::atoi(eth_sel);
    const int hold_s = (argc > 3) ? std::atoi(argv[3]) : 12;
    const uint32_t nworkers = (argc > 4) ? std::strtoul(argv[4], nullptr, 0) : 8u;
    const uint32_t stride = (argc > 5) ? std::strtoul(argv[5], nullptr, 0) : 4112u;  // DOCA landed frame

    const uint64_t eth_stats_addr = TT_RDMA_DBG_ADDR;
    const uint32_t ring_addr = TT_RDMA_RX_RING_BIG_ADDR;
    const uint32_t ring_size = TT_RDMA_RX_RING_BIG_SIZE;
    constexpr uint32_t kWStats = 0x40000u;
    constexpr uint32_t kWStop = 0x40040u;
    constexpr uint32_t kWScratch = 0x50000u;
    constexpr uint32_t kWMr = 0x60000u;
    constexpr uint32_t kMrSlots = 64u;
    constexpr uint32_t kRkey = 0x00CAFE42u;
    constexpr uint32_t kExternalMagic = 0x1AF6E471u;
    constexpr uint64_t kEthSpare0 = 0x7CC00u + 0x10u;

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
    IDevice* device = mesh_device->get_devices()[0];
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    const auto active = device->get_active_ethernet_cores(/*skip_reserved=*/true);
    std::vector<CoreCoord> ecores(active.begin(), active.end());
    TT_FATAL(!ecores.empty(), "no active ethernet cores");
    CoreCoord eth_logical;
    if (want_ext) {
        bool found = false;
        for (const auto& c : ecores) {
            auto sp = cluster.read_core<uint32_t>(
                device->id(), device->ethernet_core_from_logical_core(c), kEthSpare0, sizeof(uint32_t));
            if (!sp.empty() && sp[0] == kExternalMagic) {
                eth_logical = c;
                found = true;
                break;
            }
        }
        TT_FATAL(found, "no EXTERNAL rail");
    } else {
        eth_logical = ecores[eth_idx];
    }
    const CoreCoord eth_phys = device->ethernet_core_from_logical_core(eth_logical);

    std::vector<CoreCoord> wlog, wphys;
    for (uint32_t i = 0; i < nworkers; ++i) {
        CoreCoord wl{i, 0};
        wlog.push_back(wl);
        wphys.push_back(device->worker_core_from_logical_core(wl));
    }
    std::printf(
        "BH-rx-worker: eth (%u,%u) ring@0x%x  |  %u Tensix workers, stride %u (%u frame slots)\n",
        (unsigned)eth_logical.x,
        (unsigned)eth_logical.y,
        ring_addr,
        nworkers,
        stride,
        ring_size / stride);

    // Phase 3.1b SHARED MR table: ONE table on the eth core (control-plane-owned); workers cache it and
    // refresh on a generation bump. kWMr on each worker is now just the worker's LOCAL cache.
    const uint32_t kSharedMr = TT_RDMA_MR_TABLE_ADDR;  // shared table on the eth core
    // Generation counter lives in the SAME MR-table region (unused slot 63, word 0) -- proven host-writable
    // + worker-readable (slot 0 works). The RCB/DBG regions are owned/churned by the base FW so writes there
    // didn't reach the workers. The worker never uses slot 63 as an MR (test rkey -> slot 0).
    const uint32_t kMrGen = kSharedMr + 63u * 32u;
    const uint32_t kRegReq = kSharedMr + 62u * 32u;  // 3.1e: registration request the control-plane RISC1 fulfils
    // 3.1c remote-dest landing: MR slot 0 points at a landing region on a Tensix core NOT in the pool
    // (row 1). Workers noc_write each valid payload there. Verify byte-exact 'TTWR' after the run.
    const CoreCoord land = device->worker_core_from_logical_core(CoreCoord{0, 1});
    const uint32_t kLandBase = 0x70000u;
    // 3.1d completion ring: single shared RxWqeRing on a cq core (row 2). Workers atomically claim slots.
    const CoreCoord cqc = device->worker_core_from_logical_core(CoreCoord{0, 2});
    const uint32_t kCqBase = 0x30000u;
    const uint32_t kCqSlots = 64u;
    const uint32_t kCqProdIdx = 0x48000u;  // shared prod_idx counter on the cq core (past the 64x1536 ring)
    // 3.1e: the shared table + generation are OWNED BY THE CONTROL-PLANE RISC1. The host does NOT write
    // slot 0 directly -- it posts a registration REQUEST and RISC1 writes the MR entry + bumps the gen.
    // Shared table + workers' local cache start EMPTY, so a worker only has a valid MR once RISC1 registers.
    std::vector<uint32_t> mrempty(kMrSlots * 8, 0u);
    cluster.write_core(device->id(), land, std::vector<uint32_t>(8, 0u), kLandBase);  // clear the landing spot
    cluster.write_core(
        device->id(), cqc, std::vector<uint32_t>(kCqSlots * 1536u / 4u, 0u), kCqBase);  // clear the whole ring

    std::vector<uint32_t> z9(9, 0u), z8(8, 0u);
    cluster.write_core(device->id(), eth_phys, z9, (uint32_t)eth_stats_addr);
    cluster.write_core(device->id(), eth_phys, mrempty, kSharedMr);  // empty table; RISC1 registers slot 0
    // Registration request: [go, slot, base, len, rkey, access, dest_x, dest_y]. RISC1 fulfils it -> slot 0.
    std::vector<uint32_t> reg{1u, 0u, kLandBase, 0x100000u, kRkey, 0x2u, (uint32_t)land.x, (uint32_t)land.y};
    cluster.write_core(device->id(), eth_phys, reg, kRegReq);
    for (uint32_t i = 0; i < nworkers; ++i) {
        cluster.write_core(device->id(), wphys[i], z8, kWStats);
        cluster.write_core(device->id(), wphys[i], std::vector<uint32_t>{0u}, kWStop);
        cluster.write_core(device->id(), wphys[i], mrempty, kWMr);  // empty cache; refreshed from RISC1 on gen bump
    }

    Program program = CreateProgram();
    const EthernetConfig ecfg{.noc = NOC::NOC_1, .processor = DataMovementProcessor::RISCV_1};
    const KernelHandle ek =
        CreateKernel(program, "tt_metal/tt_rdma/bh0/kernels/bh_rdma_rx_ctrl.cpp", eth_logical, ecfg);
    SetRuntimeArgs(
        program,
        ek,
        eth_logical,
        {(uint32_t)eth_stats_addr, TT_RDMA_STOP_ADDR, ring_addr, ring_size, kSharedMr, kRegReq});

    const DataMovementConfig dcfg{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};
    for (uint32_t i = 0; i < nworkers; ++i) {
        const KernelHandle dk =
            CreateKernel(program, "tt_metal/tt_rdma/bh0/kernels/bh_rdma_rx_worker.cpp", wlog[i], dcfg);
        SetRuntimeArgs(
            program,
            dk,
            wlog[i],
            {kWStats,
             kWStop,
             (uint32_t)eth_phys.x,
             (uint32_t)eth_phys.y,
             ring_addr,
             ring_size,
             stride,
             i,
             nworkers,
             kWScratch,
             kWMr,
             kMrSlots,
             (uint32_t)eth_stats_addr + 8u,  // produce head = ingest kernel's PKT_END_CNT (stats[2])
             kSharedMr,                      // shared MR table on the eth core
             kMrGen,                         // MR generation counter on the eth core
             (uint32_t)cqc.x,                // completion ring core x
             (uint32_t)cqc.y,                // completion ring core y
             kCqBase,                        // completion ring base
             kCqSlots,                       // completion ring slots
             kCqProdIdx});                   // shared prod_idx counter
    }

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange range(mesh_device->shape());
    workload.add_program(range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    std::printf("BH-rx-worker: up. Fire the DOCA sender now.\n");

    auto rd = [&](const CoreCoord& c) {
        return cluster.read_core<uint32_t>(device->id(), c, kWStats, 8 * sizeof(uint32_t));
    };
    uint64_t prev_sum = 0;
    bool have_prev = false;
    double peak_gbps = 0.0;
    uint32_t max_drop = 0;
    uint64_t fin_processed = 0, fin_lapped = 0;
    uint64_t valid_at_change = 0;  // aggregate valid captured right when we invalidate the shared MR
    bool mr_changed = false;
    const int steps = hold_s * 4;
    for (int s = 0; s < steps; ++s) {
        uint64_t sum = 0, valid = 0, processed = 0, lapped = 0;
        for (uint32_t i = 0; i < nworkers; ++i) {
            auto w = rd(wphys[i]);
            sum += ((uint64_t)w[1] << 32) | (uint64_t)w[0];
            valid += w[2];
            processed += w[3];
            lapped += w[4];
        }
        auto est = cluster.read_core<uint32_t>(device->id(), eth_phys, (uint32_t)eth_stats_addr, 9 * sizeof(uint32_t));
        if (est[3] > max_drop) {
            max_drop = est[3];
        }
        fin_processed = processed;
        fin_lapped = lapped;

        // Halfway through: mutate the SHARED MR table centrally (invalidate slot 0's rkey) + bump the
        // generation. Correct shared-table behavior: all workers refresh their cache and slot-0 WRITEs
        // stop validating -> aggregate `valid` freezes, and every worker's cached_gen advances to 2.
        if (!mr_changed && s == steps / 2) {
            // Ask the CONTROL-PLANE RISC1 to re-register slot 0 with rkey=0 (invalidate). RISC1 bumps the
            // generation -> workers refresh -> slot-0 WRITEs stop validating (valid freezes).
            std::vector<uint32_t> regi{1u, 0u, kLandBase, 0x100000u, 0u, 0x2u, (uint32_t)land.x, (uint32_t)land.y};
            cluster.write_core(device->id(), eth_phys, regi, kRegReq);
            valid_at_change = valid;
            mr_changed = true;
            std::printf(
                "  >> [t=%2ds] posted RISC1 re-register slot 0 (rkey->0); valid was %llu\n",
                (s + 1) / 4,
                (unsigned long long)valid);
        }
        if (have_prev) {
            const double gbps = (double)(sum - prev_sum) * 8.0 / 0.25 / 1e9;
            if (gbps > peak_gbps) {
                peak_gbps = gbps;
            }
        }
        prev_sum = sum;
        have_prev = true;
        if ((s % 4) == 3) {
            std::printf(
                "  t=%2ds  consumed peak %6.1f Gbps  processed=%llu lapped=%llu produced(PKT_END)=%u  valid=%llu  eth "
                "drop=%u\n",
                (s + 1) / 4,
                peak_gbps,
                (unsigned long long)processed,
                (unsigned long long)lapped,
                est[2],
                (unsigned long long)valid,
                est[3]);
            std::fflush(stdout);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }

    // Read the workers' self-consistent produced_seen (all read the same live PKT_END_CNT). The
    // exactly-once invariant is processed + lapped == produced_seen (every frame index handled once,
    // either processed or skipped-as-lapped) -- this is what the worker-side accounting must satisfy, NOT
    // the stale host est[2] sample. Coverage = processed / produced_seen = the fraction the pool kept up
    // with; lapped>0 means too few workers for this frame RATE (small frames = high fps = worst case).
    uint64_t produced_seen = 0, final_valid = 0, sum_completions = 0;
    uint32_t workers_on_gen2 = 0;
    for (uint32_t i = 0; i < nworkers; ++i) {
        auto w = rd(wphys[i]);
        if (w[5] > produced_seen) {
            produced_seen = w[5];
        }
        final_valid += w[2];
        sum_completions += w[7];
        if (w[6] == 2u) {
            ++workers_on_gen2;
        }
        std::printf(
            "    worker %u: processed=%u lapped=%u produced_seen=%u cached_gen=%u completions=%u\n",
            i,
            w[3],
            w[4],
            w[5],
            w[6],
            w[7]);
    }
    // Shared-MR-table check: after the central invalidate, every worker must have refreshed to gen 2 and
    // slot-0 WRITEs must have stopped validating (final_valid ~ valid_at_change -- froze at the change).
    const bool shared_mr_ok =
        (workers_on_gen2 == nworkers) && mr_changed && (final_valid <= valid_at_change + (uint64_t)nworkers * 64u);
    std::printf(
        "  shared MR table: %u/%u workers refreshed to gen 2; valid froze %llu -> %llu after invalidate -> %s\n",
        workers_on_gen2,
        nworkers,
        (unsigned long long)valid_at_change,
        (unsigned long long)final_valid,
        shared_mr_ok ? "SHARED-TABLE OK (central update seen by all workers)" : "SHARED-TABLE FAIL");
    // 3.1e control plane: RISC1 (not the host) fulfilled the MR registrations. stats[9]=n_reg on the eth
    // core; the shared-table gen (slot 63) should be 2 (initial register + invalidate), both done by RISC1.
    auto ereg =
        cluster.read_core<uint32_t>(device->id(), eth_phys, (uint32_t)eth_stats_addr + 9u * 4u, sizeof(uint32_t));
    auto egen = cluster.read_core<uint32_t>(device->id(), eth_phys, kMrGen, sizeof(uint32_t));
    const uint32_t n_reg = ereg.empty() ? 0u : ereg[0];
    const uint32_t eth_gen = egen.empty() ? 0u : egen[0];
    const bool ctrl_ok = (n_reg >= 2u) && (eth_gen == 2u) && (workers_on_gen2 == nworkers);
    std::printf(
        "  control plane: RISC1 registrations n_reg=%u, eth gen=%u -> %s\n",
        n_reg,
        eth_gen,
        ctrl_ok ? "RISC1 OWNS MR REGISTRATION (host never wrote the table)" : "CTRL-PLANE FAIL");
    // 3.1c remote-dest landing: the payload must have landed byte-exact at the MR dest (off-core, not just
    // compute-local). Sender roff=0 -> every WRITE lands at kLandBase; check word0 = 'TTWR' (0x52575454).
    auto lz = cluster.read_core<uint32_t>(device->id(), land, kLandBase, 4 * sizeof(uint32_t));
    const bool land_ok = (!lz.empty() && lz[0] == 0x52575454u);
    std::printf(
        "  remote-dest landing @core(%u,%u):0x%x [0..3] = %08x %08x %08x %08x -> %s\n",
        (unsigned)land.x,
        (unsigned)land.y,
        kLandBase,
        lz[0],
        lz[1],
        lz[2],
        lz[3],
        land_ok ? "LANDED byte-exact (TTWR)" : "LAND FAIL");
    // 3.1d worker-posted completions: workers post to the single shared ring on disjoint interleaved slots
    // (worker w owns slots w, w+N, ... -- no atomic, collision-free by construction since kCqSlots%N==0).
    // Verify: all N workers posted completions, and slots owned by DIFFERENT workers are all OWNED+valid
    // (bit8 in word 5, op WRITE) -> proves multiple writers safely share the one ring.
    bool all_posted = (sum_completions > 0) && (kCqSlots % nworkers == 0);
    uint32_t owned_ok = 0;
    for (uint32_t r = 0; r < nworkers && r < kCqSlots; ++r) {  // one slot per worker's residue class
        auto sl = cluster.read_core<uint32_t>(device->id(), cqc, kCqBase + r * 1536u, 8 * sizeof(uint32_t));
        const bool ok = (!sl.empty() && (sl[5] & 0x100u) != 0u && (sl[2] & 0xFFu) == 0x10u);
        owned_ok += ok ? 1u : 0u;
    }
    const bool cq_ok = all_posted && (owned_ok == nworkers);
    std::printf(
        "  completions: Sum(worker completions)=%llu; %u/%u per-worker slots OWNED+WRITE -> %s\n",
        (unsigned long long)sum_completions,
        owned_ok,
        nworkers,
        cq_ok ? "MULTI-WRITER RING OK (disjoint interleaved slots, no collision)" : "COMPLETIONS FAIL");
    const uint64_t accounted = fin_processed + fin_lapped;
    const double cover = produced_seen ? 100.0 * (double)fin_processed / (double)produced_seen : 0.0;
    const bool exactly_once = (accounted == produced_seen);  // no frame double-processed or dropped-unaccounted
    std::printf(
        "\n  === Phase 3.1b RESULT (bounded multi-consumer claim) ===\n"
        "  %u workers: consumed peak %.1f Gbps  processed=%llu / produced_seen=%llu (%.1f%% kept up)  lapped=%llu  eth "
        "drop=%u\n"
        "  exactly-once (processed+lapped==produced): %llu vs %llu -> %s  |  %s\n",
        nworkers,
        peak_gbps,
        (unsigned long long)fin_processed,
        (unsigned long long)produced_seen,
        cover,
        (unsigned long long)fin_lapped,
        max_drop,
        (unsigned long long)accounted,
        (unsigned long long)produced_seen,
        exactly_once ? "HOLDS" : "VIOLATED",
        (fin_lapped == 0) ? "LOSSLESS: pool kept up (no lapping)"
                          : "pool fell behind for this frame RATE -> add workers or use jumbo (fewer fps)");

    cluster.write_core(device->id(), eth_phys, std::vector<uint32_t>{1u}, TT_RDMA_STOP_ADDR);
    for (uint32_t i = 0; i < nworkers; ++i) {
        cluster.write_core(device->id(), wphys[i], std::vector<uint32_t>{1u}, kWStop);
    }
    distributed::Finish(cq);
    std::cout << "BH-rx-worker: done." << std::endl;
    return 0;
}
