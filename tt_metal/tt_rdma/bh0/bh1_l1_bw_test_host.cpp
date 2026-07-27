// SPDX-License-Identifier: Apache-2.0
//
// Experiment 2 — RX drain-half: can one eth core's L1 sustain MAC write + Tensix-pool read at once?
// Launches the ingest probe on an external eth rail (MAC fills its L1 ring) + N Tensix drainer kernels
// (bh_rdma_l1_drainer.cpp) each NoC-reading chunks OUT of that eth L1 ring. While a DOCA sender fills
// the ring at line rate, we sample (a) the eth PACKET_DROP_CNT and (b) the pool's aggregate read Gbps.
//   drop stays 0 AND pool read ~ line rate -> eth L1 sustains the ~400G double-copy -> single-link 200G
//     RX drain is viable. Pool read plateaus below, or drops appear -> the eth L1 read port is the ceiling.
//
//   bh1_l1_bw_test [device_id] [eth_idx|"ext"] [hold_s] [num_workers] [chunk_bytes]

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
    const uint32_t chunk = (argc > 5) ? std::strtoul(argv[5], nullptr, 0) : 8192u;

    const uint64_t eth_stats_addr = TT_RDMA_DBG_ADDR;
    const uint32_t ring_addr = TT_RDMA_RX_RING_BIG_ADDR;
    const uint32_t ring_size = TT_RDMA_RX_RING_BIG_SIZE;  // 128 KB
    constexpr uint32_t kWStats = 0x40000u;                // per-worker byte counter (3 u32)
    constexpr uint32_t kWStop = 0x40040u;                 // per-worker stop flag
    constexpr uint32_t kWScratch = 0x50000u;              // per-worker read-dump scratch
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

    // Worker logical cores (a row); resolve phys for readback.
    std::vector<CoreCoord> wlog, wphys;
    for (uint32_t i = 0; i < nworkers; ++i) {
        CoreCoord wl{i, 0};
        wlog.push_back(wl);
        wphys.push_back(device->worker_core_from_logical_core(wl));
    }
    std::printf(
        "BH-L1-bw: eth rail (%u,%u) phys(%u,%u) ring@0x%x size %u  |  %u Tensix drainers, chunk %u\n",
        (unsigned)eth_logical.x,
        (unsigned)eth_logical.y,
        (unsigned)eth_phys.x,
        (unsigned)eth_phys.y,
        ring_addr,
        ring_size,
        nworkers,
        chunk);

    // Clear counters.
    std::vector<uint32_t> z9(9, 0u), z4(4, 0u);
    cluster.write_core(device->id(), eth_phys, z9, (uint32_t)eth_stats_addr);
    for (uint32_t i = 0; i < nworkers; ++i) {
        cluster.write_core(device->id(), wphys[i], z4, kWStats);
        cluster.write_core(device->id(), wphys[i], std::vector<uint32_t>{0u}, kWStop);
    }

    Program program = CreateProgram();
    // Eth ingest (MAC fills the ring; RISC only snapshots counters).
    const EthernetConfig ecfg{.noc = NOC::NOC_1, .processor = DataMovementProcessor::RISCV_1};
    const KernelHandle ek =
        CreateKernel(program, "tt_metal/tt_rdma/bh0/kernels/bh_rdma_ingest_probe.cpp", eth_logical, ecfg);
    SetRuntimeArgs(program, ek, eth_logical, {(uint32_t)eth_stats_addr, TT_RDMA_STOP_ADDR, ring_addr, ring_size});

    // Tensix drainer pool.
    const DataMovementConfig dcfg{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};
    for (uint32_t i = 0; i < nworkers; ++i) {
        const KernelHandle dk =
            CreateKernel(program, "tt_metal/tt_rdma/bh0/kernels/bh_rdma_l1_drainer.cpp", wlog[i], dcfg);
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
             (i * chunk) % ring_size,
             chunk,
             chunk,
             kWScratch});
    }

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange range(mesh_device->shape());
    workload.add_program(range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    std::printf("BH-L1-bw: up. Fire the DOCA sender now; watching pool drain Gbps + eth drop.\n");

    auto rd64 = [&](const CoreCoord& c) -> uint64_t {
        auto s = cluster.read_core<uint32_t>(device->id(), c, kWStats, 3 * sizeof(uint32_t));
        return ((uint64_t)s[1] << 32) | (uint64_t)s[0];
    };
    uint64_t prev_sum = 0;
    bool have_prev = false;
    double peak_gbps = 0.0;
    uint32_t max_drop = 0;
    const int steps = hold_s * 4;
    for (int s = 0; s < steps; ++s) {
        uint64_t sum = 0;
        for (uint32_t i = 0; i < nworkers; ++i) {
            sum += rd64(wphys[i]);
        }
        auto est = cluster.read_core<uint32_t>(device->id(), eth_phys, (uint32_t)eth_stats_addr, 9 * sizeof(uint32_t));
        const uint32_t drop = est[3];
        if (drop > max_drop) {
            max_drop = drop;
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
                "  t=%2ds  pool drain peak %6.1f Gbps  eth drop=%u  frames=%u\n", (s + 1) / 4, peak_gbps, drop, est[2]);
            std::fflush(stdout);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }

    std::printf(
        "\n  === EXPERIMENT 2 RESULT ===\n"
        "  peak pool drain (eth L1 -> Tensix): %.1f Gbps across %u workers   eth PACKET_DROP during: %u\n"
        "  Verdict: %s\n",
        peak_gbps,
        nworkers,
        max_drop,
        (max_drop == 0) ? "no eth drops while draining -> read did not starve MAC write at this pool rate"
                        : "eth drops appeared under concurrent drain -> L1 write+read contention is a ceiling");

    // Stop everything.
    cluster.write_core(device->id(), eth_phys, std::vector<uint32_t>{1u}, TT_RDMA_STOP_ADDR);
    for (uint32_t i = 0; i < nworkers; ++i) {
        cluster.write_core(device->id(), wphys[i], std::vector<uint32_t>{1u}, kWStop);
    }
    distributed::Finish(cq);
    std::cout << "BH-L1-bw: done." << std::endl;
    return 0;
}
