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

    // Per-worker MR table: slot 0 valid (rkey, REMOTE_WRITE, 1 MB len).
    std::vector<uint32_t> mrtab(kMrSlots * 8, 0u);
    mrtab[2] = 0x100000u;  // length lo
    mrtab[4] = kRkey;      // rkey
    mrtab[5] = 0x2u;       // access = REMOTE_WRITE

    std::vector<uint32_t> z9(9, 0u), z4(4, 0u);
    cluster.write_core(device->id(), eth_phys, z9, (uint32_t)eth_stats_addr);
    for (uint32_t i = 0; i < nworkers; ++i) {
        cluster.write_core(device->id(), wphys[i], z4, kWStats);
        cluster.write_core(device->id(), wphys[i], std::vector<uint32_t>{0u}, kWStop);
        cluster.write_core(device->id(), wphys[i], mrtab, kWMr);
    }

    Program program = CreateProgram();
    const EthernetConfig ecfg{.noc = NOC::NOC_1, .processor = DataMovementProcessor::RISCV_1};
    const KernelHandle ek =
        CreateKernel(program, "tt_metal/tt_rdma/bh0/kernels/bh_rdma_ingest_probe.cpp", eth_logical, ecfg);
    SetRuntimeArgs(program, ek, eth_logical, {(uint32_t)eth_stats_addr, TT_RDMA_STOP_ADDR, ring_addr, ring_size});

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
             kMrSlots});
    }

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange range(mesh_device->shape());
    workload.add_program(range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    std::printf("BH-rx-worker: up. Fire the DOCA sender now.\n");

    auto rd = [&](const CoreCoord& c) {
        return cluster.read_core<uint32_t>(device->id(), c, kWStats, 4 * sizeof(uint32_t));
    };
    uint64_t prev_sum = 0;
    bool have_prev = false;
    double peak_gbps = 0.0;
    uint32_t max_drop = 0;
    const int steps = hold_s * 4;
    for (int s = 0; s < steps; ++s) {
        uint64_t sum = 0, valid = 0;
        for (uint32_t i = 0; i < nworkers; ++i) {
            auto w = rd(wphys[i]);
            sum += ((uint64_t)w[1] << 32) | (uint64_t)w[0];
            valid += w[2];
        }
        auto est = cluster.read_core<uint32_t>(device->id(), eth_phys, (uint32_t)eth_stats_addr, 9 * sizeof(uint32_t));
        if (est[3] > max_drop) {
            max_drop = est[3];
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
                "  t=%2ds  processed peak %6.1f Gbps  valid_frames=%llu  eth drop=%u  eth frames=%u\n",
                (s + 1) / 4,
                peak_gbps,
                (unsigned long long)valid,
                est[3],
                est[2]);
            std::fflush(stdout);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }

    std::printf(
        "\n  === EXPERIMENT 3 RESULT ===\n"
        "  %u Tensix workers processed (read+parse+rkey->MR+validate) peak %.1f Gbps   eth drop: %u\n"
        "  Verdict: %s single-link-200G RX with this pool size (compute-local landing; remote MR dest ~2x workers).\n",
        nworkers,
        peak_gbps,
        max_drop,
        (peak_gbps >= 200.0) ? ">=200 Gbps -> SUFFICIENT for" : "<200 Gbps -> need more workers for");

    cluster.write_core(device->id(), eth_phys, std::vector<uint32_t>{1u}, TT_RDMA_STOP_ADDR);
    for (uint32_t i = 0; i < nworkers; ++i) {
        cluster.write_core(device->id(), wphys[i], std::vector<uint32_t>{1u}, kWStop);
    }
    distributed::Finish(cq);
    std::cout << "BH-rx-worker: done." << std::endl;
    return 0;
}
