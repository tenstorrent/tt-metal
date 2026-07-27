// SPDX-License-Identifier: Apache-2.0
//
// Host for experiment 1 (RX ingress-ceiling go/no-go). Loads bh_rdma_ingest_probe.cpp on an external
// rail's RISC1 (which does NO per-frame work), and samples the raw MAC RX counters at 250 ms (fast
// enough that WORD_CNT's 32-bit counter wraps at most once/sample even at line rate). Reports the
// MAC->L1 ingress bandwidth (wrap-corrected WORD_CNT delta x16) and total drops.
//
//   bh1_ingest_probe [device_id] [eth_idx|"ext"] [hold_s]
//   While it holds, blast 0x1AF6 frames at the rail (DPU-side or DOCA sender) and read the Gbps + drops.

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
    // Landed bytes/frame after L2 strip (32B hdr + payload). PKT_END_CNT is the reliable rate source
    // (WORD_CNT/BYTE_CNT under-count in raw-wrap on this HW); frames * landed = MAC->L1 bandwidth.
    const uint32_t payload = (argc > 4) ? std::strtoul(argv[4], nullptr, 0) : 4080u;
    const double landed_bytes = 32.0 + (double)payload;

    constexpr uint64_t kEthStatusSpare0 = 0x7CC00u + 0x10u;
    constexpr uint32_t kExternalMagic = 0x1AF6E471u;
    const uint64_t stats_addr = TT_RDMA_DBG_ADDR;
    const uint32_t rx_ring_addr = TT_RDMA_RX_RING_BIG_ADDR;
    const uint32_t rx_ring_size = TT_RDMA_RX_RING_BIG_SIZE;

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
    IDevice* device = mesh_device->get_devices()[0];
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    const auto active = device->get_active_ethernet_cores(/*skip_reserved=*/true);
    std::vector<CoreCoord> cores(active.begin(), active.end());
    TT_FATAL(!cores.empty(), "no active ethernet cores on device {}", device_id);
    CoreCoord eth_logical;
    if (want_ext) {
        bool found = false;
        for (const auto& c : cores) {
            auto sp = cluster.read_core<uint32_t>(
                device->id(), device->ethernet_core_from_logical_core(c), kEthStatusSpare0, sizeof(uint32_t));
            if (!sp.empty() && sp[0] == kExternalMagic) {
                eth_logical = c;
                found = true;
                break;
            }
        }
        TT_FATAL(found, "no EXTERNAL rail on device {}", device_id);
    } else {
        TT_FATAL(eth_idx < cores.size(), "eth_idx out of range");
        eth_logical = cores[eth_idx];
    }
    const CoreCoord eth_phys = device->ethernet_core_from_logical_core(eth_logical);
    std::printf(
        "BH-ingest-probe: dev %d core (%u,%u) phys (%u,%u) RXQ=%u -- RISC does NO per-frame work.\n"
        "  Blast 0x1AF6 at this rail now; watching MAC->L1 ingress Gbps + drops.\n",
        device_id,
        (unsigned)eth_logical.x,
        (unsigned)eth_logical.y,
        (unsigned)eth_phys.x,
        (unsigned)eth_phys.y,
        TT_RDMA_RX_QUEUE);

    std::vector<uint32_t> zstats(9, 0u);
    cluster.write_core(device->id(), eth_phys, zstats, (uint32_t)stats_addr);

    Program program = CreateProgram();
    const EthernetConfig cfg{.noc = NOC::NOC_1, .processor = DataMovementProcessor::RISCV_1};
    const KernelHandle k =
        CreateKernel(program, "tt_metal/tt_rdma/bh0/kernels/bh_rdma_ingest_probe.cpp", eth_logical, cfg);
    SetRuntimeArgs(program, k, eth_logical, {(uint32_t)stats_addr, TT_RDMA_STOP_ADDR, rx_ring_addr, rx_ring_size});

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    std::cout << "BH-ingest-probe: kernel up.\n";

    // Sample at 250 ms. WORD_CNT is 16-B words: at 200G that's ~1.56e9 words/s -> 3.9e8 words per 250 ms,
    // far under 2^32, so a wrap-corrected 32-bit delta is unambiguous.
    const int steps = hold_s * 4;
    uint32_t prev_fr = 0;
    bool have_prev = false;
    uint64_t total_dframes = 0, max_drop = 0, peak_afifo = 0;
    double peak_gbps = 0.0;
    for (int s = 0; s < steps; ++s) {
        auto st = cluster.read_core<uint32_t>(device->id(), eth_phys, (uint32_t)stats_addr, 9 * sizeof(uint32_t));
        const uint32_t frames = st[2], drop = st[3], afifo = st[6] & 0xFFFFu;
        if (have_prev) {
            const uint32_t dfr = frames - prev_fr;  // uint32 wrap-correct; PKT_END_CNT wraps slowly
            total_dframes += dfr;
            const double gbps = (double)dfr * landed_bytes * 8.0 / 0.25 / 1e9;
            if (gbps > peak_gbps) {
                peak_gbps = gbps;
            }
        }
        prev_fr = frames;
        have_prev = true;
        if (drop > max_drop) {
            max_drop = drop;
        }
        if (afifo > peak_afifo) {
            peak_afifo = afifo;
        }
        if ((s % 4) == 3) {
            const double avg = (double)total_dframes * landed_bytes * 8.0 / ((s + 1) / 4.0) / 1e9;
            std::printf(
                "  t=%2ds  ingress avg~%6.1f Gbps  peak %6.1f  drop=%u  afifo_peak=%llu  frames=%u\n",
                (s + 1) / 4,
                avg,
                peak_gbps,
                drop,
                (unsigned long long)peak_afifo,
                frames);
            std::fflush(stdout);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }

    std::printf(
        "\n  === EXPERIMENT 1 RESULT ===\n"
        "  peak MAC->L1 ingress: %.1f Gbps   total drops: %llu   peak AFIFO fullness: %llu\n"
        "  Verdict: %s\n",
        peak_gbps,
        (unsigned long long)max_drop,
        (unsigned long long)peak_afifo,
        (max_drop == 0)
            ? "drop==0 at the offered rate -> MAC->L1 write side keeps up so far (raise the sender toward 200G)"
            : "drops present -> MAC->L1 write side saturated at this rate (ingress is a ceiling)");

    const std::vector<uint32_t> stop_val{1u};
    cluster.write_core(device->id(), eth_phys, stop_val, TT_RDMA_STOP_ADDR);
    distributed::Finish(cq);
    std::cout << "BH-ingest-probe: done." << std::endl;
    return 0;
}
