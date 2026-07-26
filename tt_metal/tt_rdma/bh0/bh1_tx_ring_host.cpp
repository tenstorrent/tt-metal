// SPDX-License-Identifier: Apache-2.0
//
// BH.2a/2b host — RISC-off-datapath TX ring (tt-rdma-tx-ring-spec.md). Acts as the PRODUCER: pre-fills
// ring_size payload slots (each [32B tt_rdma_hdr][payload]) + descriptors into L1, then dispatches
// the ring-drainer kernel (bh_rdma_tx_ring.cpp) which arms them with accept-ahead + MAX_PKT
// auto-split — no per-frame header build/CRC/copy on the RISC. Reports arm-rate (hb delta/sec);
// pair with an ethtool rx_bytes_phy sampler on the BF3 for wire BW.
//
// BH.2b: eth_sel="ext" now dispatches a drainer to EVERY external rail at once (one MeshWorkload) so
// the aggregate across the two 200G links can be driven together (>200G). A numeric eth_idx targets a
// single rail. Each rail has its own L1 (HB/STOP/DBG/ring at the same offsets on its own core).
//
// It also reads back each rail's TXQ packet-counter snapshots (PKT_START/PKT_END/WORD/STATUS,
// before vs after the run) to diagnose egress (tx-ring-spec §11):
//   - PKT_START delta == 0            -> command accepted but no packet ever started (source-fetch /
//                                        MAX_PKT / framing).
//   - PKT_START moves, PKT_END/WORD 0 -> starts but stalls mid-drain.
//   - all three move                  -> HW is transmitting.
//
//   bh1_tx_ring [device] [eth_idx|"ext"] [dst_mac] [ring_size] [payload_len] [hold_s] [max_pkt] [txq]
//   [pace] [payload_base]
//
// NB MAX_PKT is STICKY across runs (the register keeps the previous value if not written): always pass
// max_pkt explicitly (<= the wire egress ceiling). `pace` spins between arms — raw START_RAW has no deep
// accept-ahead FIFO, so arming faster than the TXQ drains wedges it; pace to the sustainable rate.

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include "impl/kernels/kernel.hpp"
#include "impl/context/metal_context.hpp"

#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_wire.h"
#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_hdr_build.h"
#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_l1_layout.h"

static bool parse_mac(const char* s, uint32_t& hi, uint32_t& lo) {
    unsigned b[6];
    if (std::sscanf(s, "%x:%x:%x:%x:%x:%x", &b[0], &b[1], &b[2], &b[3], &b[4], &b[5]) != 6) {
        return false;
    }
    uint64_t v = 0;
    for (int i = 0; i < 6; ++i) {
        v = (v << 8) | (uint64_t)(b[i] & 0xFF);
    }
    hi = (uint32_t)(v >> 32);
    lo = (uint32_t)(v & 0xFFFFFFFFu);
    return true;
}

int main(int argc, char** argv) {
    using namespace tt;
    using namespace tt::tt_metal;

    const int device_id = (argc > 1) ? std::atoi(argv[1]) : 1;
    const char* eth_sel = (argc > 2) ? argv[2] : "ext";
    const bool want_ext = (std::strcmp(eth_sel, "ext") == 0);
    const size_t eth_idx = want_ext ? 0 : (size_t)std::atoi(argv[2]);
    const char* dst_mac_s = (argc > 3) ? argv[3] : "ff:ff:ff:ff:ff:ff";
    uint32_t ring_size = (argc > 4) ? std::strtoul(argv[4], nullptr, 0) : 32u;
    const uint32_t payload_len = (argc > 5) ? std::strtoul(argv[5], nullptr, 0) : 4048u;
    const int hold_s = (argc > 6) ? std::atoi(argv[6]) : 10;
    const uint32_t max_pkt = (argc > 7) ? std::strtoul(argv[7], nullptr, 0) : 4080u;
    const uint32_t txq = (argc > 8) ? std::strtoul(argv[8], nullptr, 0) : 2u;
    const uint32_t pace = (argc > 9) ? std::strtoul(argv[9], nullptr, 0) : 0u;  // spin between arms
    const uint32_t payload_base = (argc > 10) ? std::strtoul(argv[10], nullptr, 0) : TT_RDMA_WQE_PAYLOAD_ADDR;

    const uint32_t frame_len = TT_RDMA_HDR_BYTES + payload_len;
    const uint32_t slot_stride = (frame_len + 15u) & ~15u;  // 16-B aligned slots
    if ((uint64_t)ring_size * slot_stride > TT_RDMA_WQE_PAYLOAD_SIZE) {
        ring_size = TT_RDMA_WQE_PAYLOAD_SIZE / slot_stride;  // clamp to fit the 128 KB payload region
    }
    uint32_t dmac_hi = 0, dmac_lo = 0;
    if (!parse_mac(dst_mac_s, dmac_hi, dmac_lo)) {
        std::cout << "bad dst_mac\n";
        return 2;
    }

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
    IDevice* device = mesh_device->get_devices()[0];
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    const auto active = device->get_active_ethernet_cores(/*skip_reserved=*/true);
    std::vector<CoreCoord> cores(active.begin(), active.end());
    TT_FATAL(!cores.empty(), "no active ethernet cores on device {}", device_id);
    constexpr uint64_t kEthStatusSpare0 = 0x7CC00u + 0x10u;
    constexpr uint32_t kExternalMagic = 0x1AF6E471u;

    // Select the rail set: "ext" = ALL external/NIC rails (aggregate); a numeric idx = one core.
    std::vector<CoreCoord> rails;
    if (want_ext) {
        for (const auto& c : cores) {
            auto sp = cluster.read_core<uint32_t>(
                device->id(), device->ethernet_core_from_logical_core(c), kEthStatusSpare0, sizeof(uint32_t));
            if (!sp.empty() && sp[0] == kExternalMagic) {
                rails.push_back(c);
            }
        }
        TT_FATAL(!rails.empty(), "no EXTERNAL rail on device {}", device_id);
    } else {
        TT_FATAL(eth_idx < cores.size(), "eth_idx out of range");
        rails.push_back(cores[eth_idx]);
    }

    std::vector<CoreCoord> phys;
    for (const auto& c : rails) {
        phys.push_back(device->ethernet_core_from_logical_core(c));
    }
    std::cout << "BH.2b: dev " << device_id << "  rails=" << rails.size() << "  ring=" << ring_size
              << " frame=" << frame_len << "B slot=" << slot_stride << "B txq=" << txq << " max_pkt=" << max_pkt
              << " pace=" << pace << "\n";

    // --- PRODUCER: pre-fill each rail's L1 (payload slots + descriptors) + seed the counter sentinel ---
    for (size_t r = 0; r < rails.size(); ++r) {
        for (uint32_t s = 0; s < ring_size; ++s) {
            tt_rdma_hdr_t h;
            tt_rdma_build_hdr(&h, TT_OP_PROBE, TT_RDMA_VERSION, (uint16_t)0x50B, payload_len, s + 1, 0u, 0u, 0u);
            std::vector<uint32_t> frame(frame_len / 4, 0xAA55AA55u);
            std::memcpy(frame.data(), &h, TT_RDMA_HDR_BYTES);
            cluster.write_core(device->id(), phys[r], frame, payload_base + s * slot_stride);
            std::vector<uint32_t> d{s * slot_stride, frame_len, (1u << 8) | (txq & 3u), s};
            cluster.write_core(device->id(), phys[r], d, TT_RDMA_WQE_DESCR_ADDR + s * 16u);
        }
        std::vector<uint32_t> sentinel(8, 0xDEADBEEFu);
        cluster.write_core(device->id(), phys[r], sentinel, TT_RDMA_DBG_ADDR);
        std::printf(
            "  rail%zu core (%u,%u) phys (%u,%u) prefilled\n",
            r,
            (unsigned)rails[r].x,
            (unsigned)rails[r].y,
            (unsigned)phys[r].x,
            (unsigned)phys[r].y);
    }
    std::fflush(stdout);

    // Dispatch a drainer kernel to every rail in ONE program.
    Program program = CreateProgram();
    const EthernetConfig cfg{.noc = NOC::NOC_1, .processor = DataMovementProcessor::RISCV_1};
    for (size_t r = 0; r < rails.size(); ++r) {
        const KernelHandle k = CreateKernel(program, "tt_metal/tt_rdma/bh0/kernels/bh_rdma_tx_ring.cpp", rails[r], cfg);
        SetRuntimeArgs(
            program,
            k,
            rails[r],
            {TT_RDMA_HB_ADDR,
             TT_RDMA_STOP_ADDR,
             /*num_arms=*/0u,
             ring_size,
             txq,
             max_pkt,
             dmac_hi,
             dmac_lo,
             pace,
             payload_base});
    }

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    std::cout << "BH.2b: " << rails.size() << " ring drainer(s) dispatched. per-rail arm-rate:" << std::endl;

    std::vector<uint32_t> prev(rails.size(), 0);
    for (int s = 0; s < hold_s; ++s) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        double total = 0;
        std::printf("  t=%2ds ", s);
        for (size_t r = 0; r < rails.size(); ++r) {
            auto v = cluster.read_core<uint32_t>(device->id(), phys[r], TT_RDMA_HB_ADDR, sizeof(uint32_t));
            uint32_t cur = v.empty() ? 0 : v[0];
            double rate = (cur - prev[r]) / 1000.0;
            total += rate;
            std::printf(" rail%zu=%.0fk/s", r, rate);
            prev[r] = cur;
        }
        std::printf("  total=%.0f k arms/s\n", total);
        std::fflush(stdout);
    }

    // Stop all rails, then Finish.
    const std::vector<uint32_t> stop_val{1u};
    for (size_t r = 0; r < rails.size(); ++r) {
        cluster.write_core(device->id(), phys[r], stop_val, TT_RDMA_STOP_ADDR);
    }
    distributed::Finish(cq);

    // Per-rail TXQ counter readback (PKT_END = wire frames transmitted; compare to BF3 rx_packets_phy).
    uint64_t total_pkts = 0;
    std::printf("\nBH.2b per-rail TXQ counters:\n");
    for (size_t r = 0; r < rails.size(); ++r) {
        auto before = cluster.read_core<uint32_t>(device->id(), phys[r], TT_RDMA_DBG_BEFORE_ADDR, 4 * sizeof(uint32_t));
        auto after = cluster.read_core<uint32_t>(device->id(), phys[r], TT_RDMA_DBG_AFTER_ADDR, 4 * sizeof(uint32_t));
        auto hbv = cluster.read_core<uint32_t>(device->id(), phys[r], TT_RDMA_HB_ADDR, sizeof(uint32_t));
        const uint32_t armed = hbv.empty() ? 0 : hbv[0];
        const bool wrote_back = !after.empty() && after[0] != 0xDEADBEEFu;
        const uint32_t start_delta = (after.size() > 0 && before.size() > 0) ? (after[0] - before[0]) : 0;
        const uint32_t end_delta = (after.size() > 1 && before.size() > 1) ? (after[1] - before[1]) : 0;
        total_pkts += end_delta;
        std::printf(
            "  rail%zu core(%u,%u): armed=%u  PKT_START=%u  PKT_END=%u  %s\n",
            r,
            (unsigned)rails[r].x,
            (unsigned)rails[r].y,
            armed,
            start_delta,
            end_delta,
            wrote_back ? (start_delta ? "EGRESS" : "0-STARTED") : "no-AFTER-snapshot");
    }
    std::printf(
        "BH.2b: total wire frames transmitted across %zu rail(s) = %llu\n",
        rails.size(),
        (unsigned long long)total_pkts);

    std::cout << "BH.2b: done; clean shutdown." << std::endl;
    return 0;
}
