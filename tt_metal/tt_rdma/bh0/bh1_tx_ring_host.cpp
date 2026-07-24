// SPDX-License-Identifier: Apache-2.0
//
// BH.2a host — RISC-off-datapath TX ring (tt-rdma-tx-ring-spec.md). Acts as the PRODUCER: pre-fills
// ring_size payload slots (each [32B tt_rdma_hdr][payload]) + descriptors into L1, then dispatches
// the ring-drainer kernel (bh_rdma_tx_ring.cpp) which arms them with accept-ahead + MAX_PKT
// auto-split — no per-frame header build/CRC/copy on the RISC. Reports arm-rate (hb delta/sec);
// pair with an ethtool rx_bytes_phy sampler on the BF3 for wire BW.
//
// It also reads back the kernel's TXQ packet-counter snapshots (PKT_START/PKT_END/WORD/STATUS,
// before vs after the run) to diagnose the "CMD accepted but 0 wire bytes" bug (tx-ring-spec §11):
//   - PKT_START delta == 0            -> command accepted but no packet ever started (source-fetch /
//                                        MAX_PKT / framing). Try max_pkt=0 and/or risc_touch=1.
//   - PKT_START moves, PKT_END/WORD 0 -> starts but stalls mid-drain.
//   - all three move                  -> HW is transmitting; look downstream (TXPKT row / peer / cable).
//
//   bh1_tx_ring [device] [eth_idx|"ext"] [dst_mac] [ring_size] [payload_len] [hold_s] [max_pkt] [txq] [pace]
//   [payload_base] [risc_touch]
//
// Two experiments to isolate the bug (tx-ring-spec §11 next-diagnostic list):
//   max_pkt=0    : arg[7]=0 -> don't set ETH_TXQ_MAX_PKT_SIZE_BYTES (match the working non-burst probe,
//                  which never sets it; the only probe path that DID set it is the one that regressed).
//   risc_touch=1 : arg[11]=1 -> kernel RISC-writes the header line before each arm (the working probe
//                  does this implicitly; the ring does not). Tests the source-fetch/coherence theory.

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
    const uint32_t risc_touch = (argc > 11) ? std::strtoul(argv[11], nullptr, 0) : 0u;  // 1 = RISC-touch source

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
    CoreCoord eth_logical = cores.empty() ? CoreCoord{} : cores[0];
    constexpr uint64_t kEthStatusSpare0 = 0x7CC00u + 0x10u;
    constexpr uint32_t kExternalMagic = 0x1AF6E471u;
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
    std::cout << "BH.2a: dev " << device_id << " core (" << eth_logical.x << "," << eth_logical.y << ") phys ("
              << eth_phys.x << "," << eth_phys.y << ")  ring=" << ring_size << " frame=" << frame_len
              << "B slot=" << slot_stride << "B txq=" << txq << " max_pkt=" << max_pkt << " pace=" << pace
              << " risc_touch=" << risc_touch << "\n";

    // --- PRODUCER: pre-fill payload slots + descriptors in L1 (one-shot; BH.2c adds live/DMA feed) ---
    for (uint32_t s = 0; s < ring_size; ++s) {
        // header at the start of the slot
        tt_rdma_hdr_t h;
        tt_rdma_build_hdr(&h, TT_OP_PROBE, TT_RDMA_VERSION, (uint16_t)0x50B, payload_len, s + 1, 0u, 0u, 0u);
        std::vector<uint32_t> frame(frame_len / 4, 0xAA55AA55u);
        std::memcpy(frame.data(), &h, TT_RDMA_HDR_BYTES);
        cluster.write_core(device->id(), eth_phys, frame, payload_base + s * slot_stride);
        // descriptor: {frame_off, frame_len, flags_txq(OWNED_BY_FW|txq), cookie}
        std::vector<uint32_t> d{s * slot_stride, frame_len, (1u << 8) | (txq & 3u), s};
        cluster.write_core(device->id(), eth_phys, d, TT_RDMA_WQE_DESCR_ADDR + s * 16u);
    }

    // Debug: read back descriptor[0] + payload[0] to confirm the pre-fill landed in L1.
    {
        auto d0 = cluster.read_core<uint32_t>(device->id(), eth_phys, TT_RDMA_WQE_DESCR_ADDR, 4 * sizeof(uint32_t));
        auto pl = cluster.read_core<uint32_t>(device->id(), eth_phys, payload_base, 2 * sizeof(uint32_t));
        std::printf(
            "  prefill check: descr[0]={off=%u len=%u flags=0x%x cookie=%u}  payload[0..1]=%08x %08x\n",
            d0[0],
            d0[1],
            d0[2],
            d0[3],
            pl[0],
            pl[1]);
        std::fflush(stdout);
    }

    // Seed the TXQ-counter debug region with a sentinel so we can tell if the kernel wrote it back.
    {
        std::vector<uint32_t> sentinel(8, 0xDEADBEEFu);
        cluster.write_core(device->id(), eth_phys, sentinel, TT_RDMA_DBG_ADDR);
    }

    Program program = CreateProgram();
    const EthernetConfig cfg{.noc = NOC::NOC_1, .processor = DataMovementProcessor::RISCV_1};
    const KernelHandle k = CreateKernel(program, "tt_metal/tt_rdma/bh0/kernels/bh_rdma_tx_ring.cpp", eth_logical, cfg);
    SetRuntimeArgs(
        program,
        k,
        eth_logical,
        {TT_RDMA_HB_ADDR,
         TT_RDMA_STOP_ADDR,
         /*num_arms=*/0u,
         ring_size,
         txq,
         max_pkt,
         dmac_hi,
         dmac_lo,
         pace,
         payload_base,
         risc_touch});

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    std::cout << "BH.2a: ring drainer dispatched. arm-rate:" << std::endl;

    uint32_t prev = 0;
    for (int s = 0; s < hold_s; ++s) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        auto v = cluster.read_core<uint32_t>(device->id(), eth_phys, TT_RDMA_HB_ADDR, sizeof(uint32_t));
        uint32_t cur = v.empty() ? 0 : v[0];
        // post-dispatch: is the descriptor/payload still intact, or did tt-metal clobber it?
        auto dd = cluster.read_core<uint32_t>(device->id(), eth_phys, TT_RDMA_WQE_DESCR_ADDR, 2 * sizeof(uint32_t));
        auto pp = cluster.read_core<uint32_t>(device->id(), eth_phys, payload_base, sizeof(uint32_t));
        std::printf(
            "  t=%2ds  armed=%u  arm-rate=%.0f k/s  descr0{off=%u len=%u} pl0=%08x\n",
            s,
            cur,
            (cur - prev) / 1000.0,
            dd[0],
            dd[1],
            pp[0]);
        std::fflush(stdout);
        prev = cur;
    }

    const std::vector<uint32_t> stop_val{1u};
    cluster.write_core(device->id(), eth_phys, stop_val, TT_RDMA_STOP_ADDR);
    distributed::Finish(cq);

    // TXQ packet-counter diagnosis: the kernel snapshotted the counters before the first arm and after
    // the last. If it never wrote them back (still the 0xDEADBEEF sentinel), the kernel didn't reach
    // the snapshot points. Otherwise diff BEFORE vs AFTER to localize the 0-wire-bytes bug.
    {
        auto before =
            cluster.read_core<uint32_t>(device->id(), eth_phys, TT_RDMA_DBG_BEFORE_ADDR, 4 * sizeof(uint32_t));
        auto after = cluster.read_core<uint32_t>(device->id(), eth_phys, TT_RDMA_DBG_AFTER_ADDR, 4 * sizeof(uint32_t));
        auto hbv = cluster.read_core<uint32_t>(device->id(), eth_phys, TT_RDMA_HB_ADDR, sizeof(uint32_t));
        const uint32_t armed = hbv.empty() ? 0 : hbv[0];
        const bool wrote_back = !after.empty() && after[0] != 0xDEADBEEFu;
        std::printf("\nBH.2a TXQ counters (txq=%u, armed=%u):\n", txq, armed);
        if (!wrote_back) {
            std::printf("  kernel did NOT write AFTER snapshot (still sentinel) - it never reached the exit path.\n");
        }
        static const char* kName[4] = {"PKT_START", "PKT_END  ", "WORD_CNT ", "STATUS   "};
        for (int i = 0; i < 4; ++i) {
            const uint32_t b = before.size() > (size_t)i ? before[i] : 0;
            const uint32_t a = after.size() > (size_t)i ? after[i] : 0;
            if (i < 3) {
                std::printf("  %s  before=%-10u after=%-10u  delta=%d\n", kName[i], b, a, (int)(a - b));
            } else {
                std::printf("  %s  before=0x%08x after=0x%08x  (bit16=CMD_ONGOING)\n", kName[i], b, a);
            }
        }
        const uint32_t start_delta = (after.size() > 0 && before.size() > 0) ? (after[0] - before[0]) : 0;
        const uint32_t end_delta = (after.size() > 1 && before.size() > 1) ? (after[1] - before[1]) : 0;
        const uint32_t word_delta = (after.size() > 2 && before.size() > 2) ? (after[2] - before[2]) : 0;
        std::printf("  verdict: ");
        if (!wrote_back) {
            std::printf("inconclusive (no AFTER snapshot).\n");
        } else if (start_delta == 0) {
            std::printf(
                "CMD accepted but 0 packets STARTED -> source-fetch/MAX_PKT/framing. Retry max_pkt=0 / "
                "risc_touch=1.\n");
        } else if (end_delta == 0 || word_delta == 0) {
            std::printf("packets STARTED (%u) but did not finish draining -> stalls mid-transmit.\n", start_delta);
        } else {
            std::printf(
                "HW transmitted %u pkts / %u words -> datapath OK; check TXPKT row / peer / cable.\n",
                start_delta,
                word_delta);
        }
        std::fflush(stdout);
    }

    std::cout << "BH.2a: done; clean shutdown." << std::endl;
    return 0;
}
