// SPDX-License-Identifier: Apache-2.0
//
// Host driver for the ETH-CTRL ROCE_ICRC characterization probe (bh_rdma_icrc_probe.cpp).
// Answers: does the inline HW CRC engine engage on our raw 0x1AF6 frames, and what CTRL
// bit-order/init makes its RX_CALCULATED == the software tt_rdma_crc32 over header [0..27]?
// If yes, that config is what the RX kernel bakes in to offload the CRC (then RX_CHECK_EN on).
//
// Usage:
//   bh1_icrc_probe [device_id] [eth_idx|"ext"] [hold_s] [ctrl_hex] [rx_init_hex] [wrap]
//     ctrl_hex / rx_init_hex: value to program, or "-" / omitted => leave POR (read-only observe).
//     Sweep the 16x16 bit-order combos from a shell loop over ctrl_hex; base off POR 0x30700000.
//
// While it holds, fire a KNOWN frame from the BF3 (its header_cksum is a valid CRC-32):
//   sudo tt_rdma_bf3_send <if> 200 02:00:00:00:00:02 0x1af6 0x10 256 0x00CAFE42 0
// Then read the printed sw_crc and compare to rx_calculated.

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include "impl/kernels/kernel.hpp"
#include "impl/context/metal_context.hpp"

#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_wire.h"
#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_l1_layout.h"

static uint32_t parse_word(const char* s, uint32_t dflt) {
    if (s == nullptr || std::strcmp(s, "-") == 0) {
        return dflt;
    }
    return (uint32_t)std::strtoul(s, nullptr, 0);
}

int main(int argc, char** argv) {
    using namespace tt;
    using namespace tt::tt_metal;

    const int device_id = (argc > 1) ? std::atoi(argv[1]) : 1;
    const char* eth_sel = (argc > 2) ? argv[2] : "ext";
    const bool want_ext = (std::strcmp(eth_sel, "ext") == 0);
    const size_t eth_idx = want_ext ? 0 : (size_t)std::atoi(eth_sel);
    const int hold_s = (argc > 3) ? std::atoi(argv[3]) : 30;
    const uint32_t prog_ctrl = parse_word((argc > 4) ? argv[4] : nullptr, 0xFFFFFFFFu);     // sentinel = leave POR
    const uint32_t prog_rx_init = parse_word((argc > 5) ? argv[5] : nullptr, 0xFFFFFFFFu);  // sentinel = leave POR
    const uint32_t wrap = (argc > 6) ? std::strtoul(argv[6], nullptr, 0) : 1u;

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
        "BH-ICRC-probe: dev %d core (%u,%u) phys (%u,%u)  RXQ=%u  prog_ctrl=0x%08x rx_init=0x%08x wrap=%u\n"
        "  While holding, fire a KNOWN frame:\n"
        "    sudo tt_rdma_bf3_send <if> 200 02:00:00:00:00:02 0x1af6 0x10 256 0x00CAFE42 0\n",
        device_id,
        (unsigned)eth_logical.x,
        (unsigned)eth_logical.y,
        (unsigned)eth_phys.x,
        (unsigned)eth_phys.y,
        TT_RDMA_RX_QUEUE,
        prog_ctrl,
        prog_rx_init,
        wrap);

    std::vector<uint32_t> zstats(12, 0u);
    cluster.write_core(device->id(), eth_phys, zstats, (uint32_t)stats_addr);

    Program program = CreateProgram();
    const EthernetConfig cfg{.noc = NOC::NOC_1, .processor = DataMovementProcessor::RISCV_1};
    const KernelHandle k =
        CreateKernel(program, "tt_metal/tt_rdma/bh0/kernels/bh_rdma_icrc_probe.cpp", eth_logical, cfg);
    SetRuntimeArgs(
        program,
        k,
        eth_logical,
        {(uint32_t)stats_addr, TT_RDMA_STOP_ADDR, rx_ring_addr, rx_ring_size, prog_ctrl, prog_rx_init, wrap});

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    std::cout << "BH-ICRC-probe: kernel up. Send frames now.\n";

    bool verdict_printed = false;
    for (int s = 0; s < hold_s; ++s) {
        auto st = cluster.read_core<uint32_t>(device->id(), eth_phys, (uint32_t)stats_addr, 12 * sizeof(uint32_t));
        std::printf(
            "  t=%2ds frames=%u ctrl_por=0x%08x rx_init_por=0x%08x tx_init_por=0x%08x ctrl_rb=0x%08x\n"
            "        rx_calc=0x%08x rx_recv=0x%08x sw_crc=0x%08x hdr_cksum=0x%08x bufptr=%u drop=%u iters=%u\n",
            s,
            st[0],
            st[1],
            st[2],
            st[3],
            st[4],
            st[5],
            st[6],
            st[7],
            st[8],
            st[9],
            st[10],
            st[11]);
        std::fflush(stdout);

        if (!verdict_printed && st[0] > 0) {  // a frame has been sampled
            const bool sw_ok =
                (st[8] == st[7]);  // sender's stamped cksum == our SW CRC (sanity: sender uses same poly)
            const bool hw_matches_sw = (st[5] == st[7]);
            const bool hw_matches_wire = (st[5] == st[8]);
            const bool hw_engaged = (st[5] != 0u || st[6] != 0u);
            std::printf(
                "  >>> VERDICT: sender_cksum==sw_crc:%s  HW_engaged:%s  rx_calc==sw_crc:%s  rx_calc==wire_cksum:%s\n",
                sw_ok ? "yes" : "NO",
                hw_engaged ? "yes" : "no(engine idle on 0x1AF6 raw)",
                hw_matches_sw ? "YES-OFFLOAD-VIABLE" : "no",
                hw_matches_wire ? "yes" : "no");
            verdict_printed = true;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    const std::vector<uint32_t> stop_val{1u};
    cluster.write_core(device->id(), eth_phys, stop_val, TT_RDMA_STOP_ADDR);
    distributed::Finish(cq);
    std::cout << "BH-ICRC-probe: done; clean shutdown." << std::endl;
    return 0;
}
