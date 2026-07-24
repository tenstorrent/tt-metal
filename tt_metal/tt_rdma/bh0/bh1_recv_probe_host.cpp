// SPDX-License-Identifier: Apache-2.0
//
// BH.1 / M-1b host loader — loads bh_rdma_recv_probe.cpp onto RISC1 of an external eth core, then
// polls the on-core RX diagnostics live while a BlueField-3 sends 0x1af6 frames at the TT rail.
// Confirms the receive path: BF3 -> TT eth RXQ2 (raw) -> L1 -> read back here.
//
//   bh1_recv_probe [device_id] [eth_idx|"ext"] [hold_s]
//   ext -> first EXTERNAL/NIC rail (read the FW tag).  Then on the host, send from the BF3 netdev:
//     sudo <raw-send of an 0x1af6 unicast frame>   (see bf3_send.py)

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

#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_l1_layout.h"

int main(int argc, char** argv) {
    using namespace tt;
    using namespace tt::tt_metal;

    const int device_id = (argc > 1) ? std::atoi(argv[1]) : 1;
    const char* eth_sel = (argc > 2) ? argv[2] : "ext";
    const bool want_ext = (std::strcmp(eth_sel, "ext") == 0);
    const size_t eth_idx = want_ext ? 0 : (size_t)std::atoi(eth_sel);
    const int hold_s = (argc > 3) ? std::atoi(argv[3]) : 30;

    constexpr uint64_t kEthStatusSpare0 = 0x7CC00u + 0x10u;
    constexpr uint32_t kExternalMagic = 0x1AF6E471u;
    const uint64_t diag_addr = TT_RDMA_RCB_ADDR + 0x40u;  // 8 diag words, clear of HB(+0)/STOP(+4)

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
        TT_FATAL(found, "no EXTERNAL/NIC rail found on device {}", device_id);
    } else {
        TT_FATAL(eth_idx < cores.size(), "eth_idx {} >= {} active cores", eth_idx, cores.size());
        eth_logical = cores[eth_idx];
    }
    const CoreCoord eth_phys = device->ethernet_core_from_logical_core(eth_logical);
    std::cout << "BH.1-RX: device " << device_id << " eth core logical=(" << eth_logical.x << "," << eth_logical.y
              << ")  physical/NOC=(" << eth_phys.x << "," << eth_phys.y << "), RXQ=" << TT_RDMA_RX_QUEUE
              << ", rx ring @ 0x" << std::hex << TT_RDMA_RX_RING_ADDR << ", diag @ 0x" << diag_addr << std::dec
              << std::endl;

    Program program = CreateProgram();
    const EthernetConfig cfg{.noc = NOC::NOC_1, .processor = DataMovementProcessor::RISCV_1};
    const KernelHandle k =
        CreateKernel(program, "tt_metal/tt_rdma/bh0/kernels/bh_rdma_recv_probe.cpp", eth_logical, cfg);
    SetRuntimeArgs(
        program,
        k,
        eth_logical,
        {TT_RDMA_HB_ADDR, TT_RDMA_STOP_ADDR, (uint32_t)diag_addr, TT_RDMA_RX_RING_ADDR, TT_RDMA_RX_RING_SIZE});

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    std::cout << "BH.1-RX: kernel dispatched. Now send 0x1af6 frames from the BF3 netdev. Polling...\n";

    // Poll + print the on-core RX diagnostics each second.
    for (int s = 0; s < hold_s; ++s) {
        auto d = cluster.read_core<uint32_t>(device->id(), eth_phys, diag_addr, 8 * sizeof(uint32_t));
        std::printf(
            "  t=%2ds  rxq2_words=%u  drop[q2/q0/q1]=%u/%u/%u  rxbuf[0..3]=%08x %08x %08x %08x\n",
            s,
            d[0],
            d[1],
            d[2],
            d[3],
            d[4],
            d[5],
            d[6],
            d[7]);
        std::fflush(stdout);
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // graceful stop
    const std::vector<uint32_t> stop_val{1u};
    cluster.write_core(device->id(), eth_phys, stop_val, TT_RDMA_STOP_ADDR);
    distributed::Finish(cq);
    std::cout << "BH.1-RX: done; kernel reaped, RISC1 idle. Clean shutdown." << std::endl;
    return 0;
}
