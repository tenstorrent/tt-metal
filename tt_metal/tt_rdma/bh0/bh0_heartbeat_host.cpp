// SPDX-License-Identifier: Apache-2.0
//
// BH.0 host loader — loads bh_rdma_heartbeat.cpp onto the SUBORDINATE erisc
// (RISC1) of an active, link-trained eth core and holds it resident so the
// coexistence gate can be observed. See docs/tt-rdma-v1/tt-rdma-bh-bf3-impl-plan.md.
//
// GATE: with this running, `port_status` on the chosen eth core stays UP for
// 10 min (bh-erisc-fpga/scripts/erisc_ports.sh) AND the heartbeat word at
// TT_RDMA_RCB_ADDR advances (tt-exalens brxy). See README.md.
//
// Reconciled to this checkout's API: distributed::MeshDevice + MeshWorkload
// (pattern from tt_metal/programming_examples/hello_world_datamovement_kernel).
// Kernel is resolved relative to TT_METAL_HOME, so run with TT_METAL_HOME set to
// this repo root (see README). Build: self-contained CMake (find_package TT-Metalium).

#include <chrono>
#include <iostream>
#include <thread>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include "impl/kernels/kernel.hpp"  // EthernetConfig + Eth (internal API; not in public host_api.hpp)

#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_l1_layout.h"  // TT_RDMA_RCB_ADDR (heartbeat slot)

int main(int argc, char** argv) {
    using namespace tt;
    using namespace tt::tt_metal;

    // argv: [device_id] [active_eth_core_index] [spin_per_beat] [hold_seconds]
    const int device_id = (argc > 1) ? std::atoi(argv[1]) : 0;
    const size_t eth_idx = (argc > 2) ? std::atoi(argv[2]) : 0;
    const uint32_t spin = (argc > 3) ? std::strtoul(argv[3], nullptr, 0) : 200000u;
    const int hold_s = (argc > 4) ? std::atoi(argv[4]) : 600;

    // 1x1 mesh on the requested device.
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
    IDevice* device = mesh_device->get_devices()[0];  // underlying device for eth-core queries

    // --- 2. Pin the eth core (was *active.begin() in the skeleton) ---
    // get_active_ethernet_cores returns LOGICAL eth cores; pick one by index (argv[2]).
    const auto active = device->get_active_ethernet_cores(/*skip_reserved=*/true);
    std::vector<CoreCoord> cores(active.begin(), active.end());
    TT_FATAL(!cores.empty(), "no active ethernet cores on device {}", device_id);
    TT_FATAL(eth_idx < cores.size(), "eth_idx {} >= {} active cores", eth_idx, cores.size());
    const CoreCoord eth_logical = cores[eth_idx];
    // Print BOTH the logical and the physical/virtual coord so you know which
    // erisc_ports.sh / tt-exalens NOC coord (X-Y) to monitor for this link.
    const CoreCoord eth_phys = device->ethernet_core_from_logical_core(eth_logical);
    std::cout << "BH.0: device " << device_id << " eth core logical=(" << eth_logical.x << "," << eth_logical.y
              << ")  physical/NOC=(" << eth_phys.x << "," << eth_phys.y << ")  -- monitor this coord\n";
    std::cout << "BH.0: heartbeat @ 0x" << std::hex << TT_RDMA_RCB_ADDR << std::dec << ", spin=" << spin
              << ", hold=" << hold_s << "s\n";

    // RISC1 (subordinate) = the free data mover; base FW owns NOC0 on RISC0, so RISC1 uses NOC1.
    Program program = CreateProgram();
    const EthernetConfig cfg{
        .eth_mode = Eth::IDLE,  // resident compute kernel, not a tunneling/dispatch link
        .noc = NOC::NOC_1,
        .processor = DataMovementProcessor::RISCV_1,
    };
    const KernelHandle k =
        CreateKernel(program, "tt_metal/tt_rdma/bh0/kernels/bh_rdma_heartbeat.cpp", eth_logical, cfg);
    SetRuntimeArgs(program, k, eth_logical, {TT_RDMA_RCB_ADDR, spin, /*num_beats=*/0u});  // 0 = persistent

    // Dispatch NON-BLOCKING (kernel is persistent -> do NOT Finish()).
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);

    std::cout << "BH.0: kernel dispatched to RISC1. Observe from another terminal:\n"
              << "  erisc_ports.sh <X-Y>   (port_status stays UP)\n"
              << "  tt-exalens brxy <X-Y> 0x" << std::hex << TT_RDMA_RCB_ADDR << std::dec
              << " 1   (heartbeat advances)\n";

    std::this_thread::sleep_for(std::chrono::seconds(hold_s));

    // The kernel is still running (persistent). Stop it by resetting the chip
    // (sudo reboot, or tt-smi -r --eth_train_skip). We intentionally do NOT
    // Finish()/close() cleanly here — a never-returning kernel can't be reaped.
    std::cout << "BH.0: hold elapsed. Kernel still resident; reset the chip to stop.\n";
    return 0;
}
