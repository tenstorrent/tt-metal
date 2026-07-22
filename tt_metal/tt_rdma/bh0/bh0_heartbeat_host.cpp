// SPDX-License-Identifier: Apache-2.0
//
// BH.0 host loader — loads bh_rdma_heartbeat.cpp onto the SUBORDINATE erisc
// (RISC1) of an active, link-trained eth core and holds it resident so the
// coexistence gate can be observed. See docs/tt-rdma-v1/tt-rdma-bh-bf3-impl-plan.md.
//
// GATE: with this running, `port_status` on the chosen eth core stays UP for
// the hold window (bh-erisc-fpga/scripts/erisc_ports.sh) AND the heartbeat word
// at TT_RDMA_HB_ADDR advances (tt-exalens brxy). See README.md.
//
// GRACEFUL STOP (no reset needed): the kernel runs until we set the stop flag at
// TT_RDMA_STOP_ADDR. After the hold we write it, Finish() the queue (the kernel
// returns -> RISC0 go-loop reaps it -> RISC1 goes idle), then close cleanly. A
// never-returning busy-spin would pin RISC1 in an active power state until a
// chip reset; this avoids that entirely.
//
// Reconciled to this checkout's API: distributed::MeshDevice + MeshWorkload
// (pattern from tt_metal/programming_examples/hello_world_datamovement_kernel).
// Kernel is resolved relative to TT_METAL_HOME, so run with TT_METAL_HOME set to
// this repo root (see README). Build: self-contained CMake (find_package TT-Metalium).

#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include "impl/kernels/kernel.hpp"         // EthernetConfig + Eth (internal API; not in public host_api.hpp)
#include "impl/context/metal_context.hpp"  // MetalContext -> cluster().write_core (set the stop flag)

#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_l1_layout.h"  // TT_RDMA_HB_ADDR / TT_RDMA_STOP_ADDR

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
    std::cout << "BH.0: heartbeat @ 0x" << std::hex << TT_RDMA_HB_ADDR << ", stop-flag @ 0x" << TT_RDMA_STOP_ADDR
              << std::dec << ", spin=" << spin << ", hold=" << hold_s << "s\n";

    // RISC1 (subordinate) = the free data mover; base FW owns NOC0 on RISC0, so RISC1 uses NOC1.
    Program program = CreateProgram();
    // Default eth_mode (a dispatchable active-eth kernel) — NOT Eth::IDLE (an idle core isn't
    // dispatched -> the kernel never runs). NOC1 because base FW owns NOC0 on RISC0.
    const EthernetConfig cfg{
        .noc = NOC::NOC_1,
        .processor = DataMovementProcessor::RISCV_1,
    };
    const KernelHandle k =
        CreateKernel(program, "tt_metal/tt_rdma/bh0/kernels/bh_rdma_heartbeat.cpp", eth_logical, cfg);
    // num_beats=0 -> run until the stop flag; arg3 = stop-flag address (kernel clears it on entry).
    SetRuntimeArgs(program, k, eth_logical, {TT_RDMA_HB_ADDR, spin, /*num_beats=*/0u, TT_RDMA_STOP_ADDR});

    // Dispatch NON-BLOCKING so we can hold the kernel resident for the observation window.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);

    std::cout << "BH.0: kernel dispatched to RISC1. Observe from another terminal:\n"
              << "  erisc_ports.sh <X-Y>   (port_status stays UP)\n"
              << "  tt-exalens brxy <X-Y> 0x" << std::hex << TT_RDMA_HB_ADDR << std::dec << " 1   (heartbeat advances)"
              << std::endl;  // endl = flush

    std::this_thread::sleep_for(std::chrono::seconds(hold_s));

    // Graceful stop: write the stop flag in the eth core's L1. The kernel polls it and RETURNS,
    // so Finish() completes, the RISC0 go-loop reaps the kernel, and RISC1 returns to idle — no
    // chip reset required, and close_device() tears down cleanly (destructors below).
    const std::vector<uint32_t> stop_val{1u};
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        device->id(), device->ethernet_core_from_logical_core(eth_logical), stop_val, TT_RDMA_STOP_ADDR);
    std::cout << "BH.0: hold elapsed -> stop flag set; waiting for the kernel to finish..." << std::endl;

    distributed::Finish(cq);  // returns once the (now-terminating) kernel completes
    std::cout << "BH.0: kernel finished and reaped; RISC1 idle. Clean shutdown." << std::endl;
    return 0;  // MeshDevice destructor closes the device cleanly (no resident kernel to hang on)
}
