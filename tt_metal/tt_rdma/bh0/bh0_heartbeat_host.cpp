// SPDX-License-Identifier: Apache-2.0
//
// BH.0 host loader — loads bh_rdma_heartbeat.cpp onto the SUBORDINATE erisc
// (RISC1) of an active, link-trained eth core and holds it resident so the
// coexistence gate can be observed. See docs/tt-rdma-v1/tt-rdma-bh-bf3-impl-plan.md.
//
// GATE: with this running, `port_status` on the chosen eth core stays UP for
// 10 min (read via bh-erisc-fpga/scripts/erisc_ports.sh) AND the heartbeat word
// at TT_RDMA_RCB_ADDR advances (read via tt-exalens brxy). See README.md.
//
// NOTE: this is a SKELETON. The tt-metal host API churns across versions
// (IDevice vs MeshDevice, command_queue() vs mesh CQ). Before building, copy the
// exact device-open + program-launch boilerplate from a shipped eth test on this
// checkout — tests/tt_metal/tt_metal/deployment/eth/test_eth_data_integrity_dram.cpp
// (CreateDevice / get_active_ethernet_cores(true) / CreateProgram / CreateKernel /
// SetRuntimeArgs / EnqueueProgram) — and reconcile the calls below to it.

#include <chrono>
#include <thread>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/kernel_types.hpp>

#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_l1_layout.h"  // TT_RDMA_RCB_ADDR (heartbeat slot)

using namespace tt::tt_metal;

int main() {
    constexpr int kDeviceId = 0;
    constexpr uint32_t kSpinPerBeat = 200000u;  // pace; tune so brxy can see the counter move
    constexpr auto kHoldFor = std::chrono::minutes(10);

    IDevice* device = CreateDevice(kDeviceId);

    // Pick a trained ACTIVE eth core to watch. TODO(BH.0): select the specific core
    // whose link you are monitoring with erisc_ports.sh (e.g. an inter-chip Cage-C core).
    // Starting on a live-link core IS the coexistence test; if that is too risky for the
    // first run, use an active core with a benign partner and confirm it stays UP.
    const auto active = device->get_active_ethernet_cores(/*skip_reserved=*/true);
    TT_FATAL(!active.empty(), "no active ethernet cores");
    const CoreCoord eth_core = *active.begin();  // TODO: pin the exact core

    Program program = CreateProgram();

    // RISC1 (subordinate) is the free data mover. NOC MUST follow the processor:
    // base FW uses NOC0 on RISC0, so RISC1 uses NOC1 (matches the device_print helper
    // `.noc = static_cast<NOC>(processor)` and active_erisc's "ERISC1 -> NOC_1" rule).
    const EthernetConfig cfg{
        .eth_mode = Eth::IDLE,  // not a tunneling/dispatch link; a resident compute kernel
        .noc = NOC::NOC_1,
        .processor = DataMovementProcessor::RISCV_1,
    };

    const KernelHandle k = CreateKernel(program, "tt_metal/tt_rdma/bh0/kernels/bh_rdma_heartbeat.cpp", eth_core, cfg);

    SetRuntimeArgs(program, k, eth_core, {TT_RDMA_RCB_ADDR, kSpinPerBeat});

    // Dispatch NON-BLOCKING: the kernel is persistent (never returns), so do NOT Finish().
    CommandQueue& cq = device->command_queue();
    EnqueueProgram(cq, program, /*blocking=*/false);

    // Hold resident while you observe from another terminal:
    //   bh-erisc-fpga/scripts/erisc_ports.sh <core>            # port_status stays UP
    //   tt-exalens ... brxy <core> <TT_RDMA_RCB_ADDR> 1        # heartbeat advances
    std::this_thread::sleep_for(kHoldFor);

    // The kernel is still running (persistent). To stop it, reset the chip
    // (sudo reboot, or `tt-smi -r --eth_train_skip`). Do NOT expect a clean
    // CloseDevice() to reap a never-returning kernel.
    return 0;
}
