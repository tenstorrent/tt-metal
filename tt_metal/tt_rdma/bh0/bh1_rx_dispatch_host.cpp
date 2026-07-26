// SPDX-License-Identifier: Apache-2.0
//
// BH.2-RX host — loads bh_rdma_rx_dispatch.cpp on RISC1 of an external rail, programs one MR
// (slot 0), and polls the on-core dispatch stats while a BlueField-3 sends TT-RDMA frames.
//
// Flow: this host sets MR[0] = { base_noc = TX_BUF1 scratch (local L1), length, rkey, REMOTE_WRITE },
// clears the scratch, dispatches the kernel, then (externally) fire:
//   sudo tt_rdma_bf3_send enp193s0f0np0 <count> 02:00:00:00:00:02 0x1af6 0x10 <plen> 0x00CAFE42 0
// The kernel parses each frame, dispatches by opcode, and for WRITE lands payload at TX_BUF1+roff.
// This host prints per-opcode counts and reads TX_BUF1 back to confirm the WRITE payload ("TTWR"...).
//
//   bh1_rx_dispatch [device_id] [eth_idx|"ext"] [hold_s]

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
#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_l1_layout.h"

int main(int argc, char** argv) {
    using namespace tt;
    using namespace tt::tt_metal;

    const int device_id = (argc > 1) ? std::atoi(argv[1]) : 1;
    const char* eth_sel = (argc > 2) ? argv[2] : "ext";
    const bool want_ext = (std::strcmp(eth_sel, "ext") == 0);
    const size_t eth_idx = want_ext ? 0 : (size_t)std::atoi(eth_sel);
    const int hold_s = (argc > 3) ? std::atoi(argv[3]) : 20;
    const uint32_t wrap = (argc > 4) ? std::strtoul(argv[4], nullptr, 0) : 1u;     // 1 = BUF_WRAP streaming (Stage 2a)
    const uint32_t bigring = (argc > 5) ? std::strtoul(argv[5], nullptr, 0) : 1u;  // 1 = 128KB RX ring (Stage 2)
    // Stage 2b noc_target: 0 = local L1 copy (Stage 1/2a); 1 = loopback (own eth L1 via NoC);
    // 2 = Tensix worker (0,0) L1 via NoC (off-core, the real RDMA case).
    const uint32_t noc_target = (argc > 6) ? std::strtoul(argv[6], nullptr, 0) : 0u;
    const uint32_t crc_check = (argc > 7) ? std::strtoul(argv[7], nullptr, 0) : 1u;  // 1 = validate CRC-32 (Phase 1.1)
    const uint32_t rx_ring_addr = bigring ? TT_RDMA_RX_RING_BIG_ADDR : TT_RDMA_RX_RING_ADDR;
    const uint32_t rx_ring_size = bigring ? TT_RDMA_RX_RING_BIG_SIZE : TT_RDMA_RX_RING_SIZE;

    // MR slot 0: rkey top byte = slot = 0. The BF3 sender must use this exact rkey.
    const uint32_t kRkey = 0x00CAFE42u;               // (slot=0)<<24 | (rand=0xCAFE)<<8 | (gen=0x42)
    const uint32_t kMrTarget = TT_RDMA_TX_BUF1_ADDR;  // Stage-1 local L1 landing (TX bufs idle during RX)
    const uint32_t kMrLen = TT_RDMA_TX_BUF_BYTES;     // 4096

    constexpr uint64_t kEthStatusSpare0 = 0x7CC00u + 0x10u;
    constexpr uint32_t kExternalMagic = 0x1AF6E471u;
    const uint64_t stats_addr = TT_RDMA_DBG_ADDR;  // reuse the RCB dbg region (8 u32)

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
    std::cout << "BH.2-RX: dev " << device_id << " core (" << eth_logical.x << "," << eth_logical.y << ") phys ("
              << eth_phys.x << "," << eth_phys.y << ")  RXQ=" << TT_RDMA_RX_QUEUE << "  rx ring @ 0x" << std::hex
              << TT_RDMA_RX_RING_ADDR << "  MR[0].target @ 0x" << kMrTarget << "  rkey=0x" << kRkey << std::dec
              << "  wrap=" << wrap
              << "\n  BF3: sudo tt_rdma_bf3_send <if> <n> 02:00:00:00:00:02 0x1af6 0x10 <plen> 0x00CAFE42 0\n";

    // Resolve the WRITE landing target by noc_target mode.
    uint32_t noc_x = 0, noc_y = 0, noc_base = 0;  // noc_base==0 -> local L1 copy in the kernel
    CoreCoord verify_core = eth_phys;
    uint32_t verify_addr = kMrTarget;  // Stage 1/2a: local eth L1 (TX_BUF1)
    const char* mode = "local-copy";
    if (noc_target == 1) {  // loopback: eth core's own TX_BUF0 via the NoC (proves noc_async_write path)
        noc_x = (uint32_t)eth_phys.x;
        noc_y = (uint32_t)eth_phys.y;
        noc_base = TT_RDMA_TX_BUF0_ADDR;
        verify_core = eth_phys;
        verify_addr = TT_RDMA_TX_BUF0_ADDR;
        mode = "noc-loopback(own eth L1)";
    } else if (noc_target == 2) {  // off-core: a Tensix worker's L1 via the NoC (the real RDMA case)
        const CoreCoord w = device->worker_core_from_logical_core(CoreCoord{0, 0});
        noc_x = (uint32_t)w.x;
        noc_y = (uint32_t)w.y;
        noc_base = 0x20000u;  // safe L1 scratch on an idle worker
        verify_core = w;
        verify_addr = 0x20000u;
        mode = "noc-tensix(0,0 L1)";
    }
    std::printf(
        "  WRITE target mode=%s  noc=(%u,%u)+0x%x  verify @core(%u,%u):0x%x\n",
        mode,
        noc_x,
        noc_y,
        noc_base,
        (unsigned)verify_core.x,
        (unsigned)verify_core.y,
        verify_addr);

    // Program MR slot 0 (tt_rdma_mr_entry_t, 8 u32).
    std::vector<uint32_t> mr{kMrTarget, 0u, kMrLen, 0u, kRkey, TT_MR_REMOTE_WRITE, 0u, 0u};
    cluster.write_core(device->id(), eth_phys, mr, TT_RDMA_MR_TABLE_ADDR);
    // Clear the WRITE landing target (on its own core) + the stats region.
    std::vector<uint32_t> zeros(kMrLen / 4, 0u);
    cluster.write_core(device->id(), verify_core, zeros, verify_addr);
    std::vector<uint32_t> zstats(9, 0u);
    cluster.write_core(device->id(), eth_phys, zstats, (uint32_t)stats_addr);

    Program program = CreateProgram();
    const EthernetConfig cfg{.noc = NOC::NOC_1, .processor = DataMovementProcessor::RISCV_1};
    const KernelHandle k =
        CreateKernel(program, "tt_metal/tt_rdma/bh0/kernels/bh_rdma_rx_dispatch.cpp", eth_logical, cfg);
    SetRuntimeArgs(
        program,
        k,
        eth_logical,
        {TT_RDMA_HB_ADDR,
         TT_RDMA_STOP_ADDR,
         (uint32_t)stats_addr,
         rx_ring_addr,
         rx_ring_size,
         TT_RDMA_MR_TABLE_ADDR,
         wrap,
         noc_x,
         noc_y,
         noc_base,
         crc_check});

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    std::cout << "BH.2-RX: dispatch kernel up. Now send TT-RDMA frames from the BF3. Stats:\n";

    for (int s = 0; s < hold_s; ++s) {
        auto st = cluster.read_core<uint32_t>(device->id(), eth_phys, (uint32_t)stats_addr, 9 * sizeof(uint32_t));
        std::printf(
            "  t=%2ds  total=%u send=%u write=%u write_ok=%u unknown=%u bad=%u crc_err=%u last_op=0x%02x read_pos=%u\n",
            s,
            st[0],
            st[1],
            st[2],
            st[3],
            st[4],
            st[5],
            st[6],
            st[7],
            st[8]);
        std::fflush(stdout);
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // Verify the WRITE landing (on the resolved target core): first 8 bytes should be "TTWR" + 0x04..0x07.
    auto land = cluster.read_core<uint32_t>(device->id(), verify_core, verify_addr, 4 * sizeof(uint32_t));
    std::printf(
        "  WRITE landing @core(%u,%u):0x%x [0..3] = %08x %08x %08x %08x  (word0 'TTWR' = 0x52575454)\n",
        (unsigned)verify_core.x,
        (unsigned)verify_core.y,
        verify_addr,
        land[0],
        land[1],
        land[2],
        land[3]);

    const std::vector<uint32_t> stop_val{1u};
    cluster.write_core(device->id(), eth_phys, stop_val, TT_RDMA_STOP_ADDR);
    distributed::Finish(cq);
    std::cout << "BH.2-RX: done; clean shutdown." << std::endl;
    return 0;
}
