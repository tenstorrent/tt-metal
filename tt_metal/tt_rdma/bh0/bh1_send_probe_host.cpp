// SPDX-License-Identifier: Apache-2.0
//
// BH.1 / M-1a host loader — loads bh_rdma_send_probe.cpp onto RISC1 of an active
// eth core and drives it to emit TT-RDMA-v1 PROBE frames (ethertype 0x1AF6) to a
// BlueField-3 "tt" MAC. Confirm on the BF3:  tcpdump -i <ttport> ether proto 0x1af6 -xx
//
// Reuses BH.0's clean lifecycle: dispatch non-blocking; bounded count -> Finish;
// count=0 -> hold then set the stop flag so the kernel returns (no chip reset).
//
// Before touching hardware it runs a GOLDEN SELF-TEST: build the wire-protocol
// §7.1 SEND header and assert it is byte-for-byte TT_GOLDEN_SEND_HDR (cksum
// 0x7E9BA1C3). If that fails the frame builder is wrong — abort before the wire.

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include "impl/kernels/kernel.hpp"         // EthernetConfig (internal API)
#include "impl/context/metal_context.hpp"  // MetalContext -> cluster().write_core (stop flag)

#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_wire.h"
#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_hdr_build.h"
#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_l1_layout.h"

// Parse "aa:bb:cc:dd:ee:ff" -> 48-bit value packed as eth_send_raw expects
// (hi = value>>32, lo = value & 0xffffffff). Returns false on parse error.
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

// Build the spec §7.1 SEND header and check it matches the golden vector.
static bool golden_self_test() {
    tt_rdma_hdr_t h;
    tt_rdma_build_hdr(
        &h,
        TT_OP_SEND,
        TT_RDMA_VERSION,
        /*tag=*/0xCAFE,
        /*length=*/64,
        /*seq=*/0x1234,
        /*rkey=*/0,
        /*remote_offset=*/0,
        /*imm=*/0);
    if (std::memcmp(&h, TT_GOLDEN_SEND_HDR, TT_RDMA_HDR_BYTES) != 0) {
        std::cout << "GOLDEN SELF-TEST FAILED: header mismatch (cksum=0x" << std::hex << h.header_cksum << std::dec
                  << ", expected 0x7545F9A8)\n";
        return false;
    }
    std::cout << "golden self-test PASS: §7.1 SEND header == TT_GOLDEN_SEND_HDR (crc32c ok)\n";
    return true;
}

int main(int argc, char** argv) {
    using namespace tt;
    using namespace tt::tt_metal;

    // `--selftest`: validate the frame builder against the golden vectors and exit
    // (no device opened) — safe to run anytime, e.g. while the link is training.
    if (argc > 1 && std::strcmp(argv[1], "--selftest") == 0) {
        return golden_self_test() ? 0 : 2;
    }

    // `--list [device_id]`: open the device and print every ACTIVE ethernet core with
    // its logical + physical/NOC coord, so you can pick the eth_idx for the BF3 rail
    // (after the FW reports it PORT_UP, it joins this list — match it by NOC X-Y / speed).
    if (argc > 1 && std::strcmp(argv[1], "--list") == 0) {
        const int dev_id = (argc > 2) ? std::atoi(argv[2]) : 0;
        auto md = distributed::MeshDevice::create_unit_mesh(dev_id);
        IDevice* dev = md->get_devices()[0];
        const auto act = dev->get_active_ethernet_cores(/*skip_reserved=*/true);
        std::vector<CoreCoord> cs(act.begin(), act.end());
        std::cout << "device " << dev_id << ": " << cs.size() << " active ethernet core(s):\n";
        for (size_t i = 0; i < cs.size(); ++i) {
            const CoreCoord p = dev->ethernet_core_from_logical_core(cs[i]);
            std::cout << "  eth_idx " << i << ": logical=(" << cs[i].x << "," << cs[i].y << ")  physical/NOC=(" << p.x
                      << "," << p.y << ")\n";
        }
        std::cout.flush();
        return 0;
    }

    // argv: [device_id] [eth_idx] [dst_mac] [count] [spin] [hold_s]
    const int device_id = (argc > 1) ? std::atoi(argv[1]) : 0;
    const size_t eth_idx = (argc > 2) ? std::atoi(argv[2]) : 0;
    const char* dst_mac_s = (argc > 3) ? argv[3] : "ff:ff:ff:ff:ff:ff";           // broadcast default
    const uint32_t count = (argc > 4) ? std::strtoul(argv[4], nullptr, 0) : 16u;  // bounded by default
    const uint32_t spin = (argc > 5) ? std::strtoul(argv[5], nullptr, 0) : 2000000u;
    const int hold_s = (argc > 6) ? std::atoi(argv[6]) : 10;  // only used when count==0

    if (!golden_self_test()) {
        return 2;  // frame builder is wrong — do not put bad bytes on the wire
    }
    uint32_t dmac_hi = 0, dmac_lo = 0;
    if (!parse_mac(dst_mac_s, dmac_hi, dmac_lo)) {
        std::cout << "bad dst_mac '" << dst_mac_s << "' (want aa:bb:cc:dd:ee:ff)\n";
        return 2;
    }

    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
    IDevice* device = mesh_device->get_devices()[0];

    const auto active = device->get_active_ethernet_cores(/*skip_reserved=*/true);
    std::vector<CoreCoord> cores(active.begin(), active.end());
    TT_FATAL(!cores.empty(), "no active ethernet cores on device {}", device_id);
    TT_FATAL(eth_idx < cores.size(), "eth_idx {} >= {} active cores", eth_idx, cores.size());
    const CoreCoord eth_logical = cores[eth_idx];
    const CoreCoord eth_phys = device->ethernet_core_from_logical_core(eth_logical);
    std::cout << "BH.1: device " << device_id << " eth core logical=(" << eth_logical.x << "," << eth_logical.y
              << ")  physical/NOC=(" << eth_phys.x << "," << eth_phys.y << ")\n";
    std::cout << "BH.1: PROBE -> " << dst_mac_s << " ethertype 0x" << std::hex << TT_RDMA_ETHERTYPE << std::dec
              << ", count=" << count << " (0=until stop), queue=" << TT_RDMA_TX_QUEUE << "\n";

    Program program = CreateProgram();
    const EthernetConfig cfg{.noc = NOC::NOC_1, .processor = DataMovementProcessor::RISCV_1};
    const KernelHandle k =
        CreateKernel(program, "tt_metal/tt_rdma/bh0/kernels/bh_rdma_send_probe.cpp", eth_logical, cfg);
    SetRuntimeArgs(
        program,
        k,
        eth_logical,
        {TT_RDMA_HB_ADDR, TT_RDMA_STOP_ADDR, count, spin, dmac_hi, dmac_lo, /*payload_len=*/32u});

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    std::cout << "BH.1: kernel dispatched to RISC1. On the BF3:  tcpdump -i <ttport> ether proto 0x1af6 -xx"
              << std::endl;

    if (count == 0) {
        // Persistent: hold for observation, then graceful-stop (BH.0 pattern).
        std::this_thread::sleep_for(std::chrono::seconds(hold_s));
        const std::vector<uint32_t> stop_val{1u};
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            device->id(), device->ethernet_core_from_logical_core(eth_logical), stop_val, TT_RDMA_STOP_ADDR);
        std::cout << "BH.1: hold elapsed -> stop flag set." << std::endl;
    }
    distributed::Finish(cq);  // bounded: waits for `count` frames; persistent: waits for the stop
    std::cout << "BH.1: done; kernel reaped, RISC1 idle. Clean shutdown." << std::endl;
    return 0;
}
