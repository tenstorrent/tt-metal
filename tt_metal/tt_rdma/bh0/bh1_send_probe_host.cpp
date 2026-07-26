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
// 0x69B1EDCC). If that fails the frame builder is wrong — abort before the wire.

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
                  << ", expected 0x69B1EDCC)\n";
        return false;
    }
    std::cout << "golden self-test PASS: §7.1 SEND header == TT_GOLDEN_SEND_HDR (crc32 ok)\n";
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
        // Read the FW external-endpoint tag per core (eth_status.spare[0] @ 0x7CC10) — the same flag
        // UMD reads. A tagged core is a NIC/EXTERNAL rail (target these for a BF3 PROBE); untagged is
        // a TT-TT link. This both labels the BF3 rail AND confirms the FW is setting the tag.
        constexpr uint64_t kEthStatusSpare0 = 0x7CC00u + 0x10u;  // BOOT_RESULTS_ADDR + offsetof(eth_status,spare)
        constexpr uint32_t kExternalMagic = 0x1AF6E471u;
        std::cout << "device " << dev_id << ": " << cs.size() << " active ethernet core(s):\n";
        for (size_t i = 0; i < cs.size(); ++i) {
            const CoreCoord p = dev->ethernet_core_from_logical_core(cs[i]);
            auto spare = tt::tt_metal::MetalContext::instance().get_cluster().read_core<uint32_t>(
                dev->id(), p, kEthStatusSpare0, sizeof(uint32_t));
            const bool ext = (!spare.empty() && spare[0] == kExternalMagic);
            std::cout << "  eth_idx " << i << ": logical=(" << cs[i].x << "," << cs[i].y << ")  physical/NOC=(" << p.x
                      << "," << p.y << ")  " << (ext ? "[EXTERNAL/NIC rail <- target this]" : "[TT-TT]") << "\n";
        }
        std::cout.flush();
        return 0;
    }

    // argv: [device_id] [eth_idx|"ext"] [dst_mac] [count] [spin] [hold_s]
    //   eth_idx = index into active cores, OR "ext" = ALL external/NIC rails at once (one workload).
    const int device_id = (argc > 1) ? std::atoi(argv[1]) : 0;
    const char* eth_sel = (argc > 2) ? argv[2] : "0";
    const bool all_ext = (std::strcmp(eth_sel, "ext") == 0);
    const size_t eth_idx = all_ext ? 0 : (size_t)std::atoi(eth_sel);
    const char* dst_mac_s = (argc > 3) ? argv[3] : "ff:ff:ff:ff:ff:ff";           // broadcast default
    const uint32_t count = (argc > 4) ? std::strtoul(argv[4], nullptr, 0) : 16u;  // bounded by default
    const uint32_t spin = (argc > 5) ? std::strtoul(argv[5], nullptr, 0) : 2000000u;
    const int hold_s = (argc > 6) ? std::atoi(argv[6]) : 10;  // only used when count==0
    // payload bytes after the 32-B header; wire frame = 14 (L2) + 32 + payload. Default 32 (tiny).
    const uint32_t payload_len = (argc > 7) ? std::strtoul(argv[7], nullptr, 0) : 32u;
    // burst_bytes>0 => BURST mode: one big raw transfer/command from the 128 KB WQE region, HW
    // auto-splits into max_pkt-sized frames (raw-mode bandwidth path). max_pkt default 4080.
    const uint32_t burst_bytes = (argc > 8) ? std::strtoul(argv[8], nullptr, 0) : 0u;
    const uint32_t max_pkt = (argc > 9) ? std::strtoul(argv[9], nullptr, 0) : 4080u;

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

    // Build the target core list. "ext" => every external/NIC rail (read the FW tag, same as --list);
    // otherwise the single core at eth_idx.
    constexpr uint64_t kEthStatusSpare0 = 0x7CC00u + 0x10u;  // BOOT_RESULTS_ADDR + offsetof(eth_status,spare)
    constexpr uint32_t kExternalMagic = 0x1AF6E471u;
    std::vector<CoreCoord> targets;
    if (all_ext) {
        for (const auto& c : cores) {
            const CoreCoord p = device->ethernet_core_from_logical_core(c);
            auto spare = tt::tt_metal::MetalContext::instance().get_cluster().read_core<uint32_t>(
                device->id(), p, kEthStatusSpare0, sizeof(uint32_t));
            if (!spare.empty() && spare[0] == kExternalMagic) {
                targets.push_back(c);
            }
        }
        TT_FATAL(!targets.empty(), "no EXTERNAL/NIC rails found on device {} (need the tagged-endpoint FW)", device_id);
    } else {
        TT_FATAL(eth_idx < cores.size(), "eth_idx {} >= {} active cores", eth_idx, cores.size());
        targets.push_back(cores[eth_idx]);
    }

    if (all_ext && count == 0) {
        // Multi-core persistent + graceful-stop-all is fine, but bounded is the safe default for a
        // quick "run both" eval (self-terminates, no orphan risk). Warn but allow.
        std::cout << "BH.1: note — ext mode with count=0 (persistent); will hold " << hold_s
                  << "s then stop-flag ALL rails.\n";
    }

    // One program, one kernel per target core (each eth core has its own L1 + eth-SS regs, so the
    // per-core TXPKT/queue config and HB/STOP slots don't collide).
    Program program = CreateProgram();
    const EthernetConfig cfg{.noc = NOC::NOC_1, .processor = DataMovementProcessor::RISCV_1};
    for (const CoreCoord& t : targets) {
        const CoreCoord tp = device->ethernet_core_from_logical_core(t);
        std::cout << "BH.1: target eth core logical=(" << t.x << "," << t.y << ")  physical/NOC=(" << tp.x << ","
                  << tp.y << ")\n";
        const KernelHandle k = CreateKernel(program, "tt_metal/tt_rdma/bh0/kernels/bh_rdma_send_probe.cpp", t, cfg);
        SetRuntimeArgs(
            program,
            k,
            t,
            {TT_RDMA_HB_ADDR, TT_RDMA_STOP_ADDR, count, spin, dmac_hi, dmac_lo, payload_len, burst_bytes, max_pkt});
    }
    std::cout << "BH.1: PROBE -> " << dst_mac_s << " ethertype 0x" << std::hex << TT_RDMA_ETHERTYPE << std::dec
              << ", count=" << count << " (0=until stop), queue=" << TT_RDMA_TX_QUEUE << ", rails=" << targets.size()
              << "\n";

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    std::cout << "BH.1: " << targets.size()
              << " kernel(s) dispatched to RISC1. On the BF3:  "
                 "tcpdump -i <ttport> ether proto 0x1af6 -xx"
              << std::endl;

    if (count == 0) {
        // Persistent: hold for observation, then graceful-stop EVERY rail (BH.0 pattern).
        std::this_thread::sleep_for(std::chrono::seconds(hold_s));
        const std::vector<uint32_t> stop_val{1u};
        for (const CoreCoord& t : targets) {
            tt::tt_metal::MetalContext::instance().get_cluster().write_core(
                device->id(), device->ethernet_core_from_logical_core(t), stop_val, TT_RDMA_STOP_ADDR);
        }
        std::cout << "BH.1: hold elapsed -> stop flag set on all rails." << std::endl;
    }
    distributed::Finish(cq);  // bounded: waits for `count` frames on every rail; persistent: waits for the stops
    std::cout << "BH.1: done; kernel(s) reaped, RISC1 idle. Clean shutdown." << std::endl;
    return 0;
}
