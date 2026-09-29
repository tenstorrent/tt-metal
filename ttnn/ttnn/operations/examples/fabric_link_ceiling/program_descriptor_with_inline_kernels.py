# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""fabric_link_ceiling — the simplest possible fabric stream, as one ttnn.generic_op.

Every chip in mesh row 0 is paired with the chip below it (row 1). On each link a single sender core
streams a ring of L1 slots, one full fabric packet per slot, into the same-address landing ring on
its peer. Nothing consumes the landing ring (it is simply overwritten), so the only work measured
is: build a packet, wait for a router slot, hand the packet to the router, and the link itself. The
last packet is a fused write + atomic increment, and the receiving core waits on that increment, so
the program ends only once every byte has landed.

VARIANTS (how the sender issues packets):
  flush_per_packet  one header, reused for every packet: payload write, then a flush-blocking header
                    write (the textbook idiom; each packet waits until it has left the core's L1).
  header_ring       a ring of pre-routed headers; each packet is issued non-blocking, and the core
                    flushes only when it wraps around to a header it is about to rewrite.

Directions: "uni" (row 0 sends, row 1 receives) or "bi" (both rows send and receive at once).

Placement: `cores` picks the logical core that serves each link (sender on the sending chip, landing ring
on the receiving chip), and `sender_noc` the NoC the sender writes packets to its router on. Together
they decide which NoC links the per-link streams share on their way to and from the Ethernet cores.
"""

import ttnn

VARIANTS = ("flush_per_packet", "header_ring")
DIRECTIONS = ("uni", "bi")
NUM_HEADERS = 8  # header_ring depth (headers in flight per sender)

_SENDER_SOURCE = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

using namespace tt::tt_fabric;

// Every send is one hop. 2D fabrics route by the destination's fabric node id; 1D fabrics route by
// hop count, the connection having already fixed the direction. ROUTING_MODE comes from the build.
inline void route_one_hop(volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, uint16_t chip_id, uint16_t mesh_id) {
#if defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0)
    (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
#else
    (void)chip_id;
    (void)mesh_id;
    (void)fabric_set_unicast_route<false>(
        reinterpret_cast<volatile tt_l1_ptr LowLatencyPacketHeader*>(hdr), static_cast<uint16_t>(1));
#endif
}

void kernel_main() {
    constexpr bool header_ring = get_compile_time_arg_val(0) != 0;
    constexpr uint32_t num_headers = header_ring ? get_compile_time_arg_val(1) : 1;
    constexpr uint32_t packet_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t slots = get_compile_time_arg_val(3);

    size_t a = 0;
    const uint32_t src_addr = get_arg_val<uint32_t>(a++);  // my source ring (L1)
    const uint32_t dst_addr = get_arg_val<uint32_t>(a++);  // peer's landing ring (same L1 address)
    const uint32_t peer_x = get_arg_val<uint32_t>(a++);    // peer core, NoC coords on the peer chip
    const uint32_t peer_y = get_arg_val<uint32_t>(a++);
    const uint32_t sem_addr = get_arg_val<uint32_t>(a++);  // peer's "stream done" semaphore
    const uint32_t num_packets = get_arg_val<uint32_t>(a++);
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint16_t dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);  // appended last

    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[num_headers];
    for (uint32_t h = 0; h < num_headers; ++h) {
        hdrs[h] = PacketHeaderPool::allocate_header();
        route_one_hop(hdrs[h], dst_chip_id, dst_mesh_id);
    }
    const uint64_t done_noc = get_noc_addr(peer_x, peer_y, sem_addr);
    conn.open();

    uint32_t h = 0;
    for (uint32_t i = 0; i < num_packets; ++i) {
        const uint32_t slot = i % slots;
        const uint32_t src = src_addr + slot * packet_bytes;
        const uint64_t dst = get_noc_addr(peer_x, peer_y, dst_addr + slot * packet_bytes);
        if constexpr (header_ring) {
            if (h == num_headers) {
                noc_async_writes_flushed();  // every header's previous send has left L1
                h = 0;
            }
        }
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[header_ring ? h++ : 0];
        if (i + 1 < num_packets) {
            hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, packet_bytes);
        } else {
            hdr->to_noc_fused_unicast_write_atomic_inc(
                NocUnicastAtomicIncFusedCommandHeader{dst, done_noc, 1, true}, packet_bytes);
        }
        conn.wait_for_empty_write_slot();
        if constexpr (header_ring) {
            conn.send_current_slot_non_blocking(src, packet_bytes, reinterpret_cast<uint32_t>(hdr));
        } else {
            conn.send_payload_without_header_non_blocking_from_address(src, packet_bytes);
            conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdr), sizeof(PACKET_HEADER_TYPE));
        }
    }
    noc_async_writes_flushed();
    conn.close();
}
"""

_RECEIVER_SOURCE = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t sem_addr = get_arg_val<uint32_t>(0);
    volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    noc_semaphore_wait_min(done, 1);                             // the last packet has landed
    noc_semaphore_inc(get_noc_addr(sem_addr), static_cast<uint32_t>(-1));  // re-arm for the next launch
    noc_async_atomic_barrier();
}
"""


NOC0 = ttnn.NOC.RISCV_0_default  # routes +X, then +Y
NOC1 = ttnn.NOC.RISCV_1_default  # routes -Y, then -X


def link_cores(num_links):
    """Default placement: one core per link, side by side at logical (l, 0)."""
    return [ttnn.CoreCoord(l, 0) for l in range(num_links)]


def ring_memory_config(cores, slots, packet_bytes):
    """L1 ring of `slots` packets on each link core: uint32 row-major, one packet per row."""
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (slots, packet_bytes // 4), ttnn.ShardOrientation.ROW_MAJOR),
    )


def create_mesh_program_descriptor(
    mesh_device,
    src,
    dst,
    sem_addr,
    *,
    variant,
    direction,
    cores,
    slots,
    packet_bytes,
    packets_per_link,
    sender_noc=NOC1,
):
    assert variant in VARIANTS and direction in DIRECTIONS
    rows, cols = tuple(mesh_device.shape)
    assert rows == 2, "pairs chips along mesh axis 0: needs exactly 2 rows"
    num_links = len(cores)
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    virt = [mesh_device.worker_core_from_logical_core(c) for c in cores]  # same on every chip
    src_addr, dst_addr = int(src.buffer_address()), int(dst.buffer_address())

    mesh_desc = ttnn.MeshProgramDescriptor()
    for r in range(rows):
        for c in range(cols):
            me = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(r, c))
            peer = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(1 - r, c))
            sends = direction == "bi" or r == 0
            receives = direction == "bi" or r == 1
            program = ttnn.ProgramDescriptor()
            kernels = []
            if sends:
                links = ttnn.get_forwarding_link_indices(me, peer)
                assert len(links) >= num_links, f"only {len(links)} links between {me} and {peer}"
                rt = ttnn.RuntimeArgs()
                for l, core in enumerate(cores):
                    args = [
                        src_addr,
                        dst_addr,
                        virt[l].x,
                        virt[l].y,
                        sem_addr,
                        packets_per_link,
                        int(peer.mesh_id),
                        int(peer.chip_id),
                    ]
                    args += list(ttnn.setup_fabric_connection(me, peer, links[l], program, core))
                    rt[core.x][core.y] = args
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=_SENDER_SOURCE,
                        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                        core_ranges=core_set,
                        compile_time_args=[int(variant == "header_ring"), NUM_HEADERS, packet_bytes, slots],
                        runtime_args=rt,
                        config=ttnn.DataMovementConfigDescriptor(
                            processor=ttnn.DataMovementProcessor.RISCV_0, noc=sender_noc
                        ),
                    )
                )
            if receives:
                rt = ttnn.RuntimeArgs()
                for core in cores:
                    rt[core.x][core.y] = [sem_addr]
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=_RECEIVER_SOURCE,
                        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                        core_ranges=core_set,
                        compile_time_args=[],
                        runtime_args=rt,
                        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1),
                    )
                )
            program.kernels = kernels
            mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = program
    return mesh_desc


def fabric_link_ceiling(mesh_device, src, dst, sem_addr, **kw):
    """One dispatch: stream packets_per_link full packets over each link (one core per entry of `cores`)."""
    return ttnn.generic_op([src, dst], create_mesh_program_descriptor(mesh_device, src, dst, sem_addr, **kw))
