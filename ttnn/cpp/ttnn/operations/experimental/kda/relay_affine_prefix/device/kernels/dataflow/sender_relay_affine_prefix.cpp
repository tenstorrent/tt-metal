// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/chronology.hpp"

using namespace tt::tt_fabric;

constexpr uint32_t cb_stage = get_compile_time_arg_val(0);
constexpr uint32_t sp_rank = get_compile_time_arg_val(1);
constexpr uint32_t sp_size = get_compile_time_arg_val(2);
constexpr uint32_t local_rows = get_compile_time_arg_val(3);
constexpr uint32_t unit_bytes = get_compile_time_arg_val(4);        // one tile column of a head's state
constexpr uint32_t chunk_bytes = get_compile_time_arg_val(5);       // one fabric packet
constexpr uint32_t kt = get_compile_time_arg_val(6);                // tile columns per state
constexpr uint32_t slots = get_compile_time_arg_val(7);             // unit buffers
constexpr uint32_t ack_semaphore_id = get_compile_time_arg_val(8);  // on the head cores
constexpr auto start_args = TensorAccessorArgs<9>();
constexpr uint32_t chunks = (unit_bytes + chunk_bytes - 1) / chunk_bytes;
// One packet header per packet of each slot, out of this RISC's share of the pool.
static_assert(
    slots * chunks <= MEM_PACKET_HEADER_POOL_SIZE / MaxDMProcessorsPerCoreType / PACKET_HEADER_MAX_SIZE,
    "the sender's packet headers exceed the packet header pool");

namespace {

void route_one_hop(volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, uint16_t chip_id, uint16_t mesh_id) {
#if defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0)
    (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
#else
    (void)chip_id;
    (void)mesh_id;
    (void)fabric_set_unicast_route<false>(
        reinterpret_cast<volatile tt_l1_ptr LowLatencyPacketHeader*>(hdr), static_cast<uint16_t>(1));
#endif
}

}  // namespace

// Sender core (one per link): reads each finished tile column out of its head cores and forwards it into the
// successor's landing buffer (the first rank's final-state landing, from the last rank) over the fabric.
void kernel_main() {
    size_t arg = 0;
    const uint32_t start_address = get_arg_val<uint32_t>(arg++);
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg++));
    const uint16_t dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg++));
    const uint32_t eout_address = get_arg_val<uint32_t>(arg++);   // a head's finished state (same on every head core)
    const uint32_t ein_landing = get_arg_val<uint32_t>(arg++);    // the successor head core's entry-state storage
    const uint32_t final_landing = get_arg_val<uint32_t>(arg++);  // the first rank's final-state landing
    const uint32_t arrival_semaphore = get_arg_val<uint32_t>(arg++);
    const uint32_t final_semaphore = get_arg_val<uint32_t>(arg++);
    const uint32_t num_heads = get_arg_val<uint32_t>(arg++);  // entries: head core NoC x | y << 16
    const size_t head_args = arg;
    arg += num_heads;
    const size_t notify_args = arg;  // then each head's count semaphore id on this core
    arg += num_heads;

    cb_reserve_back(cb_stage, 1);
    const uint32_t stage = get_write_ptr(cb_stage);
    const auto start = TensorAccessor(start_args, start_address);
    noc_async_read(start.get_noc_addr(0), stage, sizeof(uint32_t));
    noc_async_read_barrier();
    const auto topology =
        kda_chronology::derive(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage)[0], sp_rank, sp_size, local_rows);
    const uint32_t position = (topology.rank + sp_size - topology.first_rank) % sp_size;
    const bool last = position == sp_size - 1;
    const uint32_t landing = last ? final_landing : ein_landing;
    const uint32_t landed_semaphore = last ? final_semaphore : arrival_semaphore;
    const uint32_t ack_semaphore = get_semaphore(ack_semaphore_id);

    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg);
    // The relay's unit is one tile column of a head's state (column-major storage): the successor computes that
    // column of its state as soon as it lands. Units go through `slots` buffers: unit u + slots - 1 is read out of its
    // head core while earlier units' packets are on the link, and a slot (and its packet headers) is reused only
    // after its packets have left L1.
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[slots][chunks];
    for (uint32_t b = 0; b < slots; ++b) {
        for (uint32_t c = 0; c < chunks; ++c) {
            hdrs[b][c] = PacketHeaderPool::allocate_header();
            route_one_hop(hdrs[b][c], dst_chip_id, dst_mesh_id);
        }
    }
    conn.open();
    const uint32_t units = num_heads * kt;
    const auto head_xy = [&](uint32_t u) { return get_arg_val<uint32_t>(head_args + u / kt); };
    // Head k's writer counts its finished columns into its count semaphore on this core.
    const auto ready = [&](uint32_t u) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                   get_semaphore(get_arg_val<uint32_t>(notify_args + u / kt)))[0] >= u % kt + 1;
    };
    uint32_t fetched = 0;  // units whose reads are issued
    const auto fetch = [&](uint32_t u) {
        const uint32_t xy = head_xy(u);
        noc_async_read(
            get_noc_addr(xy & 0xFFFF, xy >> 16, eout_address + (u % kt) * unit_bytes),
            stage + (u % slots) * unit_bytes,
            unit_bytes);
    };
    for (uint32_t u = 0; u < units; ++u) {
        // Issue every read the free slots allow, waiting only for unit u itself.
        while (fetched < units && fetched < u + slots && (fetched == u || ready(fetched))) {
            if (fetched == u) {
                while (!ready(u)) {
                }
            }
            if (fetched >= slots) {
                // Unit fetched - slots last used this slot: its packets must have left L1.
                noc_async_writes_flushed();
            }
            fetch(fetched);
            ++fetched;
        }
        noc_async_read_barrier();
        const uint32_t xy = head_xy(u);
        const uint32_t hx = xy & 0xFFFF;
        const uint32_t hy = xy >> 16;
        if (u % kt == kt - 1) {
            // The head's last column has left its head core: release the head.
            noc_semaphore_inc(get_noc_addr(hx, hy, ack_semaphore), 1);
        }
        const uint32_t buffer = stage + (u % slots) * unit_bytes;
        const uint32_t column = landing + (u % kt) * unit_bytes;
        for (uint32_t c = 0; c < chunks; ++c) {
            const uint32_t offset = c * chunk_bytes;
            const uint32_t bytes = offset + chunk_bytes <= unit_bytes ? chunk_bytes : unit_bytes - offset;
            const uint64_t dst = get_noc_addr(hx, hy, column + offset);
            volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[u % slots][c];
            if (c + 1 == chunks) {
                hdr->to_noc_fused_unicast_write_atomic_inc(
                    NocUnicastAtomicIncFusedCommandHeader{dst, get_noc_addr(hx, hy, landed_semaphore), 1, true}, bytes);
            } else {
                hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, bytes);
            }
            conn.wait_for_empty_write_slot();
            conn.send_current_slot_non_blocking(buffer + offset, bytes, reinterpret_cast<uint32_t>(hdr));
        }
    }
    noc_async_writes_flushed();
    conn.close();
    noc_async_atomic_barrier();
}
