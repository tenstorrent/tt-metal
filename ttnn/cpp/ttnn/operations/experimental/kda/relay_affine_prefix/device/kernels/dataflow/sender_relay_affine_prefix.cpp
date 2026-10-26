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
constexpr uint32_t column_bytes = get_compile_time_arg_val(4);      // one tile column of a head's state
constexpr uint32_t packet_bytes = get_compile_time_arg_val(5);      // one fabric packet, whole tiles
constexpr uint32_t kt = get_compile_time_arg_val(6);                // tile columns per state
constexpr uint32_t slots = get_compile_time_arg_val(7);             // staged tile columns
constexpr uint32_t ack_semaphore_id = get_compile_time_arg_val(8);  // on the head cores
constexpr auto start_args = TensorAccessorArgs<9>();
constexpr uint32_t tile_bytes = column_bytes / kt;  // FP32 tiles
constexpr uint32_t column_packets = (column_bytes + packet_bytes - 1) / packet_bytes;
constexpr uint32_t headers = NUM_PACKET_HEADERS / MaxDMProcessorsPerCoreType;  // this RISC's share of the pool
static_assert(slots > 0 && slots < 16, "one read transaction id per staged column");
static_assert(packet_bytes % tile_bytes == 0, "packets carry whole tiles");
static_assert(headers >= 2, "one header in flight while the next piece's is filled");

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

// One read transaction id per packet of the staging ring when there are enough, else one per staged column.
constexpr uint32_t read_trid(uint32_t piece) {
    if constexpr (slots * column_packets < 16) {
        return 1 + piece % (slots * column_packets);
    } else {
        return 1 + (piece / column_packets) % slots;
    }
}

constexpr uint32_t piece_offset(uint32_t piece) { return piece % column_packets * packet_bytes; }

constexpr uint32_t piece_bytes(uint32_t piece) {
    return piece_offset(piece) + packet_bytes <= column_bytes ? packet_bytes : column_bytes - piece_offset(piece);
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

    // Open the fabric connection while the chronology word is in flight: on the first rank the relay starts here.
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[headers];
    for (uint32_t i = 0; i < headers; ++i) {
        hdrs[i] = PacketHeaderPool::allocate_header();
        route_one_hop(hdrs[i], dst_chip_id, dst_mesh_id);
    }
    conn.open();

    noc_async_read_barrier();
    const auto topology =
        kda_chronology::derive(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage)[0], sp_rank, sp_size, local_rows);
    const uint32_t position = (topology.rank + sp_size - topology.first_rank) % sp_size;
    const bool last = position == sp_size - 1;
    const uint32_t landing = last ? final_landing : ein_landing;
    const uint32_t landed_semaphore = last ? final_semaphore : arrival_semaphore;
    const uint32_t ack_semaphore = get_semaphore(ack_semaphore_id);

    // A head's state is kt tile columns stored column-major, at the same address on its head core and (its landing)
    // on the successor's. Each column goes out as column_packets packets ("pieces"). A piece's tiles are read out of
    // the head core as soon as their column is packed, into a ring of `slots` column buffers, and the piece is sent as
    // soon as its own read lands. A column's last piece counts the column into the successor head core's arrival
    // semaphore, so the successor computes the column as soon as it lands.
    const uint32_t pieces = num_heads * kt * column_packets;
    const auto head_xy = [&](uint32_t column) { return get_arg_val<uint32_t>(head_args + column / kt); };
    // Head k's writer counts its packed columns into its count semaphore on this core.
    const auto ready = [&](uint32_t piece) {
        const uint32_t column = piece / column_packets;
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                   get_semaphore(get_arg_val<uint32_t>(notify_args + column / kt)))[0] >= column % kt + 1;
    };
    uint32_t fetched = 0;         // pieces whose reads are issued
    uint32_t sent = 0;            // pieces sent
    uint32_t flushed_pieces = 0;  // pieces that have left L1 (their headers and buffers are free)
    const auto flush = [&]() {
        noc_async_writes_flushed();
        flushed_pieces = sent;
    };
    const auto fetch = [&]() {
        const uint32_t column = fetched / column_packets;
        if (column >= slots + flushed_pieces / column_packets) {
            // Column - slots last used this buffer: its pieces must have left L1.
            flush();
        }
        const uint32_t xy = head_xy(column);
        const uint32_t offset = column % kt * column_bytes + piece_offset(fetched);
        noc_async_read_set_trid(read_trid(fetched));
        noc_async_read(
            get_noc_addr(xy & 0xFFFF, xy >> 16, eout_address + offset),
            stage + column % slots * column_bytes + piece_offset(fetched),
            piece_bytes(fetched));
        ++fetched;
    };
    for (; sent < pieces; ++sent) {
        // Issue every read the free buffers allow, waiting only for this piece's.
        while (fetched < pieces && fetched / column_packets < sent / column_packets + slots &&
               (fetched == sent || ready(fetched))) {
            if (fetched == sent) {
                while (!ready(sent)) {
                }
            }
            fetch();
        }
        const uint32_t column = sent / column_packets;
        const uint32_t xy = head_xy(column);
        const uint32_t hx = xy & 0xFFFF;
        const uint32_t hy = xy >> 16;
        const bool column_end = sent % column_packets == column_packets - 1;
        noc_async_read_barrier_with_trid(read_trid(sent));
        if (column_end && column % kt == kt - 1) {
            // The head's last column has left its head core: release the head.
            noc_semaphore_inc(get_noc_addr(hx, hy, ack_semaphore), 1);
        }
        if (sent >= headers + flushed_pieces) {
            flush();
        }
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[sent % headers];
        const uint64_t dst = get_noc_addr(hx, hy, landing + column % kt * column_bytes + piece_offset(sent));
        if (column_end) {
            hdr->to_noc_fused_unicast_write_atomic_inc(
                NocUnicastAtomicIncFusedCommandHeader{dst, get_noc_addr(hx, hy, landed_semaphore), 1, true},
                piece_bytes(sent));
        } else {
            hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, piece_bytes(sent));
        }
        conn.wait_for_empty_write_slot();
        conn.send_current_slot_non_blocking(
            stage + column % slots * column_bytes + piece_offset(sent),
            piece_bytes(sent),
            reinterpret_cast<uint32_t>(hdr));
    }
    noc_async_read_set_trid(0);
    noc_async_writes_flushed();
    conn.close();
    noc_async_atomic_barrier();
}
