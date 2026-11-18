// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/chronology.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/persistent_counter.hpp"

using namespace tt::tt_fabric;
using namespace tt::tt_fabric::linear::experimental;

constexpr uint32_t cb_stage = get_compile_time_arg_val(0);
constexpr uint32_t cb_eout = get_compile_time_arg_val(1);  // at the same address on the head cores
constexpr uint32_t sp_size = get_compile_time_arg_val(2);
constexpr uint32_t local_rows = get_compile_time_arg_val(3);
constexpr uint32_t unit_bytes = get_compile_time_arg_val(4);        // one tile column of a head's state
constexpr uint32_t chunk_bytes = get_compile_time_arg_val(5);       // one fabric packet
constexpr uint32_t kt = get_compile_time_arg_val(6);                // tile columns per state
constexpr uint32_t slots = get_compile_time_arg_val(7);             // unit buffers
constexpr uint32_t ack_semaphore_id = get_compile_time_arg_val(8);  // on the head cores
constexpr auto start_args = TensorAccessorArgs<9>();
constexpr uint32_t chunks = (unit_bytes + chunk_bytes - 1) / chunk_bytes;
constexpr auto carry_args = TensorAccessorArgs<start_args.next_compile_time_args_offset()>();
// Slab (slab_kernel_args): the carry output is the slab, written at entry `index * slab_scale + slab_offset` on every
// rank; nothing past slab_index_limit. The index lands in the stage's word slab_index_word.
constexpr uint32_t slab_ct = carry_args.next_compile_time_args_offset();
constexpr bool slab = get_compile_time_arg_val(slab_ct);
constexpr uint32_t slab_index_limit = get_compile_time_arg_val(slab_ct + 1);
constexpr uint32_t slab_scale = get_compile_time_arg_val(slab_ct + 2);
constexpr uint32_t slab_offset = get_compile_time_arg_val(slab_ct + 3);
constexpr uint32_t entry_tiles = get_compile_time_arg_val(slab_ct + 4);
constexpr uint32_t slab_index_word = get_compile_time_arg_val(slab_ct + 5);
constexpr auto slab_index_args = TensorAccessorArgs<slab_ct + 6>();
// One packet header per packet of each slot, out of this RISC's share of the pool.
static_assert(
    slots * chunks <= MEM_PACKET_HEADER_POOL_SIZE / MaxDMProcessorsPerCoreType / PACKET_HEADER_MAX_SIZE,
    "the sender's packet headers exceed the packet header pool");

// Sender core: the relay's sender of a link reads each finished tile column out of its head cores and forwards it into
// the successor's landing buffer (the first rank's final-state landing, from the last rank) over the fabric. On the
// chronologically first rank, every sender then multicasts the final carry its head cores form, carry_range ranks
// along its one connection, into every rank's carry output; together a link's senders reach every other rank once.
void kernel_main() {
    size_t arg = 0;
    const uint32_t start_address = get_common_arg_val<uint32_t>(0);
    // The landing buffer's slots (same on every head core): the successor's entry state, the first rank's final state.
    const uint32_t ein_landing = get_common_arg_val<uint32_t>(1);
    const uint32_t final_landing = ein_landing + kt * unit_bytes;
    const uint32_t eout_address = get_write_ptr(cb_eout);
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg++));
    const uint16_t dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg++));
    const uint32_t arrival_semaphore = get_arg_val<uint32_t>(arg++);
    const uint32_t final_semaphore = get_arg_val<uint32_t>(arg++);
    const uint32_t sp_rank = get_arg_val<uint32_t>(arg++);
    // 1D routing: hops to the successor, 1 forward, or sp_size - 1 backward from the last rank of a line.
    const uint32_t successor_hops = get_arg_val<uint32_t>(arg++);
    const uint32_t num_heads = get_arg_val<uint32_t>(arg++);  // entries: head core NoC x | y << 16
    const size_t head_args = arg;
    arg += num_heads;
    const size_t notify_args = arg;  // then each head's count semaphore id on this core
    arg += num_heads;
    const uint32_t carry_semaphore = get_arg_val<uint32_t>(arg++);
    const uint32_t drain_x = get_arg_val<uint32_t>(arg++);
    const uint32_t drain_y = get_arg_val<uint32_t>(arg++);
    const uint32_t links = get_arg_val<uint32_t>(arg++);  // a carry increment arrives per link
    const uint32_t link = get_arg_val<uint32_t>(arg++);   // this sender's heads are link, link + links, ...
    // The relay's own sender, else the link's second carry sender, which sends towards the predecessor.
    const bool relay = get_arg_val<uint32_t>(arg++);
    const uint8_t carry_range = get_arg_val<uint32_t>(arg++);
    // A second carry sender without ranks behind its rank holds no connection.
    const uint32_t connections = get_arg_val<uint32_t>(arg++);

    cb_reserve_back(cb_stage, 1);
    const uint32_t stage = get_write_ptr(cb_stage);
    const auto start = TensorAccessor(start_args, start_address);
    noc_async_read(start.get_noc_addr(0), stage, sizeof(uint32_t));
    if constexpr (slab) {
        const auto slab_index = TensorAccessor(slab_index_args, get_common_arg_val<uint32_t>(3));
        noc_async_read(slab_index.get_noc_addr(0), stage + slab_index_word * sizeof(uint32_t), sizeof(uint32_t));
    }
    noc_async_read_barrier();
    const uint32_t index = slab ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage)[slab_index_word] : 0;
    const bool writes_carry = !slab || index < slab_index_limit;
    const uint32_t carry_base = writes_carry ? (index * slab_scale + slab_offset) * entry_tiles : 0;
    const auto topology =
        kda_chronology::derive(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage)[0], sp_rank, sp_size, local_rows);
    const uint32_t position = (topology.rank + sp_size - topology.first_rank) % sp_size;
    const bool last = position == sp_size - 1;
    const uint32_t landing = last ? final_landing : ein_landing;
    const uint32_t landed_semaphore = last ? final_semaphore : arrival_semaphore;
    const uint32_t ack_semaphore = get_semaphore(ack_semaphore_id);

    RoutingPlaneConnectionManager fabric;
    open_connections(fabric, connections, arg);
    auto* to_successor = relay ? &fabric.get(0).sender : nullptr;
    // The relay's unit is one tile column of a head's state (column-major storage): the successor computes that
    // column of its state as soon as it lands. Units go through `slots` buffers: unit u + slots - 1 is read out of its
    // head core while earlier units' packets are on the link, and a slot (and its packet headers) is reused only
    // after its packets have left L1.
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[slots][chunks];
    for (uint32_t b = 0; b < slots; ++b) {
        for (uint32_t c = 0; c < chunks; ++c) {
            hdrs[b][c] = PacketHeaderPool::allocate_header();
#if defined(FABRIC_2D)
            // 2D routing: the hand-off from the last rank to the first crosses the mesh when the fabric does not wrap
            // the SP axis.
            (void)fabric_set_unicast_route(hdrs[b][c], dst_chip_id, dst_mesh_id);
#else
            (void)dst_chip_id;
            (void)dst_mesh_id;
#endif
        }
    }
    const uint32_t units = num_heads * kt;
    const auto head_xy = [&](uint32_t u) { return get_arg_val<uint32_t>(head_args + u / kt); };
    // Head k's writer counts its finished columns into its count semaphore on this core. Blackhole's data-movement
    // RISCs cache L1: invalidate before each poll, as noc_semaphore_wait_min does.
    const auto ready = [&](uint32_t u) {
        invalidate_l1_cache();
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                   get_semaphore(get_arg_val<uint32_t>(notify_args + u / kt)))[0] >= u % kt + 1;
    };
    // The link's second carry sender takes no part in the relay.
    const uint32_t relay_units = relay ? units : 0;
    uint32_t fetched = 0;  // units whose reads are issued
    const auto fetch = [&](uint32_t u) {
        const uint32_t xy = head_xy(u);
        noc_async_read(
            get_noc_addr(xy & 0xFFFF, xy >> 16, eout_address + (u % kt) * unit_bytes),
            stage + (u % slots) * unit_bytes,
            unit_bytes);
    };
    for (uint32_t u = 0; u < relay_units; ++u) {
        if (fetched == u) {
            while (!ready(u)) {
            }
            fetch(u);
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
            // 1D routing: the connection leaves towards the successor's side of the line (2D ignores the hops).
            if (c + 1 == chunks) {
                fabric_unicast_noc_fused_unicast_with_atomic_inc(
                    to_successor,
                    hdr,
                    buffer + offset,
                    bytes,
                    NocUnicastAtomicIncFusedCommandHeader{dst, get_noc_addr(hx, hy, landed_semaphore), 1, true},
                    successor_hops);
            } else {
                fabric_unicast_noc_unicast_write(
                    to_successor, hdr, buffer + offset, bytes, NocUnicastCommandHeader{dst}, successor_hops);
            }
        }
        // While unit u's packets leave L1, read the next ready units into the other slots: every earlier unit's packets
        // have left (flushed below in its own iteration), so only u's slot is still in use. Flushing before these reads
        // instead would serialize each unit's read behind the previous unit's send.
        while (fetched < relay_units && fetched < u + slots && ready(fetched)) {
            fetch(fetched);
            ++fetched;
        }
        noc_async_writes_flushed();
    }

    if (position == 0 && carry_range != 0) {
        // The relay's packets have left, so its headers carry the final state now: one per tile of a column, then
        // the arrival increment, all on this sender's one connection.
        constexpr uint32_t tile_bytes = unit_bytes / kt;
        static_assert(slots * chunks >= kt + 1, "the carry reuses the relay's packet headers");
        static_assert(slots >= 2, "the carry reads the next column into a second stage slot");
        uint8_t range[1] = {carry_range};
        volatile tt_l1_ptr PACKET_HEADER_TYPE* carry_hdrs[kt + 1];
        for (uint32_t i = 0; i <= kt; ++i) {
            carry_hdrs[i] = hdrs[i / chunks][i % chunks];
            fabric_set_mcast_route(fabric, carry_hdrs[i], range, 0);
            if (i < kt) {
                fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
                    carry_hdrs[i], 1, range[0], nullptr, tile_bytes);
            } else {
                fabric_multicast_noc_unicast_atomic_inc_set_state<
                    UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
                    carry_hdrs[i], 1, range[0], NocUnicastAtomicIncCommandHeader{0, 1u});
            }
        }
        auto* sender = &fabric.get(0).sender;
        const auto carry = TensorAccessor(carry_args, get_common_arg_val<uint32_t>(2));
        // The head's writer counts compute's carry columns, on the relay's sender after the relay's kt; a ready
        // column is read into stage slot u % 2 while the previous one is on the fabric.
        const uint32_t counted = relay ? kt : 0;
        const auto fetch_carry = [&](uint32_t u) {
            volatile tt_l1_ptr uint32_t* count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(get_arg_val<uint32_t>(notify_args + u / kt)));
            do {
                invalidate_l1_cache();
            } while (count[0] < counted + u % kt + 1);
            const uint32_t xy = head_xy(u);
            noc_async_read(
                get_noc_addr(xy & 0xFFFF, xy >> 16, eout_address + (u % kt) * unit_bytes),
                stage + (u % 2) * unit_bytes,
                unit_bytes);
        };
        // A link without heads (more links than heads) sends only the arrival increment.
        if (units != 0) {
            fetch_carry(0);
        }
        for (uint32_t u = 0; u < units; ++u) {
            noc_async_read_barrier();
            const uint32_t column = u % kt;
            if (column == kt - 1) {
                // The head's last carry column has left its head core: this sender is done with the head.
                const uint32_t xy = head_xy(u);
                noc_semaphore_inc(get_noc_addr(xy & 0xFFFF, xy >> 16, ack_semaphore), 1);
            }
            const uint32_t head = link + u / kt * links;
            const uint32_t buffer = stage + (u % 2) * unit_bytes;
            // The column's tiles are rows 0..kt-1 of tile column `column` of the row-major carry; a slab index past
            // the slabs writes nothing, and only the arrival increment goes out.
            for (uint32_t r = 0; r < (writes_carry ? kt : 0); ++r) {
                fabric_multicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                    sender,
                    carry_hdrs[r],
                    buffer + r * tile_bytes,
                    NocUnicastCommandHeader{linear::addrgen_detail::get_noc_address(
                        carry, carry_base + head * kt * kt + r * kt + column, 0)});
            }
            if (u + 1 < units) {
                fetch_carry(u + 1);
            }
            // The headers are reused for the next column once its packets have left L1.
            noc_async_writes_flushed();
        }
        // Flushed after this sender's columns, the increment tells each rank it reaches that its share has landed.
        fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            sender,
            carry_hdrs[kt],
            NocUnicastAtomicIncCommandHeader{safe_get_noc_addr(drain_x, drain_y, carry_semaphore, 0), 0});
        noc_async_writes_flushed();
    }
    close_connections(fabric);
    // The relay's sender of the first link drains the carry's arrivals.
    if (position != 0 && relay && link == 0) {
        kda_counter::wait_and_consume(carry_semaphore, links);
    }
    noc_async_atomic_barrier();
}
