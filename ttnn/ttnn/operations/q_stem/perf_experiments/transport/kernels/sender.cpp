// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// q_stem transport bench: port sender (RISCV_0 / NoC1). A reconstruction of q_stem_port_sender.cpp's send loop
// (ready fence + per-row / per-destination / per-head segments of hw tiles in packets of <= pkt_tiles tiles, the last
// packet of a destination core's head group fused with an atomic inc of that core's arrival counter,
// send_current_slot_non_blocking, noc_async_writes_flushed every BATCH segments, header rotation per destination),
// with the knobs the transport bake-off varies:
//
//   per destination route   0xFF = the fabric routing table (fabric_set_unicast_route, the op's only route), else an
//                           explicit 2-hop route (route_via): the antipodal quarter in the NON-default direction
//                           (balanced split).
//   BLOCKING                1: header flushed blocking per packet (send_payload_without_header + flush_blocking)
//   DIRECT_SRC              1: no port reader / CB handshake: packets leave straight from the source shard
//   INC_EVERY               rows per arrival increment of a (core, source) (op: 1; >= rows: one at the end)
//   POSTED                  1: posted NoC writes for the payload / header into the router slot
//
// Raw fabric API, as the op (helper note in q_stem_port_sender.cpp: kernel_lib/ccl FabricStream builds 1-D
// hop-count routes only; this needs 2D node routes and, for the balanced split, an explicit direction).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

using namespace tt::tt_fabric;

namespace {
inline void route_table(volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, uint16_t chip_id, uint16_t mesh_id) {
    (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
}

// Explicit 2-hop unicast: first hop = this port's connection (direction D1), second hop in direction d2 from the
// neighbour, whatever the routing table says (the table routes NS-first; this also expresses the EW->NS turn).
// route_buffer[hop] is read by the router that RECEIVED the packet: hop 0 at the neighbour (forward d2), hop 1 at the
// destination (the local-write command for a packet that travelled d2 = single_hop_route_cmd_by_direction[d2]).
// A turn onto an east/west router (from a north/south one) resets hop_index to branch_<dir>_offset: set it to 1.
inline void route_via(
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, uint16_t chip_id, uint16_t mesh_id, uint32_t d2) {
    using MF = RoutingFieldsConstants::Mesh;
    constexpr uint8_t fwd[4] = {MF::FORWARD_EAST, MF::FORWARD_WEST, MF::FORWARD_NORTH, MF::FORWARD_SOUTH};
    hdr->dst_start_node_id = ((uint32_t)mesh_id << 16) | (uint32_t)chip_id;
    hdr->mcast_params_64 = 0;
    hdr->is_mcast_active = 0;
    constexpr uint32_t n = FabricHeaderConfig::MESH_ROUTE_BUFFER_SIZE;
    for (uint32_t i = 0; i < n; ++i) {
        hdr->route_buffer[i] = MF::NOOP;
    }
    hdr->route_buffer[0] = fwd[d2];
    hdr->route_buffer[1] = single_hop_route_cmd_by_direction[d2];
    hdr->routing_fields.value = 0;
    hdr->routing_fields.branch_east_offset = 1;
    hdr->routing_fields.branch_west_offset = 1;
}
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_port_stage = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t pkt_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t batch = get_compile_time_arg_val(3);
    constexpr uint32_t hw = get_compile_time_arg_val(4);
    constexpr uint32_t max_headers = get_compile_time_arg_val(5);
    constexpr uint32_t BLOCKING = get_compile_time_arg_val(6);
    constexpr uint32_t DIRECT_SRC = get_compile_time_arg_val(7);
    constexpr uint32_t INC_EVERY = get_compile_time_arg_val(8);  // rows per arrival increment (op: 1)
    constexpr uint32_t POSTED = get_compile_time_arg_val(9);
    constexpr uint32_t seg_bytes = hw * tile_bytes;
    constexpr uint32_t HEAD_WORDS = 5;  // packed core, inc, slot base, row stride, jump
    constexpr uint32_t DEST_WORDS = 4;  // chip, mesh, second-hop direction (0xFF table), unused

    size_t a = 0;
    const uint32_t landing_addr = get_arg_val<uint32_t>(a++);
    const uint32_t arrive_addr = get_arg_val<uint32_t>(a++);
    const uint32_t ready_addr = get_arg_val<uint32_t>(a++);
    const uint32_t rows = get_arg_val<uint32_t>(a++);
    const uint32_t ts = get_arg_val<uint32_t>(a++);
    const uint32_t hpd = get_arg_val<uint32_t>(a++);  // headers per destination (rotation)
    const uint32_t src_addr = get_arg_val<uint32_t>(a++);
    const uint32_t n_src = get_arg_val<uint32_t>(a++);
    const uint32_t n_dest = get_arg_val<uint32_t>(a++);
    const uint32_t dests_idx = a;
    a += DEST_WORDS * n_dest;
    const uint32_t n_entries = get_arg_val<uint32_t>(a++);
    const uint32_t entries_idx = a;  // per entry: dest idx, n_heads, heads...
    for (uint32_t i = 0; i < n_entries; ++i) {
        const uint32_t nh = get_arg_val<uint32_t>(a + 1);
        a += 2 + nh * HEAD_WORDS;
    }
    const uint32_t n_ready_out = get_arg_val<uint32_t>(a++);
    const uint32_t ready_idx = a;
    a += 3 * n_ready_out;
    const uint32_t n_ready_wait = get_arg_val<uint32_t>(a++);
    const bool has_conn = get_arg_val<uint32_t>(a++) != 0;
    if (!has_conn) {
        return;
    }

    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[max_headers];
    for (uint32_t k = 0; k < n_dest; ++k) {
        const uint32_t t = dests_idx + DEST_WORDS * k;
        const uint16_t chip = static_cast<uint16_t>(get_arg_val<uint32_t>(t));
        const uint16_t mesh = static_cast<uint16_t>(get_arg_val<uint32_t>(t + 1));
        const uint32_t ew = get_arg_val<uint32_t>(t + 2);
        for (uint32_t r = 0; r < hpd; ++r) {
            hdrs[k * hpd + r] = PacketHeaderPool::allocate_header();
            if (ew == 0xFF) {
                route_table(hdrs[k * hpd + r], chip, mesh);
            } else {
                route_via(hdrs[k * hpd + r], chip, mesh, ew);
            }
        }
    }
    conn.open();

    uint32_t dirty = 0;
    uint32_t rot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    auto flushed = [&]() {
        noc_async_writes_flushed();
        if constexpr (POSTED) {
            noc_async_posted_writes_flushed();
        }
    };
    auto next_header = [&](uint32_t k) {
        const uint32_t idx = k * hpd + rot[k];
        rot[k] = rot[k] + 1 == hpd ? 0 : rot[k] + 1;
        if (dirty & (1u << idx)) {
            flushed();
            dirty = 0;
        }
        dirty |= 1u << idx;
        return hdrs[idx];
    };

    {
        for (uint32_t k = 0; k < n_ready_out; ++k) {
            const uint32_t t = ready_idx + 3 * k;
            volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = next_header(get_arg_val<uint32_t>(t));
            hdr->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{
                get_noc_addr(get_arg_val<uint32_t>(t + 1), get_arg_val<uint32_t>(t + 2), ready_addr, 0), 1, true});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdr), sizeof(PACKET_HEADER_TYPE));
        }
        if (n_ready_wait > 0) {
            volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
            noc_semaphore_wait_min(ready, n_ready_wait);
            noc_semaphore_inc(get_noc_addr(ready_addr), 0u - n_ready_wait);
            noc_async_atomic_barrier();
        }
    }

    uint32_t total = 0;
    {
        uint32_t e = entries_idx;
        for (uint32_t i = 0; i < n_entries; ++i) {
            const uint32_t nh = get_arg_val<uint32_t>(e + 1);
            total += nh;
            e += 2 + nh * HEAD_WORDS;
        }
        total *= rows;
    }

    uint32_t sent = 0, unflushed = 0, src_slot = 0;
    for (uint32_t row = 0; row < rows; ++row) {
        uint32_t e = entries_idx;
        for (uint32_t i = 0; i < n_entries; ++i) {
            const uint32_t k = get_arg_val<uint32_t>(e);
            const uint32_t nh = get_arg_val<uint32_t>(e + 1);
            const uint32_t heads = e + 2;
            e += 2 + nh * HEAD_WORDS;
            for (uint32_t h = 0; h < nh; ++h) {
                const uint32_t t = heads + h * HEAD_WORDS;
                const uint32_t dc = get_arg_val<uint32_t>(t);
                const uint32_t inc_row = get_arg_val<uint32_t>(t + 1);
                // increment every INC_EVERY rows (and on the last row) by the rows since the previous one
                const uint32_t inc_n = (row + 1) % INC_EVERY == 0 ? INC_EVERY : (row + 1 == rows ? (row + 1) % INC_EVERY : 0);
                const uint32_t inc = inc_row * inc_n;
                const uint32_t slot = get_arg_val<uint32_t>(t + 2) + row * get_arg_val<uint32_t>(t + 3) +
                                      (row >= ts ? get_arg_val<uint32_t>(t + 4) : 0);
                const uint32_t dx = dc >> 16, dy = dc & 0xFFFF;
                const uint64_t arrive_noc = get_noc_addr(dx, dy, arrive_addr, 0);
                uint32_t src;
                if constexpr (DIRECT_SRC) {
                    src = src_addr + src_slot * seg_bytes;
                    src_slot = src_slot + 1 == n_src ? 0 : src_slot + 1;
                } else {
                    cb_wait_front(cb_port_stage, hw * (unflushed + 1));
                    src = get_read_ptr(cb_port_stage) + unflushed * seg_bytes;
                }
                for (uint32_t off = 0; off < hw; off += pkt_tiles) {
                    const uint32_t n = (hw - off) < pkt_tiles ? hw - off : pkt_tiles;
                    const uint64_t dst = get_noc_addr(dx, dy, landing_addr + (slot + off) * tile_bytes, 0);
                    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = next_header(k);
                    if (off + n == hw && inc > 0) {
                        hdr->to_noc_fused_unicast_write_atomic_inc(
                            NocUnicastAtomicIncFusedCommandHeader{dst, arrive_noc, inc, true}, n * tile_bytes);
                    } else {
                        hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, n * tile_bytes);
                    }
                    conn.wait_for_empty_write_slot();
                    if constexpr (BLOCKING) {
                        conn.send_payload_without_header_non_blocking_from_address(
                            src + off * tile_bytes, n * tile_bytes);
                        conn.send_payload_flush_blocking_from_address(
                            reinterpret_cast<uint32_t>(hdr), sizeof(PACKET_HEADER_TYPE));
                    } else {
                        conn.template send_current_slot_non_blocking<POSTED != 0>(
                            src + off * tile_bytes, n * tile_bytes, reinterpret_cast<uint32_t>(hdr));
                    }
                }
                ++sent;
                if constexpr (DIRECT_SRC) {
                    // the source never changes: nothing to release; headers are flushed by next_header on reuse
                } else {
                    if (++unflushed == batch || sent == total) {
                        flushed();
                        dirty = 0;
                        cb_pop_front(cb_port_stage, hw * unflushed);
                        unflushed = 0;
                    }
                }
            }
        }
    }
    flushed();
    conn.close();
}
