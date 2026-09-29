// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// high_bw_all_reduce — port_bwd (RISCV_1 of the lane's port core). Owns the Fabric connection
// toward p-1 (absent on a line's first device).
//
// Start: zero the per-call control array, then raise `go` on the lane's reducers and port_fwd.
// relay_final_block, served per reducer: reducer r's blocks k = 0.. leave in r's block order
// (final slot (k mod FD) * W + r), but reducers are served round-robin by readiness — never in one
// global lane order, which would chain a local tail's slot reuse behind finals still being
// relayed from far tails of other slices (a full ring latency per block round). Per block role
// (high_bw_all_reduce_roles.hpp):
//   final present: tail -> final_ready[r] (local reducer r wrote the slot and the DRAM pages);
//   otherwise gsem_final_arrival[r] (written over Fabric by p+1's port_bwd, in r's block order);
//   non-head: wait gsem_final_credit[r] > k, forward the slot to p-1's port final landing;
//   non-tail: write the chunk's valid pages to output DRAM;
//   free the slot: final_freed[r] (port_fwd turns it into final credits for p+1) and final_egress
//   of reducer r (gates its next tail write into this slot).
// Every wait loop forwards each local reducer's landing-slot grants (partial_granted[r] delta) to
// p-1's port_fwd as gsem_partial_credit[r] (deadlock-freedom rule). All cross-device counters are
// per reducer, so one reducer's stall never withholds another's progress.
// No kernel_lib Fabric helper exists (recorded gap in op_design.md): raw EDM sender API.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "high_bw_all_reduce_roles.hpp"
#include "high_bw_all_reduce_route.hpp"
#include "high_bw_all_reduce_chunk_io.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

using namespace tt::tt_fabric;

void kernel_main() {
    constexpr uint32_t chunk_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t packet_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t final_depth = get_compile_time_arg_val(3);
    constexpr bool has_prev = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t go_sem_id = get_compile_time_arg_val(5);
    constexpr uint32_t final_egress_sem_id = get_compile_time_arg_val(6);
    constexpr uint32_t num_control_arrays = get_compile_time_arg_val(7);
    constexpr uint32_t CONTROL_WORD_STRIDE = get_compile_time_arg_val(8);  // bytes between control words
    // Data packet headers in flight per connection (host MAX_DATA_HEADERS; see port_fwd).
    constexpr uint32_t MAX_DATA_HEADERS = get_compile_time_arg_val(9);
    constexpr uint32_t pos = get_compile_time_arg_val(10);
    constexpr uint32_t group_size = get_compile_time_arg_val(11);
    constexpr uint32_t num_slices = get_compile_time_arg_val(12);
    constexpr uint32_t MAX_REDUCERS = get_compile_time_arg_val(13);  // host REDUCERS_PER_LANE (W <= it)
    // Landing-slot grants are coalesced: one credit packet per CREDIT_BATCH grants of a reducer (or
    // its last grant). A credit occupies a whole EDM packet slot, so per-chunk credits cost the link
    // ~1/packets_per_chunk of its rate (host credit_batch, from CREDIT_BATCH_CHUNKS; the host keeps the landing ring >=
    // CREDIT_BATCH, so a withheld grant always completes: see forward_partial_credits).
    constexpr uint32_t CREDIT_BATCH = get_compile_time_arg_val(14);
    constexpr bool bank_run_layout = get_compile_time_arg_val(15) != 0;  // host BANK_RUN_LAYOUT
    // Split ports (host SPLIT_PORTS, Refinement 4): port_fwd runs on its own core and this core's
    // other RISC runs port_drain, which writes the finals to output DRAM (on the other NoC) while
    // this RISC relays them over Fabric. A slot is then freed once it is forwarded AND drained.
    // false: port_fwd shares this core and this kernel writes DRAM itself (co-located ports).
    constexpr bool split_ports = get_compile_time_arg_val(16) != 0;
    constexpr auto output_args = TensorAccessorArgs<17>();
    using Io = ChunkIo<chunk_tiles, tile_bytes, bank_run_layout>;
    constexpr uint32_t packets_per_chunk = chunk_tiles / packet_tiles;
    constexpr uint32_t packet_bytes = packet_tiles * tile_bytes;
    constexpr uint32_t chunk_bytes = chunk_tiles * tile_bytes;
    constexpr uint32_t num_data_headers = packets_per_chunk < MAX_DATA_HEADERS ? packets_per_chunk : MAX_DATA_HEADERS;

    size_t a = 0;
    const uint32_t num_chunks = get_arg_val<uint32_t>(a++);
    const uint32_t num_reducers = get_arg_val<uint32_t>(a++);
    const uint32_t port_x = get_arg_val<uint32_t>(a++);  // this (backward-port) core
    const uint32_t port_y = get_arg_val<uint32_t>(a++);
    const uint32_t fwd_x = get_arg_val<uint32_t>(a++);  // the lane's port_fwd core (== this one if co-located)
    const uint32_t fwd_y = get_arg_val<uint32_t>(a++);
    const uint32_t drained_arr = get_arg_val<uint32_t>(a++);  // drained[r] (written by port_drain)
    const uint32_t ctrl_base = get_arg_val<uint32_t>(a++);
    const uint32_t final_addr = get_arg_val<uint32_t>(a++);
    const uint32_t final_ready_arr = get_arg_val<uint32_t>(a++);
    const uint32_t granted_arr = get_arg_val<uint32_t>(a++);
    const uint32_t final_freed_arr = get_arg_val<uint32_t>(a++);
    const uint32_t output_addr = get_arg_val<uint32_t>(a++);
    const uint32_t lane_start = get_arg_val<uint32_t>(a++);
    const uint32_t lane_tiles = get_arg_val<uint32_t>(a++);
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint16_t dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint32_t reducer_xy_idx = a;
    a += 2 * num_reducers;
    const uint32_t partial_credit_idx = a;  // per-reducer gsem_partial_credit[r] (on p-1's port)
    a += num_reducers;
    const uint32_t final_arrival_idx = a;  // per-reducer gsem_final_arrival[r]
    a += num_reducers;
    const uint32_t final_credit_idx = a;  // per-reducer gsem_final_credit[r]
    a += num_reducers;

    const auto output = TensorAccessor(output_args, output_addr, tile_bytes);
    auto word = [](uint32_t base, uint32_t r) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + r * CONTROL_WORD_STRIDE);
    };
    auto gsem = [](uint32_t addr) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr); };

    WorkerToFabricEdmSender connection;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* data_hdrs[num_data_headers];
    uint32_t hdr_idx = 0;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* credit_hdr = nullptr;
    {
        MaybeDeviceZoneScope("bwd_setup");  // connection open + control-array zeroing + go
        if constexpr (has_prev) {
            connection = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
            for (uint32_t h = 0; h < num_data_headers; ++h) {
                data_hdrs[h] = PacketHeaderPool::allocate_header();
                route_to_neighbor(data_hdrs[h], dst_chip_id, dst_mesh_id);
            }
            credit_hdr = PacketHeaderPool::allocate_header();
            route_to_neighbor(credit_hdr, dst_chip_id, dst_mesh_id);
            connection.open();
        }

        // Zero the control array (num_control_arrays * W words at a 16 B stride), then raise go.
        for (uint32_t i = 0; i < num_control_arrays * num_reducers; ++i) {
            word(ctrl_base, i)[0] = 0;
        }
        if constexpr (split_ports) {
            // port_fwd's words (staged[], final_freed[]) live on its own core: zero them there too.
            for (uint32_t i = 0; i < num_control_arrays * num_reducers; ++i) {
                noc_inline_dw_write(get_noc_addr(fwd_x, fwd_y, ctrl_base + i * CONTROL_WORD_STRIDE), 0);
            }
            noc_async_write_barrier();
        }
        // Read back so the stores have landed in L1 before any peer can increment.
        uint32_t sink = 0;
        for (uint32_t i = 0; i < num_control_arrays * num_reducers; ++i) {
            sink |= word(ctrl_base, i)[0];
        }
        (void)sink;
        const uint32_t go_sem_addr = get_semaphore(go_sem_id);
        for (uint32_t r = 0; r < num_reducers; ++r) {
            const uint32_t rx = get_arg_val<uint32_t>(reducer_xy_idx + 2 * r);
            const uint32_t ry = get_arg_val<uint32_t>(reducer_xy_idx + 2 * r + 1);
            noc_semaphore_inc(get_noc_addr(rx, ry, go_sem_addr), 1);
        }
        noc_semaphore_inc(get_noc_addr(fwd_x, fwd_y, go_sem_addr), 1);
        if constexpr (split_ports) {
            noc_semaphore_inc(get_noc_addr(port_x, port_y, go_sem_addr), 1);  // port_drain
        }
    }

    const uint32_t final_egress_sem_addr = get_semaphore(final_egress_sem_id);

    const ChainRoles<pos, group_size, num_slices> roles(num_reducers);
    uint32_t nb[MAX_REDUCERS];            // blocks of reducer r in this lane
    uint32_t next_k[MAX_REDUCERS];        // reducer r's next block to relay
    uint32_t freed_k[MAX_REDUCERS];       // reducer r's blocks whose final slot was freed
    uint32_t recv_total[MAX_REDUCERS];    // receive (non-head) blocks of reducer r = grants it issues
    uint32_t credits_sent[MAX_REDUCERS];  // grants of reducer r already forwarded upstream
    uint32_t tail_seq[MAX_REDUCERS];      // tail blocks of reducer r relayed so far
    uint32_t arrivals[MAX_REDUCERS];      // finals of reducer r landed over Fabric so far
    uint32_t pending = 0;
    for (uint32_t r = 0; r < num_reducers; ++r) {
        nb[r] = roles.blocks_of(r, num_chunks);
        next_k[r] = 0;
        freed_k[r] = 0;
        recv_total[r] = 0;
        credits_sent[r] = 0;
        tail_seq[r] = 0;
        arrivals[r] = 0;
        for (uint32_t k = 0; k < nb[r]; ++k) {
            recv_total[r] += roles.is_head(r, k, nb[r]) ? 0 : 1;
        }
        pending += nb[r];
    }

    uint64_t partial_credit_noc[MAX_REDUCERS];  // p-1's gsem_partial_credit[r]
    for (uint32_t r = 0; r < num_reducers; ++r) {
        partial_credit_noc[r] = get_noc_addr(fwd_x, fwd_y, get_arg_val<uint32_t>(partial_credit_idx + r));
    }

    MaybePerfAccum(acc_credit);     // landing-grant packets sent toward p-1 (slot wait + flush)
    MaybePerfAccum(acc_relay);      // one final's packets issued (includes slot waits) + DRAM issue
    MaybePerfAccum(acc_slot_wait);  // EDM sender slot not free (link / router back-pressure)
    MaybePerfAccum(acc_free);       // slot-free loop: source flushes + free notifications
    auto forward_partial_credits = [&]() {
        if constexpr (has_prev) {
            for (uint32_t r = 0; r < num_reducers; ++r) {
                // Withholding < CREDIT_BATCH grants cannot deadlock: once upstream has sent every
                // credited block, the reducer consumes them all and re-grants RECV_DEPTH >= CREDIT_BATCH
                // slots (or its last one, g == recv_total).
                const uint32_t g = word(granted_arr, r)[0];
                if (g > credits_sent[r] && (g - credits_sent[r] >= CREDIT_BATCH || g == recv_total[r])) {
                    MaybePerfBegin(acc_credit);
                    credit_hdr->to_noc_unicast_atomic_inc(
                        NocUnicastAtomicIncCommandHeader{partial_credit_noc[r], g - credits_sent[r]});
                    connection.wait_for_empty_write_slot();
                    connection.send_payload_flush_blocking_from_address(
                        reinterpret_cast<uint32_t>(credit_hdr), sizeof(PACKET_HEADER_TYPE));
                    credits_sent[r] = g;
                    MaybePerfEnd(acc_credit);
                }
            }
        }
    };

    uint32_t r = 0;
    {
        MaybeDeviceZoneScope("bwd_main");
        while (pending > 0) {
            for (uint32_t scan = 0; scan < num_reducers; ++scan, r = (r + 1 == num_reducers) ? 0 : r + 1) {
                const uint32_t k = next_k[r];
                if (k >= nb[r]) {
                    continue;
                }
                const bool is_head = roles.is_head(r, k, nb[r]);
                const bool is_tail = roles.is_tail(r, k, nb[r]);
                const uint32_t final_arrival_addr = get_arg_val<uint32_t>(final_arrival_idx + r);
                if (is_tail ? word(final_ready_arr, r)[0] <= tail_seq[r] : gsem(final_arrival_addr)[0] <= arrivals[r]) {
                    continue;  // final not here yet
                }
                const bool forward = has_prev && !is_head;
                if (forward && gsem(get_arg_val<uint32_t>(final_credit_idx + r))[0] <= k) {
                    continue;  // p-1 has not freed its slot for this block yet
                }

                // relay_final_block
                const uint32_t slot_addr = final_addr + ((k % final_depth) * num_reducers + r) * chunk_bytes;
                MaybePerfBegin(acc_relay);
                if constexpr (has_prev) {
                    if (forward) {
                        const uint64_t remote_arrival_noc = get_noc_addr(port_x, port_y, final_arrival_addr);
                        for (uint32_t pkt = 0; pkt < packets_per_chunk; ++pkt) {
                            const uint32_t off = pkt * packet_bytes;
                            if (hdr_idx == num_data_headers) {
                                noc_async_writes_flushed();  // every data header's previous send has left L1
                                hdr_idx = 0;
                            }
                            volatile tt_l1_ptr PACKET_HEADER_TYPE* data_hdr = data_hdrs[hdr_idx++];
                            const uint64_t pkt_dst = get_noc_addr(port_x, port_y, slot_addr + off);
                            if (pkt + 1 < packets_per_chunk) {
                                data_hdr->to_noc_unicast_write(NocUnicastCommandHeader{pkt_dst}, packet_bytes);
                            } else {
                                // Chunk-granular arrival: one fused (flushed) increment on the last packet.
                                data_hdr->to_noc_fused_unicast_write_atomic_inc(
                                    NocUnicastAtomicIncFusedCommandHeader{pkt_dst, remote_arrival_noc, 1, true},
                                    packet_bytes);
                            }
                            MaybePerfBegin(acc_slot_wait);
                            connection.wait_for_empty_write_slot();
                            MaybePerfEnd(acc_slot_wait);
                            connection.send_current_slot_non_blocking(
                                slot_addr + off, packet_bytes, reinterpret_cast<uint32_t>(data_hdr));
                        }
                    }
                }

                // Output DRAM: valid pages only (a tail block's reducer wrote its finals to DRAM itself;
                // with split ports port_drain writes them).
                if (!split_ports && !is_tail) {
                    const uint32_t chunk_first = (k * num_reducers + r) * chunk_tiles;
                    const uint32_t remaining = lane_tiles - chunk_first;
                    const uint32_t valid_tiles = remaining < chunk_tiles ? remaining : chunk_tiles;
                    // Bank-run transfers (high_bw_all_reduce_chunk_io.hpp): this RISC also forwards the
                    // final over Fabric, so its NoC command count per chunk is what binds the relay.
                    Io::write(output, lane_start + chunk_first, valid_tiles, slot_addr);
                }
                MaybePerfEnd(acc_relay);
                if (is_tail) {
                    ++tail_seq[r];
                } else {
                    ++arrivals[r];
                }
                next_k[r] = k + 1;
                r = (r + 1 == num_reducers) ? 0 : r + 1;  // round-robin: resume after the served reducer
                break;
            }
            // free_final_slot, per reducer in block order: relayed here and (split ports) drained to
            // DRAM by port_drain. Freeing only needs the slot's source reads (fabric payload + DRAM
            // pages) to have left L1; the DRAM write acks are collected by the barrier at kernel end.
            MaybePerfBegin(acc_free);
            for (uint32_t rr = 0; rr < num_reducers; ++rr) {
                while (freed_k[rr] < next_k[rr] && (!split_ports || word(drained_arr, rr)[0] > freed_k[rr])) {
                    noc_async_writes_flushed();
                    hdr_idx = num_data_headers;
                    const uint32_t freed = ++freed_k[rr];
                    // port_fwd forwards it as final credits to p+1.
                    if constexpr (split_ports) {
                        noc_inline_dw_write(
                            get_noc_addr(fwd_x, fwd_y, final_freed_arr + rr * CONTROL_WORD_STRIDE), freed);
                    } else {
                        word(final_freed_arr, rr)[0] = freed;
                    }
                    const uint32_t rx = get_arg_val<uint32_t>(reducer_xy_idx + 2 * rr);
                    const uint32_t ry = get_arg_val<uint32_t>(reducer_xy_idx + 2 * rr + 1);
                    noc_semaphore_inc(get_noc_addr(rx, ry, final_egress_sem_addr), 1);
                    --pending;
                }
            }
            MaybePerfEnd(acc_free);
            forward_partial_credits();
        }
    }
    MaybeDeviceZoneScope("bwd_teardown");

    for (uint32_t rr = 0; rr < num_reducers; ++rr) {
        if constexpr (has_prev) {
            while (credits_sent[rr] < recv_total[rr]) {
                forward_partial_credits();
            }
            // Every final credit must have landed before the total is subtracted (trailing head
            // blocks are never waited on above), else the counter could dip below zero.
            const uint32_t final_credit_addr = get_arg_val<uint32_t>(final_credit_idx + rr);
            while (gsem(final_credit_addr)[0] < nb[rr]) {
                forward_partial_credits();
            }
            if (nb[rr] > 0) {
                noc_semaphore_inc(get_noc_addr(port_x, port_y, final_credit_addr), 0u - nb[rr]);
            }
        }
        if (arrivals[rr] > 0) {
            noc_semaphore_inc(
                get_noc_addr(port_x, port_y, get_arg_val<uint32_t>(final_arrival_idx + rr)), 0u - arrivals[rr]);
        }
    }
    noc_async_atomic_barrier();
    noc_async_write_barrier();
    if constexpr (has_prev) {
        connection.close();
    }
    MaybePerfReport("bwd_relay", acc_relay);
    MaybePerfReport("bwd_slot_wait", acc_slot_wait);
    MaybePerfReport("bwd_free", acc_free);
    MaybePerfReport("bwd_credit", acc_credit);
}
