// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Persistent fixed-shape D2D sender for D2DStreamService.
//
// This is the only D2D kernel with no H2D analog: an upstream device worker grid
// produces into the sender backing tensor (DRAM) and this persistent service kernel
// drains it over tt-fabric into the receiver's DRAM backing tensor.
//
// Step 1 (direct-DRAM): the data lands STRAIGHT in the receiver's DRAM backing
// tensor — no receiver-side L1 FIFO copy. The MeshSocket is kept only for the
// cross-process rendezvous/routing and for its two config words, repurposed as
// monotonic per-transfer counters:
//   * bytes_sent  (receiver config): each sender LANE Flush-atomic-incs it once per
//     transfer; the total advance per transfer is num_lanes. The Flush makes the
//     receiver EDM router drain that link's payload writes to DRAM before applying
//     the inc, so when bytes_sent reaches +num_lanes every page (on every link) is
//     present.
//   * bytes_acked (sender config): the receiver atomic-incs it once per transfer
//     after its workers consume = "you may overwrite the receiver tensor".
//
// Step 2a (multi-lane): on Blackhole there are up to 2 fabric links between adjacent
// chips, so this source builds into TWO kernels on ONE service core — lane 0 (master,
// BRISC/NOC_0) and lane 1 (sub, NCRISC/NOC_1) — selected by the is_master CT arg
// (full CT layout passed to both; `if constexpr` chooses the body). Each lane owns a
// distinct link + its own scratch/header CBs + its own NoC (so the trid rings are
// independent), and streams its CONTIGUOUS half of the tensor pages [page_start,
// page_end). The lanes are independent data movers — there is no shared socket-FIFO
// ring — and synchronize only through two monotonic counters in the shared service-
// core L1: the master bumps `go_count` once it has passed the per-transfer gates; the
// sub bumps `done_count` once its half is shipped. num_lanes == 1 degrades to a single
// lane that streams the whole tensor (no sub kernel built).
//
// Per transfer:
//   master: gate (lease grant + workers produced + receiver consumed prev) -> open
//           link0 -> go_count++ -> stream half0 -> [metadata] -> Flush-inc bytes_sent
//           -> wait done_count -> mcast consumed_sem -> close link0 / release lease.
//   sub:    wait go_count (term-aware) -> open link1 -> stream half1 -> Flush-inc
//           bytes_sent -> close link1 -> done_count++.
//
// Each lane streams its half through a transaction-id (trid) ring over its scratch CB
// so DRAM reads overlap the fabric writes.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "api/tensor/tensor_accessor.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

// CT-arg layout (must stay in sync with build_sender_program in
// ttnn/core/tensor/d2d_stream_service.cpp). The SAME full layout is passed to both
// the master and the sub kernel; per-lane fields (scratch/header CB, page range,
// is_master) differ, master-only fields (worker-sync, metadata, lease) are provided
// to both and simply unread by the sub.
constexpr uint32_t socket_config_addr = get_compile_time_arg_val(0);
constexpr uint32_t termination_semaphore_addr = get_compile_time_arg_val(1);
constexpr uint32_t socket_page_size = get_compile_time_arg_val(2);
constexpr uint32_t num_socket_pages = get_compile_time_arg_val(3);
constexpr uint32_t pages_per_chunk = get_compile_time_arg_val(4);
constexpr uint32_t tensor_page_size = get_compile_time_arg_val(5);
constexpr uint32_t input_tensor_addr = get_compile_time_arg_val(6);
constexpr uint32_t scratch_cb_index = get_compile_time_arg_val(7);               // this lane's scratch CB
constexpr uint32_t fabric_packet_header_cb_index = get_compile_time_arg_val(8);  // this lane's header CB
constexpr uint32_t fabric_max_payload_size = get_compile_time_arg_val(9);
// Worker-sync block (indices 10..17). Master only.
constexpr uint32_t worker_sync_enabled = get_compile_time_arg_val(10);
constexpr uint32_t data_ready_counter_addr = get_compile_time_arg_val(11);
constexpr uint32_t consumed_sem_addr = get_compile_time_arg_val(12);
constexpr uint32_t worker_mcast_noc_x_start = get_compile_time_arg_val(13);
constexpr uint32_t worker_mcast_noc_y_start = get_compile_time_arg_val(14);
constexpr uint32_t worker_mcast_noc_x_end = get_compile_time_arg_val(15);
constexpr uint32_t worker_mcast_noc_y_end = get_compile_time_arg_val(16);
constexpr uint32_t num_workers = get_compile_time_arg_val(17);
// Metadata block (indices 18..20). Master only. The designated worker wrote the blob
// into the master service core's L1 at sender_metadata_l1_addr before acking.
constexpr uint32_t metadata_enabled = get_compile_time_arg_val(18);
constexpr uint32_t metadata_size_bytes = get_compile_time_arg_val(19);
constexpr uint32_t sender_metadata_l1_addr = get_compile_time_arg_val(20);
// Fabric-link lease (indices 21..22). Master only. share_fabric_links: 0 = OWN mode
// (open at entry, never release); 1 = LEASE mode (hold nothing until granted a turn).
constexpr uint32_t share_fabric_links = get_compile_time_arg_val(21);
constexpr uint32_t link_grant_addr = get_compile_time_arg_val(22);
// Lane block (indices 23..28).
constexpr uint32_t is_master = get_compile_time_arg_val(23);        // 1 = lane 0 / master, 0 = sub
constexpr uint32_t num_lanes = get_compile_time_arg_val(24);        // Flush-incs to bytes_sent per transfer
constexpr uint32_t page_start = get_compile_time_arg_val(25);       // this lane's first tensor page
constexpr uint32_t page_end = get_compile_time_arg_val(26);         // this lane's last tensor page + 1
constexpr uint32_t go_count_addr = get_compile_time_arg_val(27);    // shared L1: master -> sub
constexpr uint32_t done_count_addr = get_compile_time_arg_val(28);  // shared L1: sub -> master
// [29] num_read_slots: trid-ring depth (power of 2), = the scratch CB's slot capacity.
// Decoupled from pages_per_chunk so the read pipeline runs at full depth regardless of
// the (vestigial) socket chunk plan; the host sizes the scratch CB to match.
constexpr uint32_t num_read_slots = get_compile_time_arg_val(29);
constexpr auto input_tensor_accessor_args = TensorAccessorArgs<30>();

// Emit `size` bytes from a contiguous L1 source to a single remote NoC address over
// fabric, split into <= fabric_max_payload_size packets. Used for both a tensor page
// (its DRAM home in the receiver backing tensor) and the optional metadata blob.
FORCE_INLINE void fabric_write_bytes(
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    uint64_t dst_addr,
    uint32_t l1_src_addr,
    uint32_t size,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* data_packet_header_addr) {
    uint32_t remaining = size;
    while (remaining > 0) {
        const uint32_t packet_size = remaining > fabric_max_payload_size ? fabric_max_payload_size : remaining;
        data_packet_header_addr->to_noc_unicast_write(NocUnicastCommandHeader{dst_addr}, packet_size);
        fabric_connection.wait_for_empty_write_slot();
        fabric_connection.send_payload_without_header_non_blocking_from_address(l1_src_addr, packet_size);
        fabric_connection.send_payload_flush_blocking_from_address(
            (uint32_t)data_packet_header_addr, sizeof(PACKET_HEADER_TYPE));
        dst_addr += packet_size;
        l1_src_addr += packet_size;
        remaining -= packet_size;
    }
}

// Largest power of 2 <= n (n >= 1). Rounds the read-ring depth down to a power of 2 so
// the ring slot index is a bitmask rather than a modulo.
// Following uses Hacker's Delight formula
constexpr uint32_t floor_pow2(uint32_t n) {
    if (n == 0) {
        return 0;
    }
    return 1u << (31 - __builtin_clz(n));
}

// Thin compile-time transaction-id ring. Depth is a power of 2 (asserted), so a page
// index maps to its ring slot with a bitmask and to its trid with slot + Base. Page k
// and page k + Depth share a slot/trid — exactly the "ship slot for page k, then
// prefetch the page Depth ahead into it" reuse the streaming loop relies on.
template <uint32_t Depth, uint32_t Base>
struct TridRing {
    static_assert(Depth > 0 && (Depth & (Depth - 1)) == 0, "TridRing Depth must be a power of 2");
    static constexpr uint32_t depth = Depth;
    static constexpr uint32_t slot(uint32_t page) { return page & (Depth - 1); }
    static constexpr uint32_t trid(uint32_t page) { return Base + (page & (Depth - 1)); }
};

// Read pipelining: stream pages through a trid ring over the scratch CB so DRAM reads
// overlap the fabric writes. Ring depth = the scratch CB's slot capacity
// (num_read_slots, host-sized), rounded DOWN to a power of 2 (TridRing's bitmask) —
// independent of pages_per_chunk, so a large tensor page (small pages_per_chunk) still
// pipelines at full depth.
constexpr uint32_t kRingDepth = floor_pow2(num_read_slots);
// Each lane (RISC) gets a DISJOINT trid range so the two RISCs on this core never
// share a NoC transaction id — the trid-barrier counters are not assumed independent
// across the two RISCs. Lane 0 -> [1, 1+depth), lane 1 -> [1+depth, 1+2*depth).
constexpr uint32_t kLaneId = is_master ? 0u : 1u;
constexpr uint32_t kTridBase = 1u + kLaneId * kRingDepth;
static_assert(2u * kRingDepth <= 15u, "two lanes of trids must fit in the 0..15 transaction-id space");
using ReadRing = TridRing<kRingDepth, kTridBase>;

// Stream this lane's contiguous page range [page_start_, page_end_) DIRECTLY into the
// receiver DRAM tensor over `conn`, pipelining the DRAM reads (in_acc) with the fabric
// writes (out_acc) through the trid ring: only the trid we are about to ship is
// barriered, so up to ReadRing::depth reads stay in flight behind the current write;
// each fabric write flushes locally, freeing its scratch slot, so we immediately
// prefetch the page one ring ahead into it. in_acc/out_acc share the per-shard spec
// (same accessor args, different base address).
template <typename Acc>
FORCE_INLINE void stream_half(
    const Noc& noc,
    tt::tt_fabric::WorkerToFabricEdmSender& conn,
    const Acc& in_acc,
    const Acc& out_acc,
    uint32_t cb_l1_addr,
    uint32_t page_start_,
    uint32_t page_end_,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* data_packet_header_addr) {
    const uint32_t n = page_end_ - page_start_;
    const uint32_t prime = (ReadRing::depth < n) ? ReadRing::depth : n;
    CoreLocalMem<uint32_t> scratch(cb_l1_addr);
    for (uint32_t i = 0; i < prime; ++i) {
        noc.async_read<NocOptions::TXN_ID>(
            in_acc,
            scratch,
            tensor_page_size,
            {.page_id = page_start_ + i},
            {.offset_bytes = i * tensor_page_size},
            NocOptVals{.trid = ReadRing::trid(i)});
    }
    for (uint32_t k = 0; k < n; ++k) {
        const uint32_t slot = ReadRing::slot(k);
        const uint32_t trid = ReadRing::trid(k);
        const uint32_t src = cb_l1_addr + slot * tensor_page_size;
        noc.async_read_barrier<NocOptions::TXN_ID>(NocOptVals{.trid = trid});
        fabric_write_bytes(conn, out_acc.get_noc_addr(page_start_ + k), src, tensor_page_size, data_packet_header_addr);
        const uint32_t next = k + ReadRing::depth;
        if (next < n) {
            noc.async_read<NocOptions::TXN_ID>(
                in_acc,
                scratch,
                tensor_page_size,
                {.page_id = page_start_ + next},
                {.offset_bytes = slot * tensor_page_size},
                NocOptVals{.trid = trid});
        }
    }
}

// This lane's data-landed signal: Flush-atomic-inc the receiver's bytes_sent by 1.
// The Flush drains this link's payload writes (DRAM pages + any metadata) before the
// inc applies, so the inc implies this lane's writes have landed. Commutative across
// lanes — bytes_sent reaches +num_lanes once every lane has signalled.
FORCE_INLINE void flush_inc_data_landed(
    tt::tt_fabric::WorkerToFabricEdmSender& conn,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* socket_packet_header_addr,
    uint64_t recv_bytes_sent_noc_addr) {
    socket_packet_header_addr->to_noc_unicast_atomic_inc(
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{recv_bytes_sent_noc_addr, 1, /*flush=*/true});
    conn.wait_for_empty_write_slot();
    conn.send_payload_flush_blocking_from_address((uint32_t)socket_packet_header_addr, sizeof(PACKET_HEADER_TYPE));
}

void kernel_main() {
    size_t rt_args_idx = 0;
    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

    // Per-coord base address of the receiver's DRAM backing tensor (the direct
    // fabric-write destination). Appended after the fabric-connection args; per device.
    const uint32_t receiver_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);

    // This lane's two fabric headers (its own header CB): one for the bulk page
    // writes, one for the data-landed atomic-inc. Routed once below (L1 writes only).
    CircularBuffer fabric_hdr_cb(fabric_packet_header_cb_index);
    const uint32_t fabric_hdr_l1 = fabric_hdr_cb.get_write_ptr();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* data_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(fabric_hdr_l1);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* socket_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(fabric_hdr_l1 + sizeof(PACKET_HEADER_TYPE));

    SocketSenderInterface sender_socket = create_sender_socket_interface(socket_config_addr);
    set_sender_socket_page_size(sender_socket, socket_page_size);

    sender_downstream_encoding downstream_enc = get_downstream_encoding(sender_socket, 0);
    fabric_set_unicast_route(data_packet_header_addr, downstream_enc);
    fabric_set_unicast_route(socket_packet_header_addr, downstream_enc);
    const uint32_t receiver_noc_x = downstream_enc.d2d.downstream_noc_x;
    const uint32_t receiver_noc_y = downstream_enc.d2d.downstream_noc_y;

    // Sender backing (read source) and receiver backing (fabric-write destination)
    // share the per-shard spec: one set of accessor args, two base addresses.
    auto input_tensor_accessor = TensorAccessor(input_tensor_accessor_args, input_tensor_addr);
    auto output_tensor_accessor = TensorAccessor(input_tensor_accessor_args, receiver_tensor_addr);
    Noc noc(noc_index);
    CircularBuffer scratch_cb(scratch_cb_index);
    const uint32_t cb_l1_addr = scratch_cb.get_write_ptr();

    volatile tt_l1_ptr uint32_t* termination_semaphore =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_semaphore_addr);
    const uint64_t recv_bytes_sent_noc_addr =
        get_noc_addr(receiver_noc_x, receiver_noc_y, sender_socket.downstream_bytes_sent_addr);

    // Lane-to-lane sync (shared service-core L1; single writer per word, so plain
    // volatile + invalidate_l1_cache is race-free): master bumps go_count, sub bumps
    // done_count. Both are monotonic — comparisons use wrap-safe unsigned diffs.
    volatile tt_l1_ptr uint32_t* go_count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_count_addr);
    volatile tt_l1_ptr uint32_t* done_count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(done_count_addr);

    // Clear any stale NoC transaction-id barrier state before the streaming loop.
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);

    if constexpr (is_master) {
        // ---- Lane 0 / master: owns the gates, the worker handshake and the lease ----
        volatile tt_l1_ptr uint32_t* data_ready_counter =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_ready_counter_addr);
        volatile tt_l1_ptr uint32_t* link_grant = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(link_grant_addr);
        // bytes_acked (this sender's config word): the receiver atomic-incs it per
        // consumed transfer. The overwrite gate spins on it (wrap-safe diff vs my_sent).
        volatile tt_l1_ptr uint32_t* bytes_acked =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_socket.bytes_acked_base_addr);

        uint64_t consumed_mcast_addr = 0;
        if constexpr (worker_sync_enabled) {
            consumed_mcast_addr = get_noc_multicast_addr(
                worker_mcast_noc_x_start,
                worker_mcast_noc_y_start,
                worker_mcast_noc_x_end,
                worker_mcast_noc_y_end,
                consumed_sem_addr);
        }

        bool fabric_open = false;
        if constexpr (share_fabric_links == 0) {
            fabric_connection.open();
            fabric_open = true;
        }

        uint32_t last_data_ready = 0;
        // Monotonic count of transfers issued (wraps over the persistent lifetime, so
        // all comparisons use wrap-safe unsigned diffs).
        uint32_t my_sent = 0;
        uint32_t go_local = 0;
        uint32_t last_done = 0;
        // Max transfers outstanding (sent but not yet consumed). Single buffer in
        // Step 1; Step 3 (double-buffer) raises this.
        constexpr uint32_t kBufferDepth = 1;
        bool terminated = false;
        while (!terminated) {
            // 1. Idle wait. Proceed only when ALL hold (iteration boundary, no transfer
            //    in flight): (a) LEASE grant, (b) workers produced (num_workers more
            //    data_ready), (c) OVERWRITE GATE — fewer than kBufferDepth transfers
            //    outstanding (sent - acked, wrap-safe). Or break on termination.
            bool ready = false;
            while (!ready) {
                invalidate_l1_cache();
                if (termination_semaphore[0] == 1) {
                    terminated = true;
                    break;
                }
                if constexpr (share_fabric_links) {
                    if (link_grant[0] != 1) {
                        continue;
                    }
                }
                if (static_cast<uint32_t>(my_sent - *bytes_acked) >= kBufferDepth) {
                    continue;
                }
                const uint32_t cur = *data_ready_counter;
                if ((cur - last_data_ready) == num_workers) {
                    last_data_ready = cur;
                    ready = true;
                }
            }
            if (terminated) {
                break;
            }

            // 2. LEASE: acquire the link for this transfer (route/headers persist in L1).
            if constexpr (share_fabric_links) {
                fabric_connection.open();
                fabric_open = true;
            }

            // 3. Release the sub to stream its half concurrently (gates have passed).
            if constexpr (num_lanes > 1) {
                go_local += 1;
                go_count[0] = go_local;
            }

            // 4. Stream the master's half directly into the receiver DRAM tensor.
            stream_half(
                noc,
                fabric_connection,
                input_tensor_accessor,
                output_tensor_accessor,
                cb_l1_addr,
                page_start,
                page_end,
                data_packet_header_addr);

            // 4b. Optional metadata: fabric-write the blob to the receiver's vestigial
            //     socket-FIFO L1 base (covered by the master's Flush-inc below); the
            //     receiver mcasts it to its worker grid after the data-landed signal.
            if constexpr (metadata_enabled) {
                const uint64_t md_dst =
                    get_noc_addr(receiver_noc_x, receiver_noc_y, sender_socket.downstream_fifo_addr);
                fabric_write_bytes(
                    fabric_connection, md_dst, sender_metadata_l1_addr, socket_page_size, data_packet_header_addr);
            }

            // 5. Data-landed: the master's Flush-inc on bytes_sent (+1 of num_lanes).
            flush_inc_data_landed(fabric_connection, socket_packet_header_addr, recv_bytes_sent_noc_addr);
            my_sent += 1;

            // 6. Wait for the sub's half (done_count advanced). On clean teardown the
            //    loop exits at the idle wait above, never here, so no termination check
            //    is needed (the sub always completes a transfer it was told to start).
            if constexpr (num_lanes > 1) {
                while ((done_count[0] - last_done) < 1u) {
                    invalidate_l1_cache();
                }
                last_done = done_count[0];
            }

            // 7. Release the sender worker grid so it can overwrite the SENDER backing.
            if constexpr (worker_sync_enabled) {
                noc_semaphore_inc_multicast(consumed_mcast_addr, /*incr=*/1, /*num_dests=*/num_workers);
            }

            // 8. LEASE: this transfer is done — drop the link and hand it back
            //    (link_grant = 0). Both lanes' links are closed by now (the sub closes
            //    before done_count++), so the model graph may take the links.
            if constexpr (share_fabric_links) {
                fabric_connection.close();
                fabric_open = false;
                link_grant[0] = 0;
            }
        }

        update_socket_config(sender_socket);
        if (fabric_open) {
            fabric_connection.close();
        }
    } else {
        // ---- Lane 1 / sub: a pure data mover gated by the master's go_count ----
        bool fabric_open = false;
        if constexpr (share_fabric_links == 0) {
            fabric_connection.open();
            fabric_open = true;
        }

        uint32_t last_go = 0;
        bool terminated = false;
        while (!terminated) {
            // Wait for the master's go (it has passed the per-transfer gates), or break
            // on termination (the master stops bumping go_count at teardown).
            while (true) {
                invalidate_l1_cache();
                if (termination_semaphore[0] == 1) {
                    terminated = true;
                    break;
                }
                if ((go_count[0] - last_go) >= 1u) {
                    last_go = go_count[0];
                    break;
                }
            }
            if (terminated) {
                break;
            }

            if constexpr (share_fabric_links) {
                fabric_connection.open();
                fabric_open = true;
            }

            stream_half(
                noc,
                fabric_connection,
                input_tensor_accessor,
                output_tensor_accessor,
                cb_l1_addr,
                page_start,
                page_end,
                data_packet_header_addr);

            flush_inc_data_landed(fabric_connection, socket_packet_header_addr, recv_bytes_sent_noc_addr);

            if constexpr (share_fabric_links) {
                fabric_connection.close();
                fabric_open = false;
            }

            // Signal the master that this lane's half (+ Flush-inc) is shipped.
            done_count[0] += 1;
        }

        if (fabric_open) {
            fabric_connection.close();
        }
    }
}
