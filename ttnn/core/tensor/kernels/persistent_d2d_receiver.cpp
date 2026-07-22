// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Persistent fixed-shape D2D receiver for D2DStreamService.
//
// Step 1 (direct-DRAM): the sender writes the full tensor STRAIGHT into this mesh's
// DRAM backing tensor over fabric — there is NO receiver-side L1 FIFO copy. This
// kernel therefore moves no bulk data; it only runs the per-transfer handshake via
// the two MeshSocket config words repurposed as monotonic counters:
//   * bytes_sent  (this receiver's config word): the sender Flush-atomic-incs it
//     once per transfer = "the full tensor (+ optional metadata) has landed in your
//     DRAM". The Flush guarantees every page is present when we observe the inc.
//   * bytes_acked (the sender's config word): we atomic-inc it once per transfer,
//     AFTER our workers consume = "you may overwrite the receiver tensor".
//
// Each outer-loop iteration:
//   1. waits for the data-landed signal (bytes_sent advances) — termination-aware,
//   2. (optional) multicasts the metadata blob (the sender staged it in our
//      vestigial socket-FIFO L1) to the worker grid,
//   3. multicast-incs data_ready_sem so the receiver workers consume the slice,
//   4. waits num_workers acks on consumed_counter,
//   5. atomic-incs the sender's bytes_acked (overwrite-OK) over fabric.
//
// Single RISC (RISCV_0). The receiver holds a fabric connection only to send the
// per-transfer overwrite-OK inc upstream; it never moves bulk data over fabric.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "api/tensor/tensor_accessor.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/circular_buffer.h"

// CT-arg layout (must stay in sync with build_receiver_program in
// ttnn/core/tensor/d2d_stream_service.cpp).
constexpr uint32_t socket_config_addr = get_compile_time_arg_val(0);
constexpr uint32_t termination_semaphore_addr = get_compile_time_arg_val(1);
constexpr uint32_t socket_page_size = get_compile_time_arg_val(2);
constexpr uint32_t num_socket_pages = get_compile_time_arg_val(3);
constexpr uint32_t pages_per_chunk = get_compile_time_arg_val(4);
constexpr uint32_t tensor_page_size = get_compile_time_arg_val(5);
constexpr uint32_t output_tensor_addr = get_compile_time_arg_val(6);
constexpr uint32_t fabric_packet_header_cb_index = get_compile_time_arg_val(7);
// Worker-sync block (indices 8..15). Unused when worker_sync_enabled == 0.
constexpr uint32_t worker_sync_enabled = get_compile_time_arg_val(8);
constexpr uint32_t data_ready_sem_addr = get_compile_time_arg_val(9);
constexpr uint32_t consumed_counter_addr = get_compile_time_arg_val(10);
constexpr uint32_t worker_mcast_noc_x_start = get_compile_time_arg_val(11);
constexpr uint32_t worker_mcast_noc_y_start = get_compile_time_arg_val(12);
constexpr uint32_t worker_mcast_noc_x_end = get_compile_time_arg_val(13);
constexpr uint32_t worker_mcast_noc_y_end = get_compile_time_arg_val(14);
constexpr uint32_t num_workers = get_compile_time_arg_val(15);
// Metadata multicast block (indices 16..18). Unused when metadata_enabled == 0.
// Mirrors persistent_h2d_receiver.cpp: metadata_size_bytes is the un-padded user
// size; the kernel multicasts exactly that many bytes to metadata_l1_addr (the
// uniform receiver worker-grid L1 address) using the same worker bbox as
// data_ready_sem.
constexpr uint32_t metadata_enabled = get_compile_time_arg_val(16);
constexpr uint32_t metadata_size_bytes = get_compile_time_arg_val(17);
constexpr uint32_t metadata_l1_addr = get_compile_time_arg_val(18);
// Fabric-link lease (indices 19..20), mirror of the sender. share_fabric_links: 0 =
// OWN mode (open the credit-return connection at entry, never release); 1 = LEASE
// mode (hold no connection until granted a turn). link_grant is the single ping-pong
// word (0 = idle/done, 1 = granted one drain). See
// D2DStreamServiceReceiver::wait_for_fabric_links() / release_fabric_links().
constexpr uint32_t share_fabric_links = get_compile_time_arg_val(19);
constexpr uint32_t link_grant_addr = get_compile_time_arg_val(20);
// [21] num_lanes: how many sender lanes Flush-inc bytes_sent per transfer (one inc
// per lane), i.e. the per-transfer advance this receiver awaits as the data-landed
// signal. Derived identically on both sides from the symmetric link topology.
constexpr uint32_t num_lanes = get_compile_time_arg_val(21);
// Accessor args for the receiver backing tensor occupy CT indices 22+ (host CT layout
// unchanged otherwise), but the receiver no longer reads/writes the tensor itself —
// the sender writes it directly over fabric — so no accessor is built.
constexpr auto output_tensor_accessor_args = TensorAccessorArgs<22>();

void kernel_main() {
    size_t rt_args_idx = 0;
    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

    // The socket packet header lives in the fabric packet-header CB; it's used
    // only for the per-transfer overwrite-OK atomic-inc sent upstream to the sender.
    CircularBuffer fabric_hdr_cb(fabric_packet_header_cb_index);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* socket_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(fabric_hdr_cb.get_write_ptr());

    SocketReceiverInterface receiver_socket = create_receiver_socket_interface(socket_config_addr);
    set_receiver_socket_page_size(receiver_socket, socket_page_size);

    // The overwrite-OK inc routes upstream (to the sender chip); set the route once
    // (an L1 write to the header, no fabric connection needed).
    fabric_set_unicast_route(socket_packet_header_addr, receiver_socket);
    const uint64_t sender_bytes_acked_noc_addr = get_noc_addr(
        receiver_socket.d2d.upstream_noc_x,
        receiver_socket.d2d.upstream_noc_y,
        receiver_socket.d2d.upstream_bytes_acked_addr);

    // bytes_sent (this receiver's config word): each sender lane Flush-atomic-incs it
    // once per transfer, so it advances by num_lanes per transfer — wait for exactly
    // that. (Single-lane ⇒ kIncsPerTransfer == 1, the Step 1 behavior.)
    constexpr uint32_t kIncsPerTransfer = num_lanes;
    volatile tt_l1_ptr uint32_t* bytes_sent_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_socket.bytes_sent_addr);

    volatile tt_l1_ptr uint32_t* termination_semaphore =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_semaphore_addr);
    volatile tt_l1_ptr uint32_t* link_grant = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(link_grant_addr);

    // OWN mode: hold the (credit-return) fabric connection for the kernel's whole
    // life. LEASE mode: hold nothing until granted a turn — opened per drain below.
    bool fabric_open = false;
    if constexpr (share_fabric_links == 0) {
        fabric_connection.open();
        fabric_open = true;
    }

    // Worker-sync state. Allocated unconditionally for readability; the
    // `if constexpr` gates dead-code-eliminate it when worker_sync_enabled == 0.
    uint64_t worker_mcast_addr = 0;
    volatile tt_l1_ptr uint32_t* consumed_ptr = nullptr;
    uint32_t last_consumed = 0;
    if constexpr (worker_sync_enabled) {
        worker_mcast_addr = get_noc_multicast_addr(
            worker_mcast_noc_x_start,
            worker_mcast_noc_y_start,
            worker_mcast_noc_x_end,
            worker_mcast_noc_y_end,
            data_ready_sem_addr);
        consumed_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(consumed_counter_addr);
    }

    // 2.0 NoC interface + multicast endpoint for the metadata fan-out. The endpoint is
    // stateless; the worker bbox + metadata_l1_addr are supplied per call below (same
    // worker bbox as data_ready_sem, a different in-core L1 address).
    Noc noc;
    MulticastEndpoint metadata_mcast;

    // Last observed value of the receiver's bytes_sent counter (the sender's
    // data-landed signal); advances by kIncsPerTransfer each transfer.
    uint32_t last_seen_sent = 0;
    bool terminated = false;
    while (!terminated) {
        // LEASE mode: wait for a grant before the transfer's overwrite-OK inc (which
        // is sent over fabric, so it needs the link). Termination-aware. While not
        // granted the kernel holds no connection — the links are the model graph's.
        // Granted at the iteration boundary only, so the lease is honoured cleanly.
        if constexpr (share_fabric_links) {
            bool granted = false;
            while (!granted) {
                invalidate_l1_cache();
                if (termination_semaphore[0] == 1) {
                    terminated = true;
                    break;
                }
                if (link_grant[0] == 1) {
                    granted = true;
                }
            }
            if (terminated) {
                break;
            }
            // Acquire the credit-return connection for exactly this drain.
            fabric_connection.open();
            fabric_open = true;
        }

        // 1. Wait for the data-landed signal: the sender Flush-atomic-incs bytes_sent
        //    once per transfer after writing the full tensor to our DRAM. Because the
        //    inc is Flushed, every page is already present in DRAM when we observe it —
        //    there is nothing to drain here. Termination-aware.
        bool got_data = false;
        while (!got_data) {
            invalidate_l1_cache();
            if (termination_semaphore[0] == 1) {
                terminated = true;
                break;
            }
            if ((*bytes_sent_ptr - last_seen_sent) >= kIncsPerTransfer) {
                last_seen_sent += kIncsPerTransfer;
                got_data = true;
            }
        }
        if (terminated) {
            break;
        }

        // 2. Optional inline-metadata: the sender staged the blob in our vestigial
        //    socket-FIFO L1 (covered by the same Flush'd data-landed inc, so it is
        //    present). Multicast its first metadata_size_bytes to every receiver worker
        //    core's L1. The barrier must complete BEFORE the data_ready_sem mcast below
        //    so workers observing data_ready see consistent (DRAM + metadata-L1) state.
        if constexpr (metadata_enabled) {
            noc.async_write_multicast(
                CoreLocalMem<uint32_t>(receiver_socket.read_ptr),
                metadata_mcast,
                metadata_size_bytes,
                /*num_dsts=*/num_workers,
                {},
                {.noc_x_start = worker_mcast_noc_x_start,
                 .noc_y_start = worker_mcast_noc_y_start,
                 .noc_x_end = worker_mcast_noc_x_end,
                 .noc_y_end = worker_mcast_noc_y_end,
                 .addr = metadata_l1_addr});
            noc.async_write_barrier();
        }

        // 3. Receiver-side worker handshake. Compile-time-gated.
        if constexpr (worker_sync_enabled) {
            // 1. Multicast atomic-inc of data_ready to every receiver worker. The
            //    backing-tensor writes already landed (the sender's Flush'd inc), so
            //    the workers reading the slice on data_ready see consistent DRAM.
            noc_semaphore_inc_multicast(worker_mcast_addr, /*incr=*/1, /*num_dests=*/num_workers);
            // 2. Wait for exactly num_workers acks on the consumed counter, or for
            //    host-signalled termination (teardown can land here when no real
            //    workers are running, e.g. a host-driven transfer with no consumer op).
            while (true) {
                invalidate_l1_cache();
                if (termination_semaphore[0] == 1) {
                    terminated = true;
                    break;
                }
                const uint32_t cur = *consumed_ptr;
                if ((cur - last_consumed) == num_workers) {
                    last_consumed = cur;
                    break;
                }
            }
            if (terminated) {
                break;
            }
        }

        // 4. Overwrite-OK: atomic-inc the sender's bytes_acked over fabric. Our workers
        //    have consumed this transfer (gated above), so the sender may overwrite the
        //    receiver tensor with the next one. Non-Flush — there is no fabric payload
        //    to order before it (the worker reads were local and have completed).
        socket_packet_header_addr->to_noc_unicast_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sender_bytes_acked_noc_addr, 1, /*flush=*/false});
        fabric_connection.wait_for_empty_write_slot();
        fabric_connection.send_payload_flush_blocking_from_address(
            (uint32_t)socket_packet_header_addr, sizeof(PACKET_HEADER_TYPE));

        // 5. LEASE mode: this transfer is done — drop the credit-return connection and
        //    hand the links back to the model graph (link_grant = 0). The host polls
        //    for 0 in wait_for_fabric_links() and only writes 1 again after seeing 0,
        //    so the two never write concurrently.
        if constexpr (share_fabric_links) {
            fabric_connection.close();
            fabric_open = false;
            link_grant[0] = 0;
        }
    }

    noc.async_write_barrier();
    noc.async_atomic_barrier();
    update_socket_config(receiver_socket);
    // In LEASE mode the connection is already closed between transfers (and on the
    // termination path, which only fires from an idle wait); in OWN mode it is held
    // open for the kernel's life and closed here.
    if (fabric_open) {
        fabric_connection.close();
    }
}
