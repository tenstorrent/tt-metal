// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Persistent fixed-shape D2D receiver for D2DStreamService.
//
// The D2D analog of persistent_h2d_receiver.cpp: instead of draining a
// PCIe-pinned host FIFO, this kernel drains the receiver-side MeshSocket data
// FIFO (filled over tt-fabric by the persistent sender kernel on the upstream
// mesh) into the receiver backing tensor (DRAM).
//
// Each outer-loop iteration drains exactly ONE full tensor's worth of data:
// num_socket_pages socket pages, each fanned out to pages_per_chunk tensor
// pages. The socket data lives in this core's L1 FIFO, so the page is written
// straight from receiver_socket.read_ptr to DRAM (no scratch read needed) —
// mirrors recv_async's receiver_inplace_writer.cpp.
//
// The outer loop exits cleanly when the host sets `termination_semaphore` to 1.
// The check happens inside the socket-wait poll (and the consumed_counter poll)
// so shutdown stays responsive even when no sender / no workers are online.
//
// Single RISC (RISCV_0). The receiver opens a fabric connection at entry purely
// to return socket credits to the sender (fabric_socket_notify_sender); it
// never moves bulk data over fabric.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "api/tensor/tensor_accessor.h"

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
constexpr auto output_tensor_accessor_args = TensorAccessorArgs<21>();

FORCE_INLINE bool socket_wait_for_pages_with_termination(
    const SocketReceiverInterface& socket, uint32_t num_pages, volatile tt_l1_ptr uint32_t* termination_semaphore) {
    while (!socket_wait_for_pages(socket, num_pages, 1000)) {
        invalidate_l1_cache();
        if (termination_semaphore[0] == 1) {
            return false;
        }
    }
    return true;
}

void kernel_main() {
    size_t rt_args_idx = 0;
    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

    // The socket packet header lives in the fabric packet-header CB; it's used
    // only by fabric_socket_notify_sender for the credit return.
    volatile tt_l1_ptr PACKET_HEADER_TYPE* socket_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(fabric_packet_header_cb_index));

    auto output_tensor_accessor = TensorAccessor(output_tensor_accessor_args, output_tensor_addr);

    SocketReceiverInterface receiver_socket = create_receiver_socket_interface(socket_config_addr);
    set_receiver_socket_page_size(receiver_socket, socket_page_size);

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

    // Metadata multicast destination — same worker bbox as data_ready_sem, a
    // different in-core L1 address. Hoisted out of the per-transfer loop.
    uint64_t metadata_mcast_addr = 0;
    if constexpr (metadata_enabled) {
        metadata_mcast_addr = get_noc_multicast_addr(
            worker_mcast_noc_x_start,
            worker_mcast_noc_y_start,
            worker_mcast_noc_x_end,
            worker_mcast_noc_y_end,
            metadata_l1_addr);
    }

    bool terminated = false;
    while (!terminated) {
        // LEASE mode: wait for a grant before draining a transfer (the drain returns
        // socket credits over fabric, so it needs the link). Termination-aware. While
        // not granted the kernel holds no connection — the links are the model
        // graph's. Granted at the iteration boundary only (no transfer in flight), so
        // the lease is honoured cleanly.
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

        // Drain exactly one full tensor's worth of data: num_socket_pages chunks.
        for (uint32_t chunk = 0; chunk < num_socket_pages; ++chunk) {
            // Termination-aware socket wait: poll for one page with a bounded
            // early-exit count, re-checking the termination word between polls
            // so shutdown stays responsive when no sender is online.
            while (!socket_wait_for_pages_with_termination(receiver_socket, 1, termination_semaphore)) {
                terminated = true;
                break;
            }

            // The socket page sits in this core's L1 FIFO; write its
            // pages_per_chunk tensor pages straight out to DRAM.
            uint32_t l1_read_addr = receiver_socket.read_ptr;
            const uint32_t base_page = chunk * pages_per_chunk;
            for (uint32_t i = 0; i < pages_per_chunk; ++i) {
                const uint64_t noc_dst = output_tensor_accessor.get_noc_addr(base_page + i);
                noc_async_write<tensor_page_size>(l1_read_addr, noc_dst, tensor_page_size);
                l1_read_addr += tensor_page_size;
            }
            noc_async_writes_flushed();

            // Free the FIFO slot and return the credit to the sender over fabric.
            socket_pop_pages(receiver_socket, 1);
            fabric_socket_notify_sender(receiver_socket, fabric_connection, socket_packet_header_addr);
        }

        // Termination short-circuit: if the inner loop exited via the
        // termination poll, no transfer completed — skip the worker handshake
        // (workers aren't running during teardown).
        if (terminated) {
            break;
        }

        // Optional inline-metadata: drain ONE trailing socket page carrying the
        // metadata blob, then multicast its first metadata_size_bytes to every
        // receiver worker core's L1. The barrier must complete BEFORE the
        // data_ready_sem mcast below so workers observing data_ready see
        // consistent (DRAM + metadata-L1) state. The wait is termination-aware so
        // teardown stays responsive if no trailing page is coming.
        if constexpr (metadata_enabled) {
            bool got_md = true;
            while (!socket_wait_for_pages(receiver_socket, 1, /*early_exit_iter_count=*/1000)) {
                invalidate_l1_cache();
                if (termination_semaphore[0] == 1) {
                    got_md = false;
                    break;
                }
            }
            terminated = !got_md;
            if (got_md) {
                // The metadata page sits in this core's L1 FIFO at read_ptr (an
                // allocated buffer address — a valid NoC multicast source, unlike a
                // stack-local). Multicast it straight to the worker grid.
                noc_async_write_multicast(
                    receiver_socket.read_ptr,
                    metadata_mcast_addr,
                    metadata_size_bytes,
                    /*num_dests=*/num_workers);
                noc_async_write_barrier();
                socket_pop_pages(receiver_socket, 1);
                fabric_socket_notify_sender(receiver_socket, fabric_connection, socket_packet_header_addr);
            }
        }

        if (terminated) {
            break;
        }

        // Receiver-side worker handshake. Compile-time-gated.
        if constexpr (worker_sync_enabled) {
            // Ensure the backing-tensor writes are globally visible before we
            // release the workers — they read the slice as soon as they observe
            // data_ready_sem.
            noc_async_write_barrier();

            // 1. Multicast atomic-inc of data_ready to every receiver worker.
            noc_semaphore_inc_multicast(worker_mcast_addr, /*incr=*/1, /*num_dests=*/num_workers);
            // 2. Wait for exactly num_workers acks on the consumed counter, or for
            //    host-signalled termination (teardown can land here when no real
            //    workers are running, e.g. a host-driven transfer with no
            //    consumer op).
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

        // LEASE mode: this drain (+ worker handshake) is done — drop the credit-
        // return connection and hand the links back to the model graph
        // (link_grant = 0). The host polls for 0 in wait_for_fabric_links() and only
        // writes 1 again after seeing 0, so the two never write concurrently.
        if constexpr (share_fabric_links) {
            fabric_connection.close();
            fabric_open = false;
            link_grant[0] = 0;
        }
    }

    noc_async_write_barrier();
    noc_async_atomic_barrier();
    update_socket_config(receiver_socket);
    // In LEASE mode the connection is already closed between transfers (and on the
    // termination path, which only fires from an idle wait); in OWN mode it is held
    // open for the kernel's life and closed here.
    if (fabric_open) {
        fabric_connection.close();
    }
}
