// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Persistent fixed-shape D2D sender for D2DStreamService.
//
// This is the only D2D kernel with no H2D analog: the H2D model has the host
// write directly into the socket FIFO, whereas here an upstream device worker
// grid produces into the sender backing tensor (DRAM) and this persistent
// service kernel drains it over tt-fabric into the downstream receiver's socket
// FIFO.
//
// Each outer-loop iteration:
//   1. waits for the sender worker grid (data_ready_counter reaches
//      num_workers more increments) — termination-aware,
//   2. drains one full tensor's worth of data: num_socket_pages socket pages,
//      each = pages_per_chunk tensor pages staged DRAM -> scratch CB then
//      fabric-written CB -> receiver FIFO (socket_reserve / push /
//      fabric_socket_notify_receiver per page),
//   3. multicast-incs consumed_sem on the sender worker grid so the workers can
//      overwrite the backing tensor with the next iteration.
//
// Single RISC (RISCV_0). The fabric connection is opened once at entry and held
// for the kernel's lifetime; never reopened mid-loop.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "api/tensor/tensor_accessor.h"

// CT-arg layout (must stay in sync with build_sender_program in
// ttnn/core/tensor/d2d_stream_service.cpp).
constexpr uint32_t socket_config_addr = get_compile_time_arg_val(0);
constexpr uint32_t termination_semaphore_addr = get_compile_time_arg_val(1);
constexpr uint32_t socket_page_size = get_compile_time_arg_val(2);
constexpr uint32_t num_socket_pages = get_compile_time_arg_val(3);
constexpr uint32_t pages_per_chunk = get_compile_time_arg_val(4);
constexpr uint32_t tensor_page_size = get_compile_time_arg_val(5);
constexpr uint32_t input_tensor_addr = get_compile_time_arg_val(6);
constexpr uint32_t scratch_cb_index = get_compile_time_arg_val(7);
constexpr uint32_t fabric_packet_header_cb_index = get_compile_time_arg_val(8);
constexpr uint32_t fabric_max_payload_size = get_compile_time_arg_val(9);
// Worker-sync block (indices 10..16). Unused when worker_sync_enabled == 0.
constexpr uint32_t worker_sync_enabled = get_compile_time_arg_val(10);
constexpr uint32_t data_ready_counter_addr = get_compile_time_arg_val(11);
constexpr uint32_t consumed_sem_addr = get_compile_time_arg_val(12);
constexpr uint32_t worker_mcast_noc_x_start = get_compile_time_arg_val(13);
constexpr uint32_t worker_mcast_noc_y_start = get_compile_time_arg_val(14);
constexpr uint32_t worker_mcast_noc_x_end = get_compile_time_arg_val(15);
constexpr uint32_t worker_mcast_noc_y_end = get_compile_time_arg_val(16);
constexpr uint32_t num_workers = get_compile_time_arg_val(17);
// Metadata block (indices 18..20). Unused when metadata_enabled == 0. The
// designated worker wrote the blob into this service core's L1 at
// sender_metadata_l1_addr before acking; this kernel ships it as one trailing
// socket page after the data drain.
constexpr uint32_t metadata_enabled = get_compile_time_arg_val(18);
constexpr uint32_t metadata_size_bytes = get_compile_time_arg_val(19);
constexpr uint32_t sender_metadata_l1_addr = get_compile_time_arg_val(20);
// Fabric-link lease (indices 21..22). share_fabric_links: 0 = OWN mode (open the
// fabric connection once at entry, never release it — original V0 behavior); 1 =
// LEASE mode (hold no connection until granted a turn). link_grant is the single
// host<->kernel ping-pong word: 0 = idle/done (no connection, links free for the
// model graph), 1 = granted (the kernel's turn for exactly one transfer). The host
// writes 1 (release_fabric_links); the kernel writes 0 after its transfer; the host
// polls for 0 (wait_for_fabric_links). Writers never overlap → race-free.
constexpr uint32_t share_fabric_links = get_compile_time_arg_val(21);
constexpr uint32_t link_grant_addr = get_compile_time_arg_val(22);
constexpr auto input_tensor_accessor_args = TensorAccessorArgs<23>();

// Emit one socket page (socket_page_size bytes) from a contiguous L1 source to
// the downstream FIFO over fabric, split into <= fabric_max_payload_size
// packets. Adapted from send_async's sender_writer.cpp::write_data_to_remote_core
// (CB push/pop dropped — the persistent kernel owns the single-slot scratch CB).
FORCE_INLINE void fabric_write_socket_page(
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    uint64_t dst_addr,
    uint32_t l1_src_addr,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* data_packet_header_addr) {
    uint32_t remaining = socket_page_size;
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

void kernel_main() {
    size_t rt_args_idx = 0;
    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

    // Two fabric headers in the packet-header CB: one for the data writes, one for
    // the socket control-flow notify. Set up before any open — all L1 writes, no
    // connection needed.

    volatile tt_l1_ptr PACKET_HEADER_TYPE* data_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(fabric_packet_header_cb_index));
    volatile tt_l1_ptr PACKET_HEADER_TYPE* socket_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
            get_write_ptr(fabric_packet_header_cb_index) + sizeof(PACKET_HEADER_TYPE));

    SocketSenderInterface sender_socket = create_sender_socket_interface(socket_config_addr);
    set_sender_socket_page_size(sender_socket, socket_page_size);

    // Only one downstream per socket in V0. The socket config carries the
    // receiver's NoC coords, so the FIFO write target is built directly from the
    // downstream encoding (no allocator bank-id lookup — service cores on the FD
    // column have no L1 bank registered). Mirrors the canonical fabric socket
    // sender (tests/.../misc/socket/fabric_sender.cpp). The unicast route is written
    // into the packet-header CB once here and reused across every (re-)open of the
    // fabric connection — no fabric connection is required to set it.
    sender_downstream_encoding downstream_enc = get_downstream_encoding(sender_socket, 0);
    fabric_set_unicast_route(data_packet_header_addr, downstream_enc);
    const uint32_t receiver_noc_x = downstream_enc.d2d.downstream_noc_x;
    const uint32_t receiver_noc_y = downstream_enc.d2d.downstream_noc_y;

    auto input_tensor_accessor = TensorAccessor(input_tensor_accessor_args, input_tensor_addr);
    const uint32_t cb_l1_addr = get_write_ptr(scratch_cb_index);

    volatile tt_l1_ptr uint32_t* termination_semaphore =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_semaphore_addr);
    volatile tt_l1_ptr uint32_t* data_ready_counter =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_ready_counter_addr);
    volatile tt_l1_ptr uint32_t* link_grant = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(link_grant_addr);

    // OWN mode: hold the fabric connection for the kernel's whole life (open now).
    // LEASE mode: hold nothing until granted a turn — opened per transfer below.
    bool fabric_open = false;
    if constexpr (share_fabric_links == 0) {
        fabric_connection.open();
        fabric_open = true;
    }

    uint64_t consumed_mcast_addr = 0;
    if constexpr (worker_sync_enabled) {
        consumed_mcast_addr = get_noc_multicast_addr(
            worker_mcast_noc_x_start,
            worker_mcast_noc_y_start,
            worker_mcast_noc_x_end,
            worker_mcast_noc_y_end,
            consumed_sem_addr);
    }

    uint32_t last_data_ready = 0;
    bool terminated = false;
    while (!terminated) {
        // 1. Idle wait. Gate on (a) the fabric-link grant (LEASE mode only) and
        //    (b) the sender worker grid (num_workers more data_ready increments),
        //    or break on host termination. Both are checked here at the iteration
        //    boundary — no transfer is in flight and the kernel holds no connection
        //    (LEASE mode) — so the lease is honoured at a clean point. While not
        //    granted the kernel holds no link; while granted-but-waiting-for-data it
        //    holds the turn but still no connection (opened only once data is ready).
        bool ready = false;
        while (!ready) {
            invalidate_l1_cache();
            if (termination_semaphore[0] == 1) {
                terminated = true;
                break;
            }
            if constexpr (share_fabric_links) {
                if (link_grant[0] != 1) {
                    // Not our turn — links belong to the model graph; keep waiting.
                    continue;
                }
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

        // 2. LEASE mode: acquire the link for exactly this transfer. The unicast
        //    route + packet headers were set up once at entry and persist in L1, so
        //    open() is all that is needed.
        if constexpr (share_fabric_links) {
            fabric_connection.open();
            fabric_open = true;
        }

        // 3. Drain one full tensor's worth of data to the receiver via fabric.
        for (uint32_t chunk = 0; chunk < num_socket_pages; ++chunk) {
            socket_reserve_pages(sender_socket, 1);

            // Stage pages_per_chunk DRAM tensor pages into the scratch CB.
            const uint32_t base_page = chunk * pages_per_chunk;
            uint32_t dst = cb_l1_addr;
            for (uint32_t i = 0; i < pages_per_chunk; ++i) {
                noc_async_read(input_tensor_accessor.get_noc_addr(base_page + i), dst, tensor_page_size);
                dst += tensor_page_size;
            }
            noc_async_read_barrier();

            const uint64_t fifo_dst_addr = get_noc_addr(
                receiver_noc_x, receiver_noc_y, sender_socket.write_ptr + sender_socket.downstream_fifo_addr);
            fabric_write_socket_page(fabric_connection, fifo_dst_addr, cb_l1_addr, data_packet_header_addr);

            socket_push_pages(sender_socket, 1);
            fabric_socket_notify_receiver(sender_socket, fabric_connection, socket_packet_header_addr);
        }

        // 2b. Optional trailing metadata page. The designated worker wrote the
        //     blob into this service core's L1 at sender_metadata_l1_addr before
        //     acking (and the kernel only got here after num_workers acks), so the
        //     blob is present.
        //     Data from sender_metadata_l1_addr is sent directly to the receiver over fabric with
        //     intermediate storage in the scratch CB.
        if constexpr (metadata_enabled) {
            socket_reserve_pages(sender_socket, 1);
            const uint64_t md_fifo_dst_addr = get_noc_addr(
                receiver_noc_x, receiver_noc_y, sender_socket.write_ptr + sender_socket.downstream_fifo_addr);
            fabric_write_socket_page(
                fabric_connection, md_fifo_dst_addr, sender_metadata_l1_addr, data_packet_header_addr);
            socket_push_pages(sender_socket, 1);
            fabric_socket_notify_receiver(sender_socket, fabric_connection, socket_packet_header_addr);
        }

        // 4. Release the sender worker grid (consumed_sem) so it can overwrite
        //    the backing tensor with the next iteration's slice.
        if constexpr (worker_sync_enabled) {
            noc_semaphore_inc_multicast(consumed_mcast_addr, /*incr=*/1, /*num_dests=*/num_workers);
        }

        // 5. LEASE mode: this transfer is done — drop the fabric connection and hand
        //    the links back to the model graph (link_grant = 0). The host polls for
        //    0 in wait_for_fabric_links(); it only writes 1 again after seeing 0, so
        //    the two never write concurrently.
        if constexpr (share_fabric_links) {
            fabric_connection.close();
            fabric_open = false;
            link_grant[0] = 0;
        }
    }

    update_socket_config(sender_socket);

    // In LEASE mode the connection is already closed between transfers (and on the
    // termination path, which only fires from the idle wait); in OWN mode it is held
    // open for the kernel's life and closed here.
    if (fabric_open) {
        fabric_connection.close();
    }
}
