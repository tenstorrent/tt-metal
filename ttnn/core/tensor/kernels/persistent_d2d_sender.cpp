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
// Step 1 (direct-DRAM): the data lands STRAIGHT in the receiver's DRAM backing
// tensor — there is no receiver-side L1 socket FIFO copy. The MeshSocket is kept
// only for the cross-process rendezvous/routing and for its two config words,
// which are repurposed as monotonic per-transfer counters:
//   * bytes_sent  (receiver config): the sender Flush-atomic-incs it once per
//     transfer = "the full tensor (+ optional metadata) has landed in your DRAM".
//     The Flush flag makes the receiver EDM router drain this link's payload
//     writes to DRAM before applying the inc, so the inc implies every page is
//     present.
//   * bytes_acked (sender config): the receiver atomic-incs it once per transfer
//     after its workers have consumed = "you may overwrite the receiver tensor".
//
// Each outer-loop iteration:
//   1. idle-waits until granted the link (lease), the sender worker grid has
//      produced (data_ready_counter += num_workers), AND the receiver has consumed
//      the previous transfer (outstanding = sent - acked < buffer depth, computed as
//      a wrap-safe unsigned diff since both counters wrap) — termination-aware. This
//      bytes_acked gate is the single-buffer overwrite guard (Step 3 double-buffers
//      the receiver tensor to remove this serialization).
//   2. writes one full tensor's worth of data directly to the receiver DRAM tensor:
//      num_socket_pages chunks, each = pages_per_chunk tensor pages staged
//      DRAM -> scratch CB then fabric-written per page to its DRAM home
//      (output_tensor_accessor.get_noc_addr(page)). No socket_reserve/push.
//   3. (optional) fabric-writes the metadata blob to the receiver's vestigial
//      socket-FIFO L1 (the receiver mcasts it to its worker grid).
//   4. Flush-atomic-incs the receiver bytes_sent counter (data-landed signal).
//   5. multicast-incs consumed_sem on the sender worker grid so the workers can
//      overwrite the SENDER backing tensor with the next iteration.
//
// Single RISC (RISCV_0). Fabric connection: OWN mode opens once at entry; LEASE
// mode opens per granted transfer and closes after.

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

// Emit `size` bytes from a contiguous L1 source to a single remote NoC address
// over fabric, split into <= fabric_max_payload_size packets. Used for both a
// tensor page (its DRAM home in the receiver backing tensor) and the optional
// metadata blob. Adapted from send_async's sender_writer.cpp::write_data_to_remote_core
// (CB push/pop dropped — the persistent kernel owns the single-slot scratch CB).
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

void kernel_main() {
    size_t rt_args_idx = 0;
    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

    // Per-coord base address of the receiver's DRAM backing tensor (the direct
    // fabric-write destination). Appended after the fabric-connection args by
    // build_sender_program; varies per device.
    const uint32_t receiver_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);

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
    // Both headers route to the same downstream chip: the data header carries the
    // bulk page writes, the socket header carries the data-landed atomic-inc.
    fabric_set_unicast_route(data_packet_header_addr, downstream_enc);
    fabric_set_unicast_route(socket_packet_header_addr, downstream_enc);
    const uint32_t receiver_noc_x = downstream_enc.d2d.downstream_noc_x;
    const uint32_t receiver_noc_y = downstream_enc.d2d.downstream_noc_y;

    // The sender backing tensor (read source) and the receiver backing tensor
    // (fabric-write destination) share the per-shard spec, so one set of accessor
    // args serves both — only the base address differs.
    auto input_tensor_accessor = TensorAccessor(input_tensor_accessor_args, input_tensor_addr);
    auto output_tensor_accessor = TensorAccessor(input_tensor_accessor_args, receiver_tensor_addr);
    const uint32_t cb_l1_addr = get_write_ptr(scratch_cb_index);

    volatile tt_l1_ptr uint32_t* termination_semaphore =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_semaphore_addr);
    volatile tt_l1_ptr uint32_t* data_ready_counter =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_ready_counter_addr);
    volatile tt_l1_ptr uint32_t* link_grant = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(link_grant_addr);

    // bytes_acked (this sender's config word): the receiver atomic-incs it per
    // consumed transfer = "you may overwrite the receiver tensor". The overwrite
    // gate below spins on it. bytes_sent lives on the RECEIVER; the sender Flush-
    // atomic-incs it after a transfer lands (data-landed signal).
    volatile tt_l1_ptr uint32_t* bytes_acked =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_socket.bytes_acked_base_addr);
    const uint64_t recv_bytes_sent_noc_addr =
        get_noc_addr(receiver_noc_x, receiver_noc_y, sender_socket.downstream_bytes_sent_addr);

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
    // Monotonic count of transfers issued. It (and the receiver's bytes_acked) wraps
    // over the service's persistent lifetime, so every comparison against it uses a
    // wrap-safe unsigned DIFFERENCE — never a direct `<`/`>`, which mis-orders across
    // the 2^32 wrap.
    uint32_t my_sent = 0;
    // Max transfers that may be outstanding (sent but not yet consumed by the
    // receiver). Single buffer in Step 1; Step 3 (double-buffer) raises this.
    constexpr uint32_t kBufferDepth = 1;
    bool terminated = false;
    while (!terminated) {
        // 1. Idle wait. Proceed only when ALL hold (checked at the iteration
        //    boundary — no transfer in flight, no connection held in LEASE mode):
        //      (a) LEASE: we've been granted the link (link_grant == 1),
        //      (b) the sender worker grid has produced (num_workers more data_ready),
        //      (c) OVERWRITE GATE: fewer than kBufferDepth transfers are outstanding
        //          (sent - acked, wrap-safe unsigned diff), so overwriting the receiver
        //          DRAM is safe. Single-buffer ⇒ wait until nothing is outstanding;
        //          iter 0 passes immediately. Step 3 raises kBufferDepth.
        //    or break on host termination.
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
            // Outstanding = sent - acked in wrap-safe unsigned arithmetic (both
            // counters wrap over the persistent lifetime). Hold off while at least
            // kBufferDepth transfers are still unconsumed.
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

        // 2. LEASE mode: acquire the link for exactly this transfer. The unicast
        //    route + packet headers were set up once at entry and persist in L1, so
        //    open() is all that is needed.
        if constexpr (share_fabric_links) {
            fabric_connection.open();
            fabric_open = true;
        }

        // 3. Write one full tensor's worth of data DIRECTLY to the receiver's DRAM
        //    backing tensor (no receiver L1 FIFO copy). Stage each chunk
        //    DRAM -> scratch CB, then fabric-write each staged page to its DRAM home
        //    on the receiver (output_tensor_accessor.get_noc_addr(page)).
        for (uint32_t chunk = 0; chunk < num_socket_pages; ++chunk) {
            const uint32_t base_page = chunk * pages_per_chunk;
            uint32_t dst = cb_l1_addr;
            for (uint32_t i = 0; i < pages_per_chunk; ++i) {
                noc_async_read(input_tensor_accessor.get_noc_addr(base_page + i), dst, tensor_page_size);
                dst += tensor_page_size;
            }
            noc_async_read_barrier();

            uint32_t src = cb_l1_addr;
            for (uint32_t i = 0; i < pages_per_chunk; ++i) {
                const uint64_t page_dst = output_tensor_accessor.get_noc_addr(base_page + i);
                fabric_write_bytes(fabric_connection, page_dst, src, tensor_page_size, data_packet_header_addr);
                src += tensor_page_size;
            }
        }

        // 3b. Optional metadata. The designated worker wrote the blob into this
        //     service core's L1 at sender_metadata_l1_addr before acking. Fabric-write
        //     it to the receiver's vestigial socket-FIFO L1 base (write_ptr is never
        //     advanced, so the base is a fixed single-slot staging buffer); the
        //     receiver mcasts it to its worker grid after the data-landed signal.
        if constexpr (metadata_enabled) {
            const uint64_t md_dst = get_noc_addr(receiver_noc_x, receiver_noc_y, sender_socket.downstream_fifo_addr);
            fabric_write_bytes(
                fabric_connection, md_dst, sender_metadata_l1_addr, socket_page_size, data_packet_header_addr);
        }

        // 3c. Data-landed signal: Flush-atomic-inc the receiver's bytes_sent counter.
        //     The Flush flag makes the receiver EDM router drain this link's payload
        //     writes (the DRAM pages + the metadata L1 write) to their destinations
        //     before applying the inc, so when the receiver observes the inc the whole
        //     transfer is present. Commutative inc — sets up multi-link (Step 2).
        my_sent += 1;
        socket_packet_header_addr->to_noc_unicast_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{recv_bytes_sent_noc_addr, 1, /*flush=*/true});
        fabric_connection.wait_for_empty_write_slot();
        fabric_connection.send_payload_flush_blocking_from_address(
            (uint32_t)socket_packet_header_addr, sizeof(PACKET_HEADER_TYPE));

        // 4. Release the sender worker grid (consumed_sem) so it can overwrite the
        //    SENDER backing tensor with the next iteration's slice.
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
