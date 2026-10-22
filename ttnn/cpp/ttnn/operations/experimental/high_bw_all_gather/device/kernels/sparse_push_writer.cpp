// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sparse KV push writer. One fabric channel per (link, direction): every packet whose first hop leaves in that
// direction goes out on it, so the four directions inject in parallel. Each channel's K senders feed one fabric mux
// core that owns the router connection.
//
// Protocol (worker 0 of every chip is the sync core):
//   start: sender 0 of each link-0 channel tells the chips in its direction it has started (so each chip hears
//          once from every other chip); the sync core waits for num_devices - 1, then releases its local senders.
//          No output is overwritten while its owner still runs an earlier op.
//   end:   every sender increments done on the sync core of every chip in its direction (on the same route as
//          its data) and on its own chip's; the sync core waits for (num_devices - 1) * links * K + workers and
//          consumes it, leaving both semaphores at zero.

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "sparse_push_route.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"

#include <cstdint>

using namespace ttnn::operations::ccl::common;

namespace {
constexpr uint32_t kData = 0, kMeta = 1, kHeaders = 4;
constexpr uint32_t kBatch = 4;  // packets in flight per flush; matches the reader's push batches

template <typename Sender>
void send_sem_inc(
    Sender& conn,
    volatile PACKET_HEADER_TYPE* hdr,
    uint32_t dest_chip,
    uint32_t dest_mesh,
    uint64_t sem_noc_addr,
    bool flip_turn,
    uint32_t src) {
    fabric_set_unicast_route((HybridMeshPacketHeader*)hdr, dest_chip, dest_mesh);
    if (flip_turn) {
        sparse_push::flip_turn<MESH_ROWS, MESH_COLS>((volatile HybridMeshPacketHeader*)hdr, src);
    }
    hdr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sem_noc_addr, 1, true});
    conn.wait_for_empty_write_slot();
    conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdr), sizeof(PACKET_HEADER_TYPE));
}
}  // namespace

void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t num_devices = get_compile_time_arg_val(1);
    constexpr uint32_t my_rank = get_compile_time_arg_val(2);
    constexpr uint32_t workers = get_compile_time_arg_val(3);
    constexpr uint32_t packet_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t links = get_compile_time_arg_val(5);
    constexpr auto output_args = TensorAccessorArgs<6>();
    constexpr uint32_t mux_ct = output_args.next_compile_time_args_offset();
    constexpr uint8_t mux_num_buffers = get_compile_time_arg_val(mux_ct + 0);
    constexpr size_t mux_buffer_bytes = get_compile_time_arg_val(mux_ct + 1);
    constexpr size_t mux_status_address = get_compile_time_arg_val(mux_ct + 2);
    constexpr size_t mux_termination_address = get_compile_time_arg_val(mux_ct + 3);
    constexpr uint32_t per_channel = get_compile_time_arg_val(mux_ct + 4);  // senders per (link, direction)

    constexpr uint32_t dest_chip_ids[num_devices] = DEST_CHIP_ID;
    constexpr uint32_t dest_mesh_ids[num_devices] = DEST_MESH_ID;

    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t ready_sem = get_common_arg_val<uint32_t>(1);
    const uint32_t done_sem = get_common_arg_val<uint32_t>(2);

    size_t arg = 0;
    const uint32_t worker = get_arg_val<uint32_t>(arg++);
    const uint32_t dir = get_arg_val<uint32_t>(arg++);
    const uint32_t link = get_arg_val<uint32_t>(arg++);
    const uint32_t sub = get_arg_val<uint32_t>(arg++);
    const uint32_t sync_x = get_arg_val<uint32_t>(arg++);
    const uint32_t sync_y = get_arg_val<uint32_t>(arg++);
    const size_t cores_arg = arg;  // workers x (noc x, noc y)
    arg += 2 * workers;

    // Same route choice as the reader (sparse_push_route.hpp), so increments follow the data.
    auto route_to = [&](uint32_t d) {
        return sparse_push::route<MESH_ROWS, MESH_COLS>(
            my_rank, d, static_cast<uint32_t>(get_next_hop_router_direction(dest_mesh_ids[d], dest_chip_ids[d])));
    };
    auto in_my_direction = [&](uint32_t d) { return d != my_rank && route_to(d).dir == dir; };

    const uint32_t headers_l1 = get_write_ptr(kHeaders);
    auto* sem_hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(headers_l1);
    const auto output = TensorAccessor(output_args, output_addr, page_bytes);
    auto* ready_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_sem);
    auto* done_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(done_sem);

    auto run = [&](auto& conn) {
        // Start fence.
        if (link == 0 && sub == 0) {
            const uint64_t remote_ready = get_noc_addr(sync_x, sync_y, ready_sem);
            for (uint32_t d = 0; d < num_devices; ++d) {
                if (in_my_direction(d)) {
                    send_sem_inc(
                        conn,
                        sem_hdr,
                        dest_chip_ids[d],
                        dest_mesh_ids[d],
                        remote_ready,
                        route_to(d).flip_turn,
                        my_rank);
                }
            }
        }
        if (worker == 0) {
            noc_semaphore_wait_min(ready_ptr, num_devices - 1);
            noc_semaphore_inc(get_noc_addr(ready_sem), uint32_t{0} - (num_devices - 1));
            for (uint32_t w = 1; w < workers; ++w) {
                const uint32_t x = get_arg_val<uint32_t>(cores_arg + 2 * w);
                const uint32_t y = get_arg_val<uint32_t>(cores_arg + 2 * w + 1);
                noc_semaphore_inc(get_noc_addr(x, y, ready_sem), 1);
            }
            noc_async_atomic_barrier();
        } else {
            noc_semaphore_wait_min(ready_ptr, 1);
            noc_semaphore_inc(get_noc_addr(ready_sem), uint32_t{0} - 1u);
            noc_async_atomic_barrier();
        }

        // Packets: up to kBatch sends in flight, each with its own header slot; one flush retires the batch.
        uint32_t pending = 0;
        auto retire = [&]() {
            if (pending) {
                noc_async_writes_flushed();
                cb_pop_front(kMeta, pending);
                cb_pop_front(kData, pending);
                pending = 0;
            }
        };
        while (true) {
            cb_wait_front(kMeta, pending + 1);
            cb_wait_front(kData, pending + 1);
            invalidate_l1_cache();
            const auto* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(kMeta) + pending * 16);
            const uint32_t dest = meta[0], first_page = meta[1], n = meta[2];
            if (n == 0) {
                retire();
                cb_pop_front(kMeta, 1);
                cb_pop_front(kData, 1);
                break;
            }
            const uint32_t data_l1 = get_read_ptr(kData) + pending * packet_bytes;
            const uint32_t bytes = n * page_bytes;
            if (dest == my_rank) {
                noc_async_write(data_l1, output.get_noc_addr(first_page), bytes);
            } else {
                auto* hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(
                    headers_l1 + (1 + pending) * sizeof(PACKET_HEADER_TYPE));
                fabric_set_unicast_route((HybridMeshPacketHeader*)hdr, dest_chip_ids[dest], dest_mesh_ids[dest]);
                if (route_to(dest).flip_turn) {
                    sparse_push::flip_turn<MESH_ROWS, MESH_COLS>((volatile HybridMeshPacketHeader*)hdr, my_rank);
                }
                tt::tt_fabric::linear::to_noc_unicast_write(bytes, hdr, first_page, output);
                perform_payload_send<false, false>(conn, data_l1, bytes, hdr);
            }
            if (++pending == kBatch) {
                retire();
            }
        }

        // Completion: on the same route as the data, so each increment lands after this sender's rows.
        noc_async_write_barrier();
        const uint64_t remote_done = get_noc_addr(sync_x, sync_y, done_sem);
        for (uint32_t d = 0; d < num_devices; ++d) {
            if (d == my_rank) {
                noc_semaphore_inc(remote_done, 1);
            } else if (in_my_direction(d)) {
                send_sem_inc(
                    conn, sem_hdr, dest_chip_ids[d], dest_mesh_ids[d], remote_done, route_to(d).flip_turn, my_rank);
            }
        }
        noc_async_atomic_barrier();
    };

    const bool is_termination_master = get_arg_val<uint32_t>(arg + 1) != 0;
    const uint8_t mux_x = get_arg_val<uint32_t>(arg + 2);
    const uint8_t mux_y = get_arg_val<uint32_t>(arg + 3);
    const uint32_t termination_sync_id = get_arg_val<uint32_t>(arg + 10);
    const uint32_t local_status_address = get_semaphore(get_arg_val<uint32_t>(arg + 11));
    auto conn = tt::tt_fabric::build_connection_to_fabric_endpoint<mux_num_buffers>(
        mux_x,
        mux_y,
        get_arg_val<uint32_t>(arg + 9),
        mux_num_buffers,
        mux_buffer_bytes,
        get_arg_val<uint32_t>(arg + 4),
        get_arg_val<uint32_t>(arg + 5),
        get_arg_val<uint32_t>(arg + 6),
        get_arg_val<uint32_t>(arg + 7),
        get_arg_val<uint32_t>(arg + 8),
        get_semaphore(get_arg_val<uint32_t>(arg + 12)),
        get_semaphore(get_arg_val<uint32_t>(arg + 13)),
        get_semaphore(get_arg_val<uint32_t>(arg + 14)));
    const uint8_t master_x = get_arg_val<uint32_t>(arg + 15);
    const uint8_t master_y = get_arg_val<uint32_t>(arg + 16);
    tt::tt_fabric::wait_for_fabric_endpoint_ready(mux_x, mux_y, mux_status_address, local_status_address);
    tt::tt_fabric::fabric_client_connect(conn);
    run(conn);
    // The mux forwards asynchronously: wait for every channel slot to drain before disconnecting.
    while (conn.get_num_free_write_slots() != mux_num_buffers) {
    }
    tt::tt_fabric::fabric_client_disconnect(conn);
    auto* term_sync = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(termination_sync_id));
    if (is_termination_master) {
        noc_semaphore_wait_min(term_sync, per_channel - 1);
        noc_semaphore_set(term_sync, 0);
        tt::tt_fabric::fabric_endpoint_terminate(mux_x, mux_y, mux_termination_address);
    } else {
        noc_semaphore_inc(get_noc_addr(master_x, master_y, get_semaphore(termination_sync_id)), 1);
        noc_async_atomic_barrier();
    }
    if (worker == 0) {
        constexpr uint32_t expected = (num_devices - 1) * links * per_channel + workers;
        noc_semaphore_wait_min(done_ptr, expected);
        noc_semaphore_inc(get_noc_addr(done_sem), uint32_t{0} - expected);
        noc_async_atomic_barrier();
    }
}
