// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "sparse_push_constants.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"

#include <cstdint>

using namespace ttnn::operations::ccl::common;

using sparse_push::kBatch;
using sparse_push::kData;
using sparse_push::kHeaders;
using sparse_push::kMeta;
using sparse_push::kMetaEntryBytes;
using sparse_push::PacketMeta;

// The writer of one sender core, paired with sparse_push_table_reader.cpp: sends each packet the reader hands it as one
// fabric unicast write into the requester's compact buffer (a local NoC write for this chip's own rows).
//
// Two handshakes, on caller-owned semaphores at each chip's sync core (worker 0); both are left at zero for reuse:
//   start: a chip writes into other chips' buffers only after all of them have started this op, so no requester's
//          buffer is overwritten while its previous op still reads it. Every chip increments every other chip's start
//          count; worker 0 waits for all of them, then releases its other senders.
//   done:  after its rows, each sender increments the done count of every requester it served, on the same route as
//          its data. The fabric delivers in order along a route, so a count lands after the rows it covers; worker 0
//          of the requester waits for all counts, and the op ends once they are in.

namespace {
// Sends a header-only packet that increments the semaphore at sem_noc_addr on chip (dest_mesh, dest_chip), on the same
// route as this sender's data.
template <typename Sender>
void send_sem_inc(
    Sender& mux_conn, volatile PACKET_HEADER_TYPE* hdr, uint32_t dest_chip, uint32_t dest_mesh, uint64_t sem_noc_addr) {
    fabric_set_unicast_route((HybridMeshPacketHeader*)hdr, dest_chip, dest_mesh);
    hdr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sem_noc_addr, 1, true});
    mux_conn.wait_for_empty_write_slot();
    mux_conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdr), sizeof(PACKET_HEADER_TYPE));
}
}  // namespace

void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t num_devices = get_compile_time_arg_val(1);
    constexpr uint32_t workers = get_compile_time_arg_val(2);
    constexpr uint32_t packet_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t links = get_compile_time_arg_val(4);
    constexpr auto output_args = TensorAccessorArgs<5>();
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
    const uint32_t sender_index = get_arg_val<uint32_t>(arg++);  // among this channel's senders
    const uint32_t my_rank = get_arg_val<uint32_t>(arg++);
    // NoC coordinates of the sync core (worker 0); the same on every chip, so they address remote chips' sync cores
    // too.
    const uint32_t sync_core_x = get_arg_val<uint32_t>(arg++);
    const uint32_t sync_core_y = get_arg_val<uint32_t>(arg++);
    const size_t cores_arg = arg;  // workers x (noc x, noc y)
    arg += 2 * workers;

    // The fabric's next hop, as in the reader, so increments follow the data.
    auto next_hop = [&](uint32_t requester) {
        return static_cast<uint32_t>(get_next_hop_router_direction(dest_mesh_ids[requester], dest_chip_ids[requester]));
    };
    auto in_my_direction = [&](uint32_t requester) { return requester != my_rank && next_hop(requester) == dir; };

    const uint32_t headers_l1 = get_write_ptr(kHeaders);
    auto* sem_hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(headers_l1);
    const auto output = TensorAccessor(output_args, output_addr, page_bytes);
    auto* ready_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_sem);
    auto* done_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(done_sem);

    auto run = [&](auto& mux_conn) {
        // Start fence.
        if (link == 0 && sender_index == 0) {
            const uint64_t remote_ready = get_noc_addr(sync_core_x, sync_core_y, ready_sem);
            for (uint32_t requester = 0; requester < num_devices; ++requester) {
                if (in_my_direction(requester)) {
                    send_sem_inc(mux_conn, sem_hdr, dest_chip_ids[requester], dest_mesh_ids[requester], remote_ready);
                }
            }
        }
        if (worker == 0) {
            noc_semaphore_wait_min(ready_ptr, num_devices - 1);
            // Reset to zero: semaphores only count up, so add the negated count (it wraps).
            noc_semaphore_inc(get_noc_addr(ready_sem), uint32_t{0} - (num_devices - 1));
            for (uint32_t w = 1; w < workers; ++w) {
                const uint32_t x = get_arg_val<uint32_t>(cores_arg + 2 * w);
                const uint32_t y = get_arg_val<uint32_t>(cores_arg + 2 * w + 1);
                noc_semaphore_inc(get_noc_addr(x, y, ready_sem), 1);
            }
            noc_async_atomic_barrier();
        } else {
            noc_semaphore_wait_min(ready_ptr, 1);
            noc_semaphore_inc(get_noc_addr(ready_sem), uint32_t{0} - 1u);  // reset to zero
            noc_async_atomic_barrier();
        }

        // Packets: up to kBatch sends in flight, each with its own header slot; one flush retires the batch.
        uint32_t in_flight = 0;
        // Once the batch's writes have left L1, frees its kData/kMeta slots for the reader.
        auto retire = [&]() {
            if (in_flight) {
                noc_async_writes_flushed();
                cb_pop_front(kMeta, in_flight);
                cb_pop_front(kData, in_flight);
                in_flight = 0;
            }
        };
        while (true) {
            cb_wait_front(kMeta, in_flight + 1);
            cb_wait_front(kData, in_flight + 1);
            invalidate_l1_cache();
            const auto* meta =
                reinterpret_cast<volatile tt_l1_ptr PacketMeta*>(get_read_ptr(kMeta) + in_flight * kMetaEntryBytes);
            const uint32_t requester = meta->requester, first_page = meta->first_page, packet_rows = meta->rows;
            if (packet_rows == 0) {  // the reader's end marker (PacketBuilder::finish)
                retire();
                cb_pop_front(kMeta, 1);
                cb_pop_front(kData, 1);
                break;
            }
            const uint32_t data_l1 = get_read_ptr(kData) + in_flight * packet_bytes;
            const uint32_t bytes = packet_rows * page_bytes;
            if (requester == my_rank) {
                noc_async_write(data_l1, output.get_noc_addr(first_page), bytes);
            } else {
                // Header slot 0 belongs to the semaphore increments; in-flight packet k uses slot 1 + k.
                auto* hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(
                    headers_l1 + (1 + in_flight) * sizeof(PACKET_HEADER_TYPE));
                fabric_set_unicast_route(
                    (HybridMeshPacketHeader*)hdr, dest_chip_ids[requester], dest_mesh_ids[requester]);
                tt::tt_fabric::linear::to_noc_unicast_write(bytes, hdr, first_page, output);
                perform_payload_send<false, false>(mux_conn, data_l1, bytes, hdr);
            }
            if (++in_flight == kBatch) {
                retire();
            }
        }

        // Completion: on the same route as the data, so each increment lands after this sender's rows.
        noc_async_write_barrier();
        const uint64_t remote_done = get_noc_addr(sync_core_x, sync_core_y, done_sem);
        for (uint32_t requester = 0; requester < num_devices; ++requester) {
            if (requester == my_rank) {
                noc_semaphore_inc(remote_done, 1);  // this chip's own: a local increment
            } else if (in_my_direction(requester)) {
                send_sem_inc(mux_conn, sem_hdr, dest_chip_ids[requester], dest_mesh_ids[requester], remote_done);
            }
        }
        noc_async_atomic_barrier();
    };

    // The mux connection, in the order ttnn::ccl::fabric_mux_connection_rt_args lays it out.
    arg++;  // mux_connection_valid: always set
    const bool is_termination_master = get_arg_val<uint32_t>(arg++) != 0;
    const uint8_t fabric_mux_x = get_arg_val<uint32_t>(arg++);
    const uint8_t fabric_mux_y = get_arg_val<uint32_t>(arg++);
    const size_t fabric_mux_channel_base_address = get_arg_val<uint32_t>(arg++);
    const size_t fabric_mux_connection_info_address = get_arg_val<uint32_t>(arg++);
    const size_t fabric_mux_connection_handshake_address = get_arg_val<uint32_t>(arg++);
    const size_t fabric_mux_flow_control_address = get_arg_val<uint32_t>(arg++);
    const size_t fabric_mux_buffer_index_address = get_arg_val<uint32_t>(arg++);
    const uint8_t fabric_mux_channel_id = get_arg_val<uint32_t>(arg++);
    const uint32_t termination_sync_id = get_arg_val<uint32_t>(arg++);
    const uint32_t local_fabric_mux_status_address = get_semaphore(get_arg_val<uint32_t>(arg++));
    const uint32_t local_flow_control_address = get_semaphore(get_arg_val<uint32_t>(arg++));
    const uint32_t local_teardown_address = get_semaphore(get_arg_val<uint32_t>(arg++));
    const uint32_t local_buffer_index_address = get_semaphore(get_arg_val<uint32_t>(arg++));
    const uint8_t termination_master_noc_x = get_arg_val<uint32_t>(arg++);
    const uint8_t termination_master_noc_y = get_arg_val<uint32_t>(arg++);
    auto mux_conn = tt::tt_fabric::build_connection_to_fabric_endpoint<mux_num_buffers>(
        fabric_mux_x,
        fabric_mux_y,
        fabric_mux_channel_id,
        mux_num_buffers,
        mux_buffer_bytes,
        fabric_mux_channel_base_address,
        fabric_mux_connection_info_address,
        fabric_mux_connection_handshake_address,
        fabric_mux_flow_control_address,
        fabric_mux_buffer_index_address,
        local_flow_control_address,
        local_teardown_address,
        local_buffer_index_address);
    tt::tt_fabric::wait_for_fabric_endpoint_ready(
        fabric_mux_x, fabric_mux_y, mux_status_address, local_fabric_mux_status_address);
    tt::tt_fabric::fabric_client_connect(mux_conn);
    run(mux_conn);
    // The mux forwards asynchronously: wait for every channel slot to drain before disconnecting.
    while (mux_conn.get_num_free_write_slots() != mux_num_buffers) {
    }
    tt::tt_fabric::fabric_client_disconnect(mux_conn);
    // The channel's senders share one mux. Sender 0 (the termination master) waits until its partners have
    // disconnected, then shuts the mux down; the others just report in.
    auto* term_sync = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(termination_sync_id));
    if (is_termination_master) {
        noc_semaphore_wait_min(term_sync, per_channel - 1);
        noc_semaphore_set(term_sync, 0);
        tt::tt_fabric::fabric_endpoint_terminate(fabric_mux_x, fabric_mux_y, mux_termination_address);
    } else {
        noc_semaphore_inc(
            get_noc_addr(termination_master_noc_x, termination_master_noc_y, get_semaphore(termination_sync_id)), 1);
        noc_async_atomic_barrier();
    }
    // Worker 0 waits for this chip's done count: one increment from every remote sender whose channel serves this
    // chip ((N - 1) * links * per_channel), plus one from each of this chip's own senders. Then it resets the count
    // (semaphores only count up, so it adds the negated count, which wraps to zero).
    if (worker == 0) {
        constexpr uint32_t done_count = (num_devices - 1) * links * per_channel + workers;
        noc_semaphore_wait_min(done_ptr, done_count);
        noc_semaphore_inc(get_noc_addr(done_sem), uint32_t{0} - done_count);
        noc_async_atomic_barrier();
    }
}
