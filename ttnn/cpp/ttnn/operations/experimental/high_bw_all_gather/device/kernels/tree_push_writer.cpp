// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PROTOTYPE multicast-tree push writer (see tree_push_reader.cpp). Every packet goes one hop, to the torus
// neighbour in this sender's direction, as one contiguous bank-lane write.
//
// Protocol (worker 0 of every chip is the sync core):
//   start:   the sync core zeroes this chip's arrival counters and raises `go` on every local worker; then each
//            link-0, k-0 sender tells its neighbour it has started. The sync core waits for its four neighbours and
//            releases the local senders, so no neighbour writes here before the counters are reset.
//   packets: after every kIncEvery packets and at each level end, the sender adds the packets it has delivered to
//            its own counter on the neighbour's sync core, on the same connection as the data.

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"

#include <cstdint>

using namespace ttnn::operations::ccl::common;

namespace {
constexpr uint32_t kData = 0, kMeta = 1, kHeaders = 4, kCounters = 5;
constexpr uint32_t kBatch = 4;  // packets in flight per flush; matches the reader's push batches
constexpr uint32_t kSlots = 8;
constexpr uint32_t kIncEvery = 4;
constexpr uint32_t kStreams = 16;
constexpr uint32_t kNeighbours = 4;
enum : uint32_t { kEnd = 0, kSendInput = 1, kSendOutput = 2, kLocal = 3, kLevel = 6 };

template <typename Sender>
void send_sem_inc(Sender& conn, volatile PACKET_HEADER_TYPE* hdr, uint64_t sem_noc_addr, uint32_t val) {
    hdr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sem_noc_addr, val, true});
    conn.wait_for_empty_write_slot();
    conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdr), sizeof(PACKET_HEADER_TYPE));
}
}  // namespace

void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t workers = get_compile_time_arg_val(1);
    constexpr uint32_t packet_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t go_sem_id = get_compile_time_arg_val(3);
    constexpr auto output_args = TensorAccessorArgs<4>();
    constexpr uint32_t mux_ct = output_args.next_compile_time_args_offset();
    constexpr bool use_mux = get_compile_time_arg_val(mux_ct + 0) != 0;
    constexpr uint8_t mux_num_buffers = get_compile_time_arg_val(mux_ct + 1);
    constexpr size_t mux_buffer_bytes = get_compile_time_arg_val(mux_ct + 2);
    constexpr size_t mux_status_address = get_compile_time_arg_val(mux_ct + 3);
    constexpr size_t mux_termination_address = get_compile_time_arg_val(mux_ct + 4);
    constexpr uint32_t per_channel = get_compile_time_arg_val(mux_ct + 5);
    static_assert(workers <= kStreams, "one arrival counter per neighbour sender");

    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t ready_sem = get_common_arg_val<uint32_t>(1);

    size_t arg = 0;
    const uint32_t worker = get_arg_val<uint32_t>(arg++);
    const uint32_t link = get_arg_val<uint32_t>(arg++);
    const uint32_t sub = get_arg_val<uint32_t>(arg++);
    const uint32_t sync_x = get_arg_val<uint32_t>(arg++);
    const uint32_t sync_y = get_arg_val<uint32_t>(arg++);
    const uint32_t nbr_chip = get_arg_val<uint32_t>(arg++);
    const uint32_t nbr_mesh = get_arg_val<uint32_t>(arg++);
    const size_t cores_arg = arg;  // workers x (noc x, noc y)
    arg += 2 * workers;

    const uint32_t headers_l1 = get_write_ptr(kHeaders);
    for (uint32_t h = 0; h < 1 + kBatch; ++h) {
        fabric_set_unicast_route(
            reinterpret_cast<HybridMeshPacketHeader*>(headers_l1 + h * sizeof(PACKET_HEADER_TYPE)), nbr_chip, nbr_mesh);
    }
    auto* sem_hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(headers_l1);
    const auto output = TensorAccessor(output_args, output_addr, page_bytes);
    auto* ready_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_sem);
    const uint32_t go_addr = get_semaphore(go_sem_id);
    const uint32_t counters_l1 = get_write_ptr(kCounters);
    const uint64_t my_counter_there = get_noc_addr(sync_x, sync_y, counters_l1 + worker * 4);

    auto run = [&](auto& conn) {
        if (worker == 0) {
            auto* counters = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counters_l1);
            for (uint32_t s = 0; s < kStreams; ++s) {
                counters[s] = 0;
            }
            for (uint32_t w = 0; w < workers; ++w) {
                const uint32_t x = get_arg_val<uint32_t>(cores_arg + 2 * w);
                const uint32_t y = get_arg_val<uint32_t>(cores_arg + 2 * w + 1);
                noc_semaphore_inc(get_noc_addr(x, y, go_addr), 2);
            }
            noc_async_atomic_barrier();
        }
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_addr), 1);
        noc_semaphore_inc(get_noc_addr(go_addr), uint32_t{0} - 1u);
        noc_async_atomic_barrier();
        if (link == 0 && sub == 0) {
            send_sem_inc(conn, sem_hdr, get_noc_addr(sync_x, sync_y, ready_sem), 1);
        }
        if (worker == 0) {
            noc_semaphore_wait_min(ready_ptr, kNeighbours);
            noc_semaphore_inc(get_noc_addr(ready_sem), uint32_t{0} - kNeighbours);
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

        uint32_t pending = 0, rpos = 0, unreported = 0;
        auto retire = [&]() {
            if (pending) {
                noc_async_writes_flushed();
                cb_pop_front(kMeta, pending);
                cb_pop_front(kData, pending);
                rpos = (rpos + pending) % kSlots;
                pending = 0;
            }
        };
        auto report = [&]() {
            if (unreported) {
                send_sem_inc(conn, sem_hdr, my_counter_there, unreported);
                unreported = 0;
            }
        };
        while (true) {
            cb_wait_front(kMeta, pending + 1);
            cb_wait_front(kData, pending + 1);
            invalidate_l1_cache();
            const auto* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(kMeta) + pending * 16);
            const uint32_t kind = meta[0], target = meta[1], n = meta[2];
            if (kind == kEnd || kind == kLevel) {
                retire();
                report();
                cb_pop_front(kMeta, 1);
                cb_pop_front(kData, 1);
                rpos = (rpos + 1) % kSlots;
                if (kind == kEnd) {
                    break;
                }
                continue;
            }
            const uint32_t data_l1 = get_read_ptr(kData) + pending * packet_bytes;
            const uint32_t bytes = n * page_bytes;
            if (kind == kLocal) {
                noc_async_write(data_l1, output.get_noc_addr(target), bytes);
            } else {
                auto* hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(
                    headers_l1 + (1 + pending) * sizeof(PACKET_HEADER_TYPE));
                tt::tt_fabric::linear::to_noc_unicast_write(bytes, hdr, target, output);
                perform_payload_send<false, false>(conn, data_l1, bytes, hdr);
                if (++unreported == kIncEvery) {
                    report();
                }
            }
            if (++pending == kBatch || rpos + pending == kSlots) {
                retire();
            }
        }
        noc_async_write_barrier();
        noc_async_atomic_barrier();
    };

    if constexpr (use_mux) {
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
    } else {
        auto conn = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg);
        conn.open();
        run(conn);
        conn.close();
    }
}
