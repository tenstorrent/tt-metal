// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sender of a link worker (terms: fabric_all_gather_chunk_walk.hpp):
//   1. fence: signal the downstream link worker that sends back into this chip (ready counter), then wait for our own
//      downstream's signal;
//   2. send every chunk of the send list one fabric hop into the same pages of the next chip's output; the last chunk
//      of each entry (or a bare packet, if the entry has no chunk on this worker's banks) increments the downstream
//      arrival counter;
//   3. wait until all of upstream's entries have landed (the output is complete when the op ends) and reset the
//      arrival counter for the next call.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "fabric_all_gather_common.hpp"

using namespace tt::tt_fabric;

// One hop: 2D fabrics route by destination chip, 1D fabrics by hop count.
FORCE_INLINE void route_one_hop(volatile tt_l1_ptr PACKET_HEADER_TYPE* header, uint16_t chip_id, uint16_t mesh_id) {
#if (defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0)) || defined(EMULE_FABRIC_2D)
    (void)fabric_set_unicast_route(header, chip_id, mesh_id);
#else
    (void)chip_id;
    (void)mesh_id;
    (void)fabric_set_unicast_route<false>(
        reinterpret_cast<volatile tt_l1_ptr LowLatencyPacketHeader*>(header), static_cast<uint16_t>(1));
#endif
}

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t metadata_cb = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t pages_per_chunk = get_compile_time_arg_val(3);
    constexpr uint32_t num_banks = get_compile_time_arg_val(4);
    constexpr uint32_t batch_chunks = get_compile_time_arg_val(5);
    constexpr bool kPrefixFromMetadata = get_compile_time_arg_val(6) != 0;
    constexpr auto output_args = TensorAccessorArgs<7>();
    constexpr auto prefix_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr uint32_t chunk_bytes = pages_per_chunk * page_bytes;
    constexpr uint32_t cb_chunks = 2 * batch_chunks;

    // Common args: [0] output [1] arrival counter [2] ready counter, then the shard geometry.
    const ShardGeometry shard = read_shard_geometry<kPrefixFromMetadata>(get_write_ptr(metadata_cb), prefix_args);
    const auto output = TensorAccessor(output_args, get_common_arg_val<uint32_t>(0), page_bytes);
    const uint32_t arrival_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t ready_addr = get_common_arg_val<uint32_t>(2);

    // Per-core args: [0] first bank [1] bank stride [2] send ready [3, 4] ready target core (downstream's link worker
    // of the opposite direction) [5, 6] downstream link worker core [7] downstream mesh id [8] downstream chip id
    // [9] upstream entries [10] entries, then the send list, then the fabric connection.
    size_t arg = 0;
    const uint32_t first_bank = get_arg_val<uint32_t>(arg++);
    const uint32_t bank_stride = get_arg_val<uint32_t>(arg++);
    const bool send_ready = get_arg_val<uint32_t>(arg++) != 0;
    const uint32_t ready_x = get_arg_val<uint32_t>(arg++);
    const uint32_t ready_y = get_arg_val<uint32_t>(arg++);
    const uint32_t downstream_x = get_arg_val<uint32_t>(arg++);
    const uint32_t downstream_y = get_arg_val<uint32_t>(arg++);
    const uint16_t mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg++));
    const uint16_t chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg++));
    const uint32_t upstream_entries = get_arg_val<uint32_t>(arg++);
    const uint32_t num_entries = get_arg_val<uint32_t>(arg++);
    const uint32_t entries_arg = arg;
    arg += num_entries;

    if (num_entries > 0 || send_ready) {
        auto connection = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg);
        // one header per chunk in flight: a header must not change until its packet has left L1
        volatile tt_l1_ptr PACKET_HEADER_TYPE* headers[batch_chunks];
        for (uint32_t i = 0; i < batch_chunks; ++i) {
            headers[i] = PacketHeaderPool::allocate_header();
            route_one_hop(headers[i], chip_id, mesh_id);
        }
        connection.open();

        // 1. fence
        if (send_ready) {
            headers[0]->to_noc_unicast_atomic_inc(
                NocUnicastAtomicIncCommandHeader{get_noc_addr(ready_x, ready_y, ready_addr), 1, true});
            connection.wait_for_empty_write_slot();
            connection.send_payload_flush_blocking_from_address(
                reinterpret_cast<uint32_t>(headers[0]), sizeof(PACKET_HEADER_TYPE));
        }
        if (num_entries > 0) {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr), 1);
            noc_semaphore_inc(get_noc_addr(ready_addr), 0u - 1u);
            noc_async_atomic_barrier();
        }

        // 2. send, in the reader's batches (full, at the end of the CB, at the end of an entry)
        const uint64_t downstream_arrivals = get_noc_addr(downstream_x, downstream_y, arrival_addr);
        uint32_t cb_offset = 0, next_header = 0;
        for (uint32_t k = 0; k < num_entries; ++k) {
            const uint32_t entry = get_arg_val<uint32_t>(entries_arg + k);
            const uint32_t entry_chunks =
                count_chunks(shard, num_banks, pages_per_chunk, first_bank, bank_stride, entry_half(entry));
            uint32_t batch = 0, sent = 0;
            auto pop = [&]() {
                noc_async_writes_flushed();  // the batch's chunks and headers have left L1
                cb_pop_front(cb, pages_per_chunk * batch);
                cb_offset = (cb_offset + batch) % cb_chunks;
                batch = 0;
            };
            for_each_chunk(
                shard,
                num_banks,
                pages_per_chunk,
                first_bank,
                bank_stride,
                entry_half(entry),
                [&](uint32_t stripe, uint32_t page, uint32_t num_pages) {
                    cb_wait_front(cb, pages_per_chunk * (batch + 1));
                    const uint64_t dst = output.get_noc_addr(output_page(shard, entry_rank(entry), stripe, page), 0, 0);
                    auto* header = headers[next_header];
                    next_header = (next_header + 1) % batch_chunks;
                    if (++sent == entry_chunks) {
                        header->to_noc_fused_unicast_write_atomic_inc(
                            NocUnicastAtomicIncFusedCommandHeader{dst, downstream_arrivals, 1, true},
                            num_pages * page_bytes);
                    } else {
                        header->to_noc_unicast_write(NocUnicastCommandHeader{dst}, num_pages * page_bytes);
                    }
                    connection.wait_for_empty_write_slot();
                    connection.send_current_slot_non_blocking(
                        get_read_ptr(cb) + batch * chunk_bytes,
                        num_pages * page_bytes,
                        reinterpret_cast<uint32_t>(header));
                    const uint32_t batch_cap =
                        batch_chunks < cb_chunks - cb_offset ? batch_chunks : cb_chunks - cb_offset;
                    if (++batch == batch_cap) {
                        pop();
                    }
                });
            if (batch > 0) {
                pop();
            }
            if (entry_chunks == 0) {  // none of this entry is on our banks: still count it downstream
                auto* header = headers[next_header];
                header->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{downstream_arrivals, 1, true});
                connection.wait_for_empty_write_slot();
                connection.send_payload_flush_blocking_from_address(
                    reinterpret_cast<uint32_t>(header), sizeof(PACKET_HEADER_TYPE));
            }
        }
        connection.close();
    }

    // 3. everything from upstream has landed; reset the arrival counter
    if (upstream_entries > 0) {
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_addr), upstream_entries);
        noc_semaphore_inc(get_noc_addr(arrival_addr), 0u - upstream_entries);
        noc_async_atomic_barrier();
    }
}
