// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sender of a fabric link worker (terms: fabric_all_gather_chunk_walk.hpp):
//   1. fence: tell the downstream link worker that sends back into this chip that this chip has started (its
//      downstream-started counter), then wait for our own downstream's signal;
//   2. send every fabric chunk of the outgoing shards one fabric hop into the same pages of the next chip's output (a
//      chunk is one packet; only a single page larger than the payload takes several); the last packet of each
//      outgoing shard (or a bare packet, if the shard has no chunk on this worker's banks) increments the downstream
//      shards-arrived counter;
//   3. wait until every shard expected from upstream has landed (the output is complete when the op ends) and reset
//      the shards-arrived counter for the next call.

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
    constexpr uint32_t chunk_cb = get_compile_time_arg_val(0);
    constexpr uint32_t metadata_cb = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t pages_per_fabric_chunk = get_compile_time_arg_val(3);
    constexpr uint32_t num_dram_banks = get_compile_time_arg_val(4);
    constexpr uint32_t chunks_per_cb_batch = get_compile_time_arg_val(5);
    constexpr bool kValidPrefixFromMetadata = get_compile_time_arg_val(6) != 0;
    constexpr uint32_t fabric_payload_bytes = get_compile_time_arg_val(7);  // max bytes per fabric packet
    constexpr auto output_args = TensorAccessorArgs<8>();
    constexpr auto valid_prefix_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr uint32_t fabric_chunk_bytes = pages_per_fabric_chunk * page_bytes;
    constexpr uint32_t chunk_slots_in_cb = 2 * chunks_per_cb_batch;

    // Common args: [0] output [1] shards-arrived counter [2] downstream-started counter, then the chip shard geometry.
    const ChipShardGeometry geometry =
        read_chip_shard_geometry<kValidPrefixFromMetadata>(get_write_ptr(metadata_cb), valid_prefix_args);
    const auto output = TensorAccessor(output_args, get_common_arg_val<uint32_t>(0), page_bytes);
    const uint32_t shards_arrived_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t downstream_started_addr = get_common_arg_val<uint32_t>(2);

    // Per-core args: [0] first owned bank [1] owned bank stride [2] signal started to downstream? [3, 4] the core that
    // gets that signal (downstream's link worker of the opposite direction) [5, 6] downstream link worker core (same
    // direction: gets our data and shards-arrived increments) [7] downstream mesh id [8] downstream chip id [9] shards
    // expected from upstream [10] number of outgoing shards, then the outgoing shards, then the fabric connection.
    size_t arg = 0;
    const uint32_t first_owned_bank = get_arg_val<uint32_t>(arg++);
    const uint32_t owned_bank_stride = get_arg_val<uint32_t>(arg++);
    const bool signal_started_to_downstream = get_arg_val<uint32_t>(arg++) != 0;
    const uint32_t started_signal_x = get_arg_val<uint32_t>(arg++);
    const uint32_t started_signal_y = get_arg_val<uint32_t>(arg++);
    const uint32_t downstream_worker_x = get_arg_val<uint32_t>(arg++);
    const uint32_t downstream_worker_y = get_arg_val<uint32_t>(arg++);
    const uint16_t mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg++));
    const uint16_t chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg++));
    const uint32_t shards_expected_from_upstream = get_arg_val<uint32_t>(arg++);
    const uint32_t num_outgoing_shards = get_arg_val<uint32_t>(arg++);
    const uint32_t outgoing_shards_arg = arg;
    arg += num_outgoing_shards;

    if (num_outgoing_shards > 0 || signal_started_to_downstream) {
        auto connection = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg);
        // one header per chunk in a CB batch: a header must not change until its packet has left L1
        volatile tt_l1_ptr PACKET_HEADER_TYPE* headers[chunks_per_cb_batch];
        for (uint32_t i = 0; i < chunks_per_cb_batch; ++i) {
            headers[i] = PacketHeaderPool::allocate_header();
            route_one_hop(headers[i], chip_id, mesh_id);
        }
        connection.open();

        // 1. fence (one round trip per call)
        if (signal_started_to_downstream) {
            headers[0]->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{
                get_noc_addr(started_signal_x, started_signal_y, downstream_started_addr), 1, true});
            connection.wait_for_empty_write_slot();
            connection.send_payload_flush_blocking_from_address(
                reinterpret_cast<uint32_t>(headers[0]), sizeof(PACKET_HEADER_TYPE));
        }
        if (num_outgoing_shards > 0) {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(downstream_started_addr), 1);
            noc_semaphore_inc(get_noc_addr(downstream_started_addr), 0u - 1u);
            noc_async_atomic_barrier();
        }

        // 2. send, in CB batches (full, at the end of the CB, at the end of an outgoing shard)
        const uint64_t downstream_shards_arrived =
            get_noc_addr(downstream_worker_x, downstream_worker_y, shards_arrived_addr);
        uint32_t cb_chunk_offset = 0, next_header = 0, unflushed_headers = 0;
        auto next_free_header = [&]() {  // flush before a header is reused
            if (unflushed_headers == chunks_per_cb_batch) {
                noc_async_writes_flushed();
                unflushed_headers = 0;
            }
            ++unflushed_headers;
            auto* header = headers[next_header];
            next_header = (next_header + 1) % chunks_per_cb_batch;
            return header;
        };
        for (uint32_t outgoing_index = 0; outgoing_index < num_outgoing_shards; ++outgoing_index) {
            const uint32_t outgoing_shard = get_arg_val<uint32_t>(outgoing_shards_arg + outgoing_index);
            const uint32_t shard_fabric_chunks = count_fabric_chunks(
                geometry,
                num_dram_banks,
                pages_per_fabric_chunk,
                first_owned_bank,
                owned_bank_stride,
                outgoing_shard_bank_half(outgoing_shard));
            uint32_t chunks_in_batch = 0, chunks_sent = 0;
            auto pop_batch = [&]() {
                noc_async_writes_flushed();  // the CB batch's chunks and headers have left L1
                unflushed_headers = 0;
                cb_pop_front(chunk_cb, pages_per_fabric_chunk * chunks_in_batch);
                cb_chunk_offset = (cb_chunk_offset + chunks_in_batch) % chunk_slots_in_cb;
                chunks_in_batch = 0;
            };
            for_each_fabric_chunk(
                geometry,
                num_dram_banks,
                pages_per_fabric_chunk,
                first_owned_bank,
                owned_bank_stride,
                outgoing_shard_bank_half(outgoing_shard),
                [&](uint32_t outer_slice, uint32_t page_in_slice, uint32_t num_pages) {
                    cb_wait_front(chunk_cb, pages_per_fabric_chunk * (chunks_in_batch + 1));
                    const uint64_t dst_noc_addr = output.get_noc_addr(
                        output_page_index(geometry, outgoing_shard_rank(outgoing_shard), outer_slice, page_in_slice),
                        0,
                        0);
                    const uint32_t chunk_slot = get_read_ptr(chunk_cb) + chunks_in_batch * fabric_chunk_bytes;
                    const uint32_t chunk_payload_bytes = num_pages * page_bytes;
                    const bool last_chunk_of_shard = ++chunks_sent == shard_fabric_chunks;
                    for (uint32_t offset = 0; offset < chunk_payload_bytes; offset += fabric_payload_bytes) {
                        const uint32_t packet_bytes = chunk_payload_bytes - offset < fabric_payload_bytes
                                                          ? chunk_payload_bytes - offset
                                                          : fabric_payload_bytes;
                        auto* header = next_free_header();
                        if (last_chunk_of_shard && offset + packet_bytes == chunk_payload_bytes) {
                            header->to_noc_fused_unicast_write_atomic_inc(
                                NocUnicastAtomicIncFusedCommandHeader{
                                    dst_noc_addr + offset, downstream_shards_arrived, 1, true},
                                packet_bytes);
                        } else {
                            header->to_noc_unicast_write(NocUnicastCommandHeader{dst_noc_addr + offset}, packet_bytes);
                        }
                        connection.wait_for_empty_write_slot();
                        connection.send_current_slot_non_blocking(
                            chunk_slot + offset, packet_bytes, reinterpret_cast<uint32_t>(header));
                    }
                    const uint32_t batch_capacity = chunks_per_cb_batch < chunk_slots_in_cb - cb_chunk_offset
                                                        ? chunks_per_cb_batch
                                                        : chunk_slots_in_cb - cb_chunk_offset;
                    if (++chunks_in_batch == batch_capacity) {
                        pop_batch();
                    }
                });
            if (chunks_in_batch > 0) {
                pop_batch();
            }
            if (shard_fabric_chunks == 0) {  // none of this shard is on our banks: still count it downstream
                auto* header = next_free_header();
                header->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{downstream_shards_arrived, 1, true});
                connection.wait_for_empty_write_slot();
                connection.send_payload_flush_blocking_from_address(
                    reinterpret_cast<uint32_t>(header), sizeof(PACKET_HEADER_TYPE));
            }
        }
        connection.close();
    }

    // 3. everything from upstream has landed; reset the shards-arrived counter
    if (shards_expected_from_upstream > 0) {
        noc_semaphore_wait_min(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(shards_arrived_addr), shards_expected_from_upstream);
        noc_semaphore_inc(get_noc_addr(shards_arrived_addr), 0u - shards_expected_from_upstream);
        noc_async_atomic_barrier();
    }
}
