// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Port sender (BRISC, NoC1) — TRANSPORT, adapted from examples/fabric_all_gather's sender.
//
// fence_block: tell the peer port that writes into this device that it may (ready), wait for my own peer's ready.
// send: every chunk the reader staged goes into the downstream device's landing slot of its source; the last chunk
//       of every (entry, source) item is a fused write + increment of the downstream port's arrival counter (one
//       increment per item: that is the landing notification the downstream relays and compute readers wait on).
// drain_and_rearm: wait until every item from upstream landed and every compute reader of this device stopped
//       polling my counter (consumers_done), then re-arm both counters for the next call.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "ring_joint_sdpa_chunk_walk.hpp"

using namespace tt::tt_fabric;

// One hop: 2D fabrics route by destination fabric node, 1D fabrics by hop count (ROUTING_MODE from the build).
inline void route_one_hop(volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, uint16_t chip_id, uint16_t mesh_id) {
#if (defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0)) || defined(EMULE_FABRIC_2D)
    (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
#else
    (void)chip_id;
    (void)mesh_id;
    (void)fabric_set_unicast_route<false>(
        reinterpret_cast<volatile tt_l1_ptr LowLatencyPacketHeader*>(hdr), static_cast<uint16_t>(1));
#endif
}

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t run_pages = get_compile_time_arg_val(2);
    constexpr uint32_t num_banks = get_compile_time_arg_val(3);
    constexpr uint32_t group = get_compile_time_arg_val(4);
    constexpr auto lk_args = TensorAccessorArgs<5>();
    constexpr auto lv_args = TensorAccessorArgs<lk_args.next_compile_time_args_offset()>();
    constexpr uint32_t chunk_bytes = run_pages * page_bytes;
    constexpr uint32_t num_headers = group;

    size_t a = 0;
    const uint32_t lk_addr = get_arg_val<uint32_t>(a++);
    const uint32_t lv_addr = get_arg_val<uint32_t>(a++);
    const uint32_t kpe = get_arg_val<uint32_t>(a++);
    const uint32_t vpe = get_arg_val<uint32_t>(a++);
    const uint32_t first_bank = get_arg_val<uint32_t>(a++);
    const uint32_t bank_stride = get_arg_val<uint32_t>(a++);
    const uint32_t arrival_addr =
        get_arg_val<uint32_t>(a++);                         // my counter; the downstream port's is at the same address
    const uint32_t expect_in = get_arg_val<uint32_t>(a++);  // increments I receive from upstream this call
    const uint32_t total = get_arg_val<uint32_t>(a++);      // chunks I send
    const uint32_t item_chunks = get_arg_val<uint32_t>(a++);
    const uint32_t ready_addr = get_arg_val<uint32_t>(a++);
    const uint32_t send_ready = get_arg_val<uint32_t>(a++);
    const uint32_t ready_x = get_arg_val<uint32_t>(a++);
    const uint32_t ready_y = get_arg_val<uint32_t>(a++);
    const uint32_t peer_x = get_arg_val<uint32_t>(a++);
    const uint32_t peer_y = get_arg_val<uint32_t>(a++);
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint16_t dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint32_t cdone_addr = get_arg_val<uint32_t>(a++);
    const uint32_t num_consumers = get_arg_val<uint32_t>(a++);
    const uint32_t num_entries = get_arg_val<uint32_t>(a++);
    const uint32_t num_sources = get_arg_val<uint32_t>(a++);
    const uint32_t has_conn = get_arg_val<uint32_t>(a++);
    const uint32_t slots_idx = a;  // downstream landing slot of source k
    a += num_sources;

    if (has_conn) {
        const auto lk = TensorAccessor(lk_args, lk_addr, page_bytes);
        const auto lv = TensorAccessor(lv_args, lv_addr, page_bytes);
        const uint32_t k_slot_pages = num_entries * kpe;
        const uint32_t v_slot_pages = num_entries * vpe;
        auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);  // appended last
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[num_headers];
        for (uint32_t h = 0; h < num_headers; ++h) {
            hdrs[h] = PacketHeaderPool::allocate_header();
            route_one_hop(hdrs[h], dst_chip_id, dst_mesh_id);
        }
        conn.open();
        // ---- fence_block ----
        if (send_ready) {
            hdrs[0]->to_noc_unicast_atomic_inc(
                NocUnicastAtomicIncCommandHeader{get_noc_addr(ready_x, ready_y, ready_addr), 1, true});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_blocking_from_address(
                reinterpret_cast<uint32_t>(hdrs[0]), sizeof(PACKET_HEADER_TYPE));
        }
        if (num_sources > 0) {  // the peer sent me exactly one ready iff I send to it
            volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
            noc_semaphore_wait_min(ready, 1);
            noc_semaphore_inc(get_noc_addr(ready_addr), 0u - 1u);
            noc_async_atomic_barrier();

            const uint64_t arrival_noc = get_noc_addr(peer_x, peer_y, arrival_addr);
            uint32_t chunks_left = total;
            uint32_t h = 0, unflushed = 0;
            for (uint32_t e = 0; e < num_entries; ++e) {
                for (uint32_t k = 0; k < num_sources; ++k) {
                    const uint32_t slot = get_arg_val<uint32_t>(slots_idx + k);
                    uint32_t item_sent = 0;
                    auto send = [&](uint64_t dst, uint32_t n) {
                        cb_wait_front(cb, run_pages * (unflushed + 1));
                        const uint32_t src = get_read_ptr(cb) + unflushed * chunk_bytes;
                        const uint32_t bytes = n * page_bytes;
                        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[h];
                        h = (h + 1 == num_headers) ? 0 : h + 1;
                        if (++item_sent == item_chunks) {
                            hdr->to_noc_fused_unicast_write_atomic_inc(
                                NocUnicastAtomicIncFusedCommandHeader{dst, arrival_noc, 1, true}, bytes);
                        } else {
                            hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, bytes);
                        }
                        conn.wait_for_empty_write_slot();
                        conn.send_current_slot_non_blocking(src, bytes, reinterpret_cast<uint32_t>(hdr));
                        --chunks_left;
                        if (++unflushed == group || chunks_left == 0) {
                            noc_async_writes_flushed();  // sources + headers of the in-flight chunks left L1
                            cb_pop_front(cb, run_pages * unflushed);
                            unflushed = 0;
                        }
                    };
                    for_each_chunk(kpe, first_bank, bank_stride, num_banks, run_pages, [&](uint32_t page, uint32_t n) {
                        send(lk.get_noc_addr(slot * k_slot_pages + e * kpe + page, 0, 0), n);
                    });
                    for_each_chunk(vpe, first_bank, bank_stride, num_banks, run_pages, [&](uint32_t page, uint32_t n) {
                        send(lv.get_noc_addr(slot * v_slot_pages + e * vpe + page, 0, 0), n);
                    });
                }
            }
        }
        conn.close();
    }

    // ---- drain_and_rearm ----
    // Every compute reader of this device must have stopped polling my arrival counter before it is re-armed
    // (consumers_done), and every upstream item must have landed.
    volatile tt_l1_ptr uint32_t* cdone = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cdone_addr);
    noc_semaphore_wait_min(cdone, num_consumers);
    if (expect_in > 0) {
        volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_addr);
        noc_semaphore_wait_min(arrived, expect_in);
        noc_semaphore_inc(get_noc_addr(arrival_addr), 0u - expect_in);
    }
    noc_semaphore_inc(get_noc_addr(cdone_addr), 0u - num_consumers);
    noc_async_atomic_barrier();
}
