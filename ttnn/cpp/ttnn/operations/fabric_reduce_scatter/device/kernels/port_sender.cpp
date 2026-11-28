// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Port sender (BRISC, NoC1): every chunk of cb::OUT (the sums, or a line end's partials) into the downstream chip's
// scratch. Three increment streams: the relay blocks count on the downstream port's counter (A), the last block (the
// downstream chip's own) on the counter (A forward, B backward) of the downstream final core that owns the chunk's
// bank; every inc_every-th chunk and each stream's last carries one. After sending: wait for everything upstream sends
// me, then re-arm my counter.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "ttnn/cpp/ttnn/operations/fabric_reduce_scatter/device/kernels/fabric_reduce_scatter_args.hpp"
#include "ttnn/cpp/ttnn/operations/fabric_reduce_scatter/device/kernels/fabric_reduce_scatter_chunk_walk.hpp"

using namespace tt::tt_fabric;
using namespace fabric_reduce_scatter;

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
    constexpr uint32_t page_bytes = get_compile_time_arg_val(ct::PAGE_BYTES);
    constexpr uint32_t run_pages = get_compile_time_arg_val(ct::RUN_PAGES);
    constexpr uint32_t num_banks = get_compile_time_arg_val(ct::NUM_BANKS);
    constexpr uint32_t group = get_compile_time_arg_val(ct::GROUP);
    constexpr uint32_t inc_every = get_compile_time_arg_val(ct::INC_EVERY);
    constexpr auto scr_args = TensorAccessorArgs<ct::COUNT>();
    constexpr uint32_t chunk_bytes = run_pages * page_bytes;
    constexpr uint32_t num_headers = group;

    const uint32_t scr_addr = get_common_arg_val<uint32_t>(port_sender_common::SCR_ADDR);
    const uint32_t blk_pages = get_common_arg_val<uint32_t>(port_sender_common::BLK_PAGES);
    // A: my counter, and the downstream port's
    const uint32_t arrival_addr = get_common_arg_val<uint32_t>(port_sender_common::ARRIVAL_ADDR);
    const uint32_t ready_addr = get_common_arg_val<uint32_t>(port_sender_common::READY_ADDR);
    const uint32_t first_bank = get_arg_val<uint32_t>(port_sender_rt::FIRST_BANK);
    const uint32_t bank_stride = get_arg_val<uint32_t>(port_sender_rt::BANK_STRIDE);
    const uint32_t final_sem_addr = get_arg_val<uint32_t>(port_sender_rt::FINAL_SEM_ADDR);
    const uint32_t expect_in = get_arg_val<uint32_t>(port_sender_rt::EXPECT_IN);
    const uint32_t full = get_arg_val<uint32_t>(port_sender_rt::FULL);
    const uint32_t send_ready = get_arg_val<uint32_t>(port_sender_rt::SEND_READY);
    const uint32_t ready_x = get_arg_val<uint32_t>(port_sender_rt::READY_X);
    const uint32_t ready_y = get_arg_val<uint32_t>(port_sender_rt::READY_Y);
    const uint32_t peer_x = get_arg_val<uint32_t>(port_sender_rt::PEER_X);
    const uint32_t peer_y = get_arg_val<uint32_t>(port_sender_rt::PEER_Y);
    const uint32_t final_x = get_arg_val<uint32_t>(port_sender_rt::FINAL_X);
    const uint32_t final_y = get_arg_val<uint32_t>(port_sender_rt::FINAL_Y);
    const uint32_t final_x1 = get_arg_val<uint32_t>(port_sender_rt::FINAL_X1);  // final core of my set's second half
    const uint32_t final_y1 = get_arg_val<uint32_t>(port_sender_rt::FINAL_Y1);
    const uint32_t full_f0 = get_arg_val<uint32_t>(port_sender_rt::FULL_F0);  // last block's chunks per final core
    const uint32_t full_f1 = get_arg_val<uint32_t>(port_sender_rt::FULL_F1);
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(port_sender_rt::DST_MESH_ID));
    const uint16_t dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(port_sender_rt::DST_CHIP_ID));
    const uint32_t nblocks = get_arg_val<uint32_t>(port_sender_rt::NUM_BLOCKS);  // entries: block | slot << 16
    constexpr uint32_t blocks_idx = port_sender_rt::COUNT;
    size_t a = blocks_idx + nblocks;  // the fabric connection's args follow the block list
    const auto scr = TensorAccessor(scr_args, scr_addr, page_bytes);

    if (nblocks > 0 || send_ready) {
        auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[num_headers];
        for (uint32_t h = 0; h < num_headers; ++h) {
            hdrs[h] = PacketHeaderPool::allocate_header();
            route_one_hop(hdrs[h], dst_chip_id, dst_mesh_id);
        }
        conn.open();
        if (send_ready) {
            hdrs[0]->to_noc_unicast_atomic_inc(
                NocUnicastAtomicIncCommandHeader{get_noc_addr(ready_x, ready_y, ready_addr), 1, true});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_blocking_from_address(
                reinterpret_cast<uint32_t>(hdrs[0]), sizeof(PACKET_HEADER_TYPE));
        }
        if (nblocks > 0) {
            volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
            noc_semaphore_wait_min(ready, 1);
            noc_semaphore_inc(get_noc_addr(ready_addr), 0u - 1u);
            noc_async_atomic_barrier();
        }
        const uint64_t relay_noc = get_noc_addr(peer_x, peer_y, arrival_addr);
        const uint64_t final_noc = get_noc_addr(final_x, final_y, final_sem_addr);
        const uint64_t final_noc1 = get_noc_addr(final_x1, final_y1, final_sem_addr);
        const uint32_t relay_total = (nblocks - 1) * full;
        uint32_t chunks_left = nblocks * full;
        uint32_t sent_relay = 0, sent_f0 = 0, sent_f1 = 0;
        uint32_t h = 0, unflushed = 0;
        for (uint32_t k = 0; k < nblocks; ++k) {
            const uint32_t entry = get_arg_val<uint32_t>(blocks_idx + k);
            const uint32_t base = (entry >> BLOCK_SLOT_SHIFT) * blk_pages;
            const bool fin = k + 1 == nblocks;
            for_each_chunk(
                blk_pages, first_bank, bank_stride, num_banks, run_pages, [&](uint32_t page, uint32_t n, uint32_t) {
                    cb_wait_front(cb::OUT, run_pages * (unflushed + 1));
                    const uint32_t src = get_read_ptr(cb::OUT) + unflushed * chunk_bytes;
                    const uint32_t bytes = n * page_bytes;
                    const uint64_t dst = scr.get_noc_addr(base + page, 0, 0);
                    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[h];
                    h = (h + 1 == num_headers) ? 0 : h + 1;
                    bool inc;
                    uint64_t ctr;
                    if (fin) {
                        // final core h owns banks first_bank + (2 i + h) * bank_stride of my set
                        const bool h1 = (((page % num_banks) - first_bank) / bank_stride) & 1;
                        uint32_t& sent = h1 ? sent_f1 : sent_f0;
                        ++sent;
                        inc = (sent % inc_every) == 0 || sent == (h1 ? full_f1 : full_f0);
                        ctr = h1 ? final_noc1 : final_noc;
                    } else {
                        ++sent_relay;
                        inc = (sent_relay % inc_every) == 0 || sent_relay == relay_total;
                        ctr = relay_noc;
                    }
                    if (inc) {
                        hdr->to_noc_fused_unicast_write_atomic_inc(
                            NocUnicastAtomicIncFusedCommandHeader{dst, ctr, 1, true}, bytes);
                    } else {
                        hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, bytes);
                    }
                    conn.wait_for_empty_write_slot();
                    conn.send_current_slot_non_blocking(src, bytes, reinterpret_cast<uint32_t>(hdr));
                    --chunks_left;
                    if (++unflushed == group || chunks_left == 0) {
                        noc_async_writes_flushed();
                        cb_pop_front(cb::OUT, run_pages * unflushed);
                        unflushed = 0;
                    }
                });
        }
        conn.close();
    }
    if (expect_in > 0) {
        volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_addr);
        noc_semaphore_wait_min(arrived, expect_in);
        noc_semaphore_inc(get_noc_addr(arrival_addr), 0u - expect_in);
        noc_async_atomic_barrier();
    }
}
