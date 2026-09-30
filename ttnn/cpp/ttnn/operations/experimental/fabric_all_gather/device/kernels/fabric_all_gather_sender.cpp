// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sender of a link worker: sends every chunk its reader fetched into the same pages of the neighbour's output,
// one fabric hop. Every inc_every-th packet (and the last) is a fused write + increment of the neighbour link
// worker's arrival counter. Fence between calls: before sending, tell the peer link worker that writes into this
// chip that it may (ready), and wait for the same from this worker's own peer. At the end, wait until everything
// expected from upstream has landed, then re-arm the arrival counter to zero.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "fabric_all_gather_common.hpp"

using namespace tt::tt_fabric;

// One hop: 2D fabrics route by destination fabric node, 1D fabrics by hop count.
FORCE_INLINE void route_one_hop(volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, uint16_t chip_id, uint16_t mesh_id) {
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
    constexpr uint32_t cb_meta = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t run_pages = get_compile_time_arg_val(3);
    constexpr uint32_t num_banks = get_compile_time_arg_val(4);
    constexpr uint32_t group = get_compile_time_arg_val(5);      // chunks per flush; the CB holds 2 x group chunks
    constexpr uint32_t inc_every = get_compile_time_arg_val(6);  // fused write + increment once per this many chunks
    constexpr bool kPrefixFromMetadata = get_compile_time_arg_val(7) != 0;
    constexpr auto out_args = TensorAccessorArgs<8>();
    constexpr auto prefix_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr uint32_t chunk_bytes = run_pages * page_bytes;
    constexpr uint32_t num_headers = group;

    // Common args: [0] output address [1] arrival counter address [2] ready counter address [3..7] unused
    // [8..15] geometry block.
    constexpr uint32_t kGeom = 8;
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t arrival_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t ready_addr = get_common_arg_val<uint32_t>(2);
    const Geometry geo = read_geometry<kPrefixFromMetadata>(kGeom, page_bytes, get_write_ptr(cb_meta), prefix_args);

    // Per-core args: [0] first bank [1] bank stride [2] walk rotation [3] send ready [4, 5] ready target core
    // (the peer's link worker that sends to this chip) [6, 7] peer link worker core [8] peer mesh id [9] peer chip id
    // [10, 11, 12] upstream entries that are whole / first half / second half [13] entries, then entries
    // (rank | part << 16), then the fabric connection (only when the worker sends anything or sends ready).
    size_t a = 0;
    const uint32_t first = get_arg_val<uint32_t>(a++);
    const uint32_t stride = get_arg_val<uint32_t>(a++);
    const uint32_t rot = get_arg_val<uint32_t>(a++);
    const uint32_t send_ready = get_arg_val<uint32_t>(a++);
    const uint32_t ready_x = get_arg_val<uint32_t>(a++);
    const uint32_t ready_y = get_arg_val<uint32_t>(a++);
    const uint32_t peer_x = get_arg_val<uint32_t>(a++);
    const uint32_t peer_y = get_arg_val<uint32_t>(a++);
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint16_t dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint32_t up_whole = get_arg_val<uint32_t>(a++);
    const uint32_t up_first_half = get_arg_val<uint32_t>(a++);
    const uint32_t up_second_half = get_arg_val<uint32_t>(a++);
    const uint32_t num_entries = get_arg_val<uint32_t>(a++);
    const uint32_t entries_idx = a;
    a += num_entries;

    auto count = [&](uint32_t part) {
        return fag::port_chunks(geo.num_stripes, geo.stripe_pages, num_banks, run_pages, first, stride, part);
    };
    // upstream sends its whole shards first, then at most one half (balanced ring)
    const uint32_t expect_in = fag::expected_increments(
        up_whole, count(0), up_first_half + up_second_half, up_first_half ? count(1) : count(2), inc_every);
    uint32_t total = 0;
    for (uint32_t k = 0; k < num_entries; ++k) {
        total += count(get_arg_val<uint32_t>(entries_idx + k) >> 16);
    }

    const auto out = TensorAccessor(out_args, out_addr, page_bytes);
    if (num_entries > 0 || send_ready) {
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
        if (num_entries > 0) {
            volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
            noc_semaphore_wait_min(ready, 1);
            noc_semaphore_inc(get_noc_addr(ready_addr), 0u - 1u);
            noc_async_atomic_barrier();
        }
        const uint64_t arrival_noc = get_noc_addr(peer_x, peer_y, arrival_addr);
        uint32_t chunks_left = total;
        uint32_t sent = 0;
        // A flush group never crosses the end of the CB (2 x group chunks; `offset` is where it starts), so the
        // chunks of one group are contiguous from the read pointer. The reader may push fewer than a group at a time.
        constexpr uint32_t cb_chunks = 2 * group;
        uint32_t h = 0, unflushed = 0, offset = 0;
        for (uint32_t k = 0; k < num_entries; ++k) {
            const uint32_t entry = get_arg_val<uint32_t>(entries_idx + k);
            const uint32_t rank = entry & 0xFFFF;
            const uint32_t entry_chunks = count(entry >> 16);
            uint32_t in_entry = 0;
            fag::for_each_chunk(
                geo.num_stripes,
                geo.stripe_pages,
                num_banks,
                run_pages,
                first,
                stride,
                rot,
                entry >> 16,
                [&](uint32_t stripe, uint32_t page, uint32_t n, uint32_t) {
                    cb_wait_front(cb, run_pages * (unflushed + 1));
                    const uint32_t src = get_read_ptr(cb) + unflushed * chunk_bytes;
                    const uint32_t bytes = n * page_bytes;
                    const uint64_t dst = out.get_noc_addr(output_page(geo, rank, stripe, page), 0, 0);
                    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[h];
                    h = (h + 1 == num_headers) ? 0 : h + 1;
                    // The receiving router issues an increment only after its write lands, which stalls it, so
                    // only every inc_every-th chunk and the last chunk of every entry carry one (see
                    // fag::increments_through).
                    const bool on_grid = (++sent % inc_every) == 0;
                    const bool entry_last = ++in_entry == entry_chunks;
                    if (on_grid || entry_last) {
                        hdr->to_noc_fused_unicast_write_atomic_inc(
                            NocUnicastAtomicIncFusedCommandHeader{dst, arrival_noc, 1, true}, bytes);
                    } else {
                        hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, bytes);
                    }
                    conn.wait_for_empty_write_slot();
                    conn.send_current_slot_non_blocking(src, bytes, reinterpret_cast<uint32_t>(hdr));
                    --chunks_left;
                    const uint32_t cap = group < cb_chunks - offset ? group : cb_chunks - offset;
                    if (++unflushed == cap || chunks_left == 0) {
                        noc_async_writes_flushed();  // sources + headers of the in-flight chunks have left L1
                        cb_pop_front(cb, run_pages * unflushed);
                        offset += unflushed;
                        if (offset >= cb_chunks) {
                            offset -= cb_chunks;
                        }
                        unflushed = 0;
                    }
                });
        }
        conn.close();
    }

    // Everything from upstream has landed (relays already needed part of it); re-arm for the next call.
    if (expect_in > 0) {
        volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_addr);
        noc_semaphore_wait_min(arrived, expect_in);
        noc_semaphore_inc(get_noc_addr(arrival_addr), 0u - expect_in);
        noc_async_atomic_barrier();
    }
}
