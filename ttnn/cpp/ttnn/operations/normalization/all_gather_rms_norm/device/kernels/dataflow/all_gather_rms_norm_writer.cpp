// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer dataflow kernel for the fused all_gather_rms_norm op (designated-gather, no mux).
//
// ring_size == 1: write the normalized output tiles from cb_out back to DRAM (head-split scatter).
// ring_size  > 1: this worker owns one tile-row (chunk size 1). Per row it:
//   1. copies its local stat partial into its OWN gathered-stats slot (slot = ring_index),
//   2. NoC-relays the partial + destination metadata (dst gathered-slot + peer out-ready NoC addresses) into
//      its assigned gather core's per-worker relay slot and bumps that gather core's relay_ready semaphore,
//   3. waits on its out-ready semaphore for the (ring_size-1) peer partials (delivered by the peers' gather
//      cores straight into this worker's gathered-stats), releases cb_gathered_stats to the compute kernel,
//   4. drains this row's normalized output to DRAM (head-split scatter).
// The dedicated gather cores own the direct fabric connections and do the cross-device line-multicast; the
// worker never touches the fabric (so no mux, no per-worker fabric channel limit -> full-grid compute).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"

void kernel_main() {
    // Compile-time args (order must match writer_ct_args in the program factory).
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);
    constexpr uint32_t ring_size = get_compile_time_arg_val(2);
    constexpr uint32_t cb_local_stats = get_compile_time_arg_val(3);
    constexpr uint32_t cb_gathered_stats = get_compile_time_arg_val(4);
    constexpr uint32_t Wt = get_compile_time_arg_val(5);
    constexpr uint32_t ring_index = get_compile_time_arg_val(6);
    // Head-split output addressing: width-tile w of global tile-row gr -> head h = w / head_dim_tiles,
    // within-head tile e = w % head_dim_tiles, output page = h*per_head_stride + gr*head_dim_tiles + e.
    constexpr uint32_t head_dim_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t m_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t relay_ready_sem_id = get_compile_time_arg_val(9);  // on the gather core
    constexpr uint32_t slot_stride_bytes = get_compile_time_arg_val(10);  // gather-core relay slot stride
    constexpr uint32_t done_sem_id = get_compile_time_arg_val(11);        // our local back-pressure sem
    constexpr auto dst_args = TensorAccessorArgs<12>();
    constexpr uint32_t per_head_stride = m_tiles * head_dim_tiles;

    (void)cb_local_stats;
    (void)cb_gathered_stats;
    (void)ring_index;
    (void)relay_ready_sem_id;
    (void)slot_stride_bytes;

    // Runtime args.
    uint32_t ai = 0;
    const uint32_t dst_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t num_tiles = get_arg_val<uint32_t>(ai++);
    const uint32_t tile_offset = get_arg_val<uint32_t>(ai++);

    const auto s = TensorAccessor(dst_args, dst_addr);
    const uint32_t tile_bytes = get_tile_size(cb_out);

#if defined(RING_GT_1)
    // ----- designated-gather all-gather (ring_size > 1) -----
    const uint32_t out_ready_sem_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t out_ready_wait_value = get_arg_val<uint32_t>(ai++);  // = ring_size - 1 (peers)
    const uint32_t gather_x = get_arg_val<uint32_t>(ai++);              // virtual coords of this worker's gather core
    const uint32_t gather_y = get_arg_val<uint32_t>(ai++);
    const uint32_t relay_base = get_arg_val<uint32_t>(ai++);     // relay buffer base on the gather core
    const uint32_t my_relay_slot = get_arg_val<uint32_t>(ai++);  // this worker's slot index on the gather core

    const uint32_t stat_tile_bytes = get_tile_size(cb_local_stats);
    const uint32_t NCHt = num_tiles / Wt;  // == 1 for the chunk-size-1 worker assignment

    volatile tt_l1_ptr uint32_t* out_ready_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_addr);
    // Back-pressure: the gather core bumps this each round once it has consumed (read) our relay slot, so we
    // can reuse the slot for the next row. done accumulates -> before relaying row ncht we wait done >= ncht.
    volatile tt_l1_ptr uint32_t* done_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(done_sem_id));
    // Where the gather core writes this worker's relay (partial then 16B metadata).
    const uint32_t relay_slot = relay_base + my_relay_slot * slot_stride_bytes;
    const uint64_t relay_payload_noc = get_noc_addr(gather_x, gather_y, relay_slot);
    const uint64_t relay_meta_noc = get_noc_addr(gather_x, gather_y, relay_slot + stat_tile_bytes);
    const uint64_t relay_ready_noc = safe_get_noc_addr(gather_x, gather_y, get_semaphore(relay_ready_sem_id));

    uint32_t gr = tile_offset / Wt;  // global tile-row index of this worker's first row (for head-split)
    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        noc_semaphore_wait_min(done_sem_ptr, ncht);  // gather core has freed our relay slot from the prior row
        cb_wait_front(cb_local_stats, 1);
        cb_reserve_back(cb_gathered_stats, ring_size);
        const uint32_t local_base = get_read_ptr(cb_local_stats);
        const uint32_t gathered_base = get_write_ptr(cb_gathered_stats);
        const uint32_t my_slot_addr = gathered_base + ring_index * stat_tile_bytes;

        // local copy of our partial into our own gathered slot (L1->L1 loopback)
        const uint64_t my_slot_noc_addr = get_noc_addr(my_x[noc_index], my_y[noc_index], my_slot_addr);
        noc_async_write(local_base, my_slot_noc_addr, stat_tile_bytes);

        // The peer worker sits at the same coords, so these (own-coord) NoC addresses route to the peer's
        // matching gathered slot / out-ready semaphore once the gather core line-multicasts them over fabric.
        const uint64_t out_ready_noc_addr_pkt =
            safe_get_noc_addr(my_x[noc_index], my_y[noc_index], out_ready_sem_addr, 0);

        // Relay the partial + metadata to our gather core, then signal it.
        noc_async_write(local_base, relay_payload_noc, stat_tile_bytes);
        noc_inline_dw_write(relay_meta_noc + 0, static_cast<uint32_t>(my_slot_noc_addr & 0xFFFFFFFF));
        noc_inline_dw_write(relay_meta_noc + 4, static_cast<uint32_t>(my_slot_noc_addr >> 32));
        noc_inline_dw_write(relay_meta_noc + 8, static_cast<uint32_t>(out_ready_noc_addr_pkt & 0xFFFFFFFF));
        noc_inline_dw_write(relay_meta_noc + 12, static_cast<uint32_t>(out_ready_noc_addr_pkt >> 32));
        noc_async_writes_flushed();
        noc_semaphore_inc(relay_ready_noc, 1);

        cb_pop_front(cb_local_stats, 1);

        // wait for the (ring_size-1) peer partials to land in our gathered slots, then reset
        noc_semaphore_wait(out_ready_sem_ptr, out_ready_wait_value);
        noc_semaphore_set(out_ready_sem_ptr, 0);

        cb_push_back(cb_gathered_stats, ring_size);

        // ----- drain this row's output tiles back to DRAM (head-split scatter) -----
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            cb_wait_front(cb_out, blk);
            uint32_t write_offset = 0;
            for (uint32_t j = 0; j < blk; j++) {
                const uint32_t w = wt + j;
                const uint32_t h = w / head_dim_tiles;
                const uint32_t out_id = h * per_head_stride + gr * head_dim_tiles + (w - h * head_dim_tiles);
                noc_async_write_tile(out_id, s, get_read_ptr(cb_out) + write_offset);
                write_offset += tile_bytes;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, blk);
        }
        gr++;
    }
#else
    // ----- single device (ring_size == 1): write output, head-split scatter (blk divides Wt). -----
    for (uint32_t i = 0; i < num_tiles; i += blk) {
        const uint32_t gl0 = tile_offset + i;  // global linear tile index of this block's first tile
        const uint32_t gr = gl0 / Wt;          // global tile-row
        const uint32_t w0 = gl0 - gr * Wt;     // width-tile of the block start
        cb_wait_front(cb_out, blk);
        uint32_t write_offset = 0;
        for (uint32_t j = 0; j < blk; j++) {
            const uint32_t w = w0 + j;
            const uint32_t h = w / head_dim_tiles;
            const uint32_t out_id = h * per_head_stride + gr * head_dim_tiles + (w - h * head_dim_tiles);
            noc_async_write_tile(out_id, s, get_read_ptr(cb_out) + write_offset);
            write_offset += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, blk);
    }
#endif
}
