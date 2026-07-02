// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Shared A-gather handshake for the partial-width-sharded matmul_decode readers: gather the full A
// onto every core once (reshard K-slice -> loopback-multicast -> coordinator/done semaphores ->
// publish). Compile args 0..19 are the gather config; `ta_base` is the interleaved-A TensorAccessor
// slot (differs per caller). Callers publish their own resident weights before calling this.

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/noc_semaphore.h"

template <uint32_t ta_base>
inline void gather_full_a() {
    using experimental::CircularBuffer;
    using experimental::Noc;
    using experimental::Semaphore;
    using experimental::use;

    constexpr uint32_t in0_cb_index = get_compile_time_arg_val(0);       // this core's A slice (gather source)
    constexpr uint32_t full_in0_cb_index = get_compile_time_arg_val(1);  // gathered full A (destination)
    constexpr uint32_t shard_num_tiles = get_compile_time_arg_val(2);    // tiles per A slice
    constexpr uint32_t tile_size_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t num_senders = get_compile_time_arg_val(4);    // # of A-holding cores
    constexpr uint32_t num_receivers = get_compile_time_arg_val(5);  // cores in the mcast rectangle (incl. self)
    uint32_t mcast_x_start = get_compile_time_arg_val(6);
    uint32_t mcast_y_start = get_compile_time_arg_val(7);
    uint32_t mcast_x_end = get_compile_time_arg_val(8);
    uint32_t mcast_y_end = get_compile_time_arg_val(9);
    constexpr uint32_t gather_sem_id = get_compile_time_arg_val(10);  // bumped by every sender on the coordinator
    constexpr uint32_t done_sem_id = get_compile_time_arg_val(11);    // broadcast by the coordinator to all cores
    constexpr uint32_t coordinator_noc_x = get_compile_time_arg_val(12);
    constexpr uint32_t coordinator_noc_y = get_compile_time_arg_val(13);
    // 14/15: the caller's resident weight CB index/tiles -- published by the caller, not here.
    constexpr uint32_t read_interleaved = get_compile_time_arg_val(16);      // senders NoC-read A's K-slice first
    constexpr uint32_t in0_M_tiles = get_compile_time_arg_val(17);           // A height in tiles
    constexpr uint32_t in0_K_tiles_per_core = get_compile_time_arg_val(18);  // this sender's K-slice width (tiles)
    constexpr uint32_t in0_K_tiles_total = get_compile_time_arg_val(19);     // global A width in tiles (page stride)
    constexpr auto in0_args = TensorAccessorArgs<ta_base>();

    const uint32_t is_sender = get_arg_val<uint32_t>(0);
    const uint32_t sender_id = get_arg_val<uint32_t>(1);
    const uint32_t is_coordinator = get_arg_val<uint32_t>(2);
    const uint32_t in0_buffer_addr = get_arg_val<uint32_t>(3);  // interleaved A base addr (read_interleaved only)

    constexpr uint32_t full_num_tiles = num_senders * shard_num_tiles;
    const uint32_t shard_size_bytes = shard_num_tiles * tile_size_bytes;

    // NOC_1 uses an inverted coordinate system, so the rectangle corners swap.
    if (noc_index == 1) {
        std::swap(mcast_x_start, mcast_x_end);
        std::swap(mcast_y_start, mcast_y_end);
    }

    Noc noc;
    CircularBuffer in0_cb(in0_cb_index);
    CircularBuffer full_in0_cb(full_in0_cb_index);
    Semaphore<> gather_sem(gather_sem_id);
    Semaphore<> done_sem(done_sem_id);

    // reshard_input: this sender NoC-reads its K-slice of the interleaved A into in0_cb in the SAME
    // m-major contiguous order the buffer-backed shard used, so the multicast sees identical bytes.
    if (read_interleaved && is_sender) {
        const auto in0_acc = TensorAccessor(in0_args, in0_buffer_addr, tile_size_bytes);
        in0_cb.reserve_back(shard_num_tiles);
        uint32_t l1_write_addr = in0_cb.get_write_ptr();
        const uint32_t k_base = sender_id * in0_K_tiles_per_core;
        for (uint32_t m = 0; m < in0_M_tiles; ++m) {
            for (uint32_t kk = 0; kk < in0_K_tiles_per_core; ++kk) {
                const uint32_t page = m * in0_K_tiles_total + k_base + kk;
                noc_async_read_tile(page, in0_acc, l1_write_addr);
                l1_write_addr += tile_size_bytes;
            }
        }
        noc_async_read_barrier();
        in0_cb.push_back(shard_num_tiles);
    }

    // Reserve space for the whole A matrix; multicast writes land directly here.
    full_in0_cb.reserve_back(full_num_tiles);

    if (is_sender) {
        // Broadcast this core's contiguous A slice into every core's full_in0_cb at this K-slice's
        // offset (full_in0_cb is allocated identically on all cores, so the dst L1 addr is the same).
        const uint32_t dst_offset_bytes = sender_id * shard_size_bytes;
        noc.async_write_multicast<Noc::McastMode::INCLUDE_SRC>(
            use<CircularBuffer::AddrSelector::READ_PTR>(in0_cb),
            full_in0_cb,
            shard_size_bytes,
            num_receivers,
            {.offset_bytes = 0},
            {.noc_x_start = mcast_x_start,
             .noc_y_start = mcast_y_start,
             .noc_x_end = mcast_x_end,
             .noc_y_end = mcast_y_end,
             .offset_bytes = dst_offset_bytes});
        noc.async_write_barrier();

        if (num_senders > 1) {
            // Report completion to the coordinator by bumping its gather semaphore.
            gather_sem.up(noc, coordinator_noc_x, coordinator_noc_y, 1);
            noc.async_atomic_barrier();
        } else {
            done_sem.set(1);
            done_sem.set_multicast<Noc::McastMode::INCLUDE_SRC>(
                noc, mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end, num_receivers);
            noc.async_write_barrier();
        }
    }

    if (is_coordinator && num_senders > 1) {
        // Wait for every sender to have broadcast its slice, then tell everyone A is ready.
        gather_sem.wait(num_senders);
        done_sem.set(1);
        done_sem.set_multicast<Noc::McastMode::INCLUDE_SRC>(
            noc, mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end, num_receivers);
        noc.async_write_barrier();
    }

    // The full A matrix is now resident on this core; hand it to compute.
    done_sem.wait(1);
    full_in0_cb.push_back(full_num_tiles);
}
