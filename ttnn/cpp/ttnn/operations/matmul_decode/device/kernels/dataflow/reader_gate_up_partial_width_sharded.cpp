// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused gate+up partial-width-sharded matmul A-gather reader.
//
// Identical to reader_partial_width_sharded.cpp (gather the full A onto every core ONCE via the
// reshard + loopback-multicast handshake) but additionally publishes the SECOND resident weight
// (up_b / in1b) alongside the first (gate_b / in1). A is gathered exactly once and shared by both
// the gate and up partial matmuls -- the whole point of the fused op.
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/noc_semaphore.h"
using experimental::CircularBuffer;
using experimental::Noc;
using experimental::Semaphore;
using experimental::use;

void kernel_main() {
    constexpr uint32_t in0_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t full_in0_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t shard_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t tile_size_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t num_senders = get_compile_time_arg_val(4);
    constexpr uint32_t num_receivers = get_compile_time_arg_val(5);
    uint32_t mcast_x_start = get_compile_time_arg_val(6);
    uint32_t mcast_y_start = get_compile_time_arg_val(7);
    uint32_t mcast_x_end = get_compile_time_arg_val(8);
    uint32_t mcast_y_end = get_compile_time_arg_val(9);
    constexpr uint32_t gather_sem_id = get_compile_time_arg_val(10);
    constexpr uint32_t done_sem_id = get_compile_time_arg_val(11);
    constexpr uint32_t coordinator_noc_x = get_compile_time_arg_val(12);
    constexpr uint32_t coordinator_noc_y = get_compile_time_arg_val(13);
    constexpr uint32_t in1_cb_index = get_compile_time_arg_val(14);
    constexpr uint32_t in1_num_tiles = get_compile_time_arg_val(15);
    constexpr uint32_t read_interleaved = get_compile_time_arg_val(16);
    constexpr uint32_t in0_M_tiles = get_compile_time_arg_val(17);
    constexpr uint32_t in0_K_tiles_per_core = get_compile_time_arg_val(18);
    constexpr uint32_t in0_K_tiles_total = get_compile_time_arg_val(19);
    constexpr uint32_t in1b_cb_index = get_compile_time_arg_val(20);  // second resident weight (up_b)
    constexpr auto in0_args = TensorAccessorArgs<21>();

    const uint32_t is_sender = get_arg_val<uint32_t>(0);
    const uint32_t sender_id = get_arg_val<uint32_t>(1);
    const uint32_t is_coordinator = get_arg_val<uint32_t>(2);
    const uint32_t in0_buffer_addr = get_arg_val<uint32_t>(3);

    constexpr uint32_t full_num_tiles = num_senders * shard_num_tiles;
    const uint32_t shard_size_bytes = shard_num_tiles * tile_size_bytes;

    if (noc_index == 1) {
        std::swap(mcast_x_start, mcast_x_end);
        std::swap(mcast_y_start, mcast_y_end);
    }

    Noc noc;
    CircularBuffer in0_cb(in0_cb_index);
    CircularBuffer in1_cb(in1_cb_index);
    CircularBuffer in1b_cb(in1b_cb_index);
    CircularBuffer full_in0_cb(full_in0_cb_index);
    Semaphore<> gather_sem(gather_sem_id);
    Semaphore<> done_sem(done_sem_id);

    // Both weights are already resident in L1; just publish them to compute.
    in1_cb.reserve_back(in1_num_tiles);
    in1_cb.push_back(in1_num_tiles);
    in1b_cb.reserve_back(in1_num_tiles);
    in1b_cb.push_back(in1_num_tiles);

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

    full_in0_cb.reserve_back(full_num_tiles);

    if (is_sender) {
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
        gather_sem.wait(num_senders);
        done_sem.set(1);
        done_sem.set_multicast<Noc::McastMode::INCLUDE_SRC>(
            noc, mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end, num_receivers);
        noc.async_write_barrier();
    }

    done_sem.wait(1);
    full_in0_cb.push_back(full_num_tiles);
}
