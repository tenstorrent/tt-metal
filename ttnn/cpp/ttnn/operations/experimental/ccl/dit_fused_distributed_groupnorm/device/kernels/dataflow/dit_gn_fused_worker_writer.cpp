// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Worker writer for fused distributed GroupNorm AG (forwarder model).
 *
 * Contiguous fp32 stick (num_groups * 16 B) — no face-row packing.
 *   1. Wait compute's local stats stick; NoC-write into forwarder packet_buf + slot
 *   2. Wait go-sem (gather landed in DRAM)
 *   3. Read ring_size sticks from DRAM into stats_gathered_cb; push to compute
 *   4. Drain output_cb tiles to the output tensor
 */

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t output_cb = get_compile_time_arg_val(0);
constexpr uint32_t num_tile_cols = get_compile_time_arg_val(1);
constexpr uint32_t block_size = get_compile_time_arg_val(2);
constexpr uint32_t stats_local_cb = get_compile_time_arg_val(3);
constexpr uint32_t stats_gathered_cb = get_compile_time_arg_val(4);
constexpr uint32_t ring_size = get_compile_time_arg_val(5);
constexpr uint32_t total_num_tile_rows = get_compile_time_arg_val(6);
constexpr uint32_t max_rounds = get_compile_time_arg_val(7);
constexpr uint32_t stick_bytes = get_compile_time_arg_val(8);
constexpr uint32_t num_chunks_per_device = get_compile_time_arg_val(9);
constexpr uint32_t packet_cb = get_compile_time_arg_val(10);
constexpr uint32_t arrival_sem_id = get_compile_time_arg_val(11);
constexpr uint32_t go_sem_id = get_compile_time_arg_val(12);
constexpr uint32_t epsilon_cb = get_compile_time_arg_val(13);
constexpr uint32_t eps_bits = get_compile_time_arg_val(14);
constexpr auto output_args = TensorAccessorArgs<15>();
constexpr auto stats_dram_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();

void kernel_main() {
    size_t arg_idx = 0;
    const uint32_t output_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t tile_row_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t tile_row_end = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t stats_dram_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t fwd_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t fwd_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t my_forwarder_index = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t my_slot = get_arg_val<uint32_t>(arg_idx++);

    const uint32_t fwd_packet_buf_addr = get_write_ptr(packet_cb);
    const uint32_t packet_slot_bytes = get_tile_size(packet_cb);
    const uint32_t fwd_arrival_sem_addr = get_semaphore(arrival_sem_id);
    const uint32_t go_sem_addr = get_semaphore(go_sem_id);

    const uint32_t output_tile_bytes = get_tile_size(output_cb);
    const auto output_accessor = TensorAccessor(output_args, output_addr);
    const auto stats_dram = TensorAccessor(stats_dram_args, stats_dram_addr);

    // Populate epsilon CB once (fp32 scalar tile: first face element = eps).
    {
        cb_reserve_back(epsilon_cb, 1);
        volatile tt_l1_ptr uint32_t* eps_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(epsilon_cb));
        eps_ptr[0] = eps_bits;
        cb_push_back(epsilon_cb, 1);
    }

    volatile tt_l1_ptr uint32_t* go_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_sem_addr);
    const uint64_t fwd_arrival_noc = safe_get_noc_addr(fwd_x, fwd_y, fwd_arrival_sem_addr, 0);

    // Single AG round for GroupNorm (max_rounds==1): one stick for the whole tensor.
    {
        DeviceZoneScopedN("GN_W_PUSH");
        cb_wait_front(stats_local_cb, 1);
        const uint32_t src = get_read_ptr(stats_local_cb);
        const uint32_t dst = fwd_packet_buf_addr + my_slot * stick_bytes;
        const uint64_t dst_noc = safe_get_noc_addr(fwd_x, fwd_y, dst, 0);
        noc_async_write(src, dst_noc, stick_bytes);
        noc_async_write_barrier();
        noc_semaphore_inc(fwd_arrival_noc, 1);
        noc_async_atomic_barrier();
        cb_pop_front(stats_local_cb, 1);
    }

    {
        DeviceZoneScopedN("GN_W_AGWAIT");
        noc_semaphore_wait_min(go_sem_ptr, 1);
    }

    {
        DeviceZoneScopedN("GN_W_GATHER");
        cb_reserve_back(stats_gathered_cb, ring_size);
        const uint32_t gbase = get_write_ptr(stats_gathered_cb);
        for (uint32_t d = 0; d < ring_size; d++) {
            const uint32_t page_idx = d * num_chunks_per_device + my_forwarder_index * max_rounds + 0;
            const uint64_t src = get_noc_addr(page_idx, stats_dram, my_slot * stick_bytes);
            noc_async_read(src, gbase + d * stick_bytes, stick_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(stats_gathered_cb, ring_size);
    }

    // Drain all output tiles produced by POST.
    {
        DeviceZoneScopedN("GN_W_DRAIN");
        for (uint32_t tile_row = tile_row_start; tile_row < tile_row_end; tile_row++) {
            for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                const uint32_t tiles_in_block =
                    ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
                cb_wait_front(output_cb, block_size);
                uint32_t rd = get_read_ptr(output_cb);
                for (uint32_t i = 0; i < tiles_in_block; i++) {
                    const uint32_t out_idx = tile_row * num_tile_cols + col_tile + i;
                    noc_async_write_tile(out_idx, output_accessor, rd);
                    rd += output_tile_bytes;
                }
                noc_async_writes_flushed();
                cb_pop_front(output_cb, block_size);
            }
        }
    }
    noc_async_write_barrier();
    noc_semaphore_set(go_sem_ptr, 0);
    (void)total_num_tile_rows;
    (void)packet_slot_bytes;
}
