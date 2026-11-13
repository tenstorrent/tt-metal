// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "sharded_rmsnorm_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_local_stat_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gather_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scale_src_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scale_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out_id = get_compile_time_arg_val(4);
    constexpr uint32_t cb_tile_out_id = get_compile_time_arg_val(5);
    constexpr uint32_t w_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t num_cores = get_compile_time_arg_val(7);
    constexpr uint32_t stat_tile_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t row_tile_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t arrive_sem_id = get_compile_time_arg_val(10);
    constexpr uint32_t ready_sem_id = get_compile_time_arg_val(11);
    constexpr uint32_t hub_noc_x = get_compile_time_arg_val(12);
    constexpr uint32_t hub_noc_y = get_compile_time_arg_val(13);
    constexpr bool is_rect = get_compile_time_arg_val(14) != 0;
    uint32_t mcast_start_x = get_compile_time_arg_val(15);
    uint32_t mcast_start_y = get_compile_time_arg_val(16);
    uint32_t mcast_end_x = get_compile_time_arg_val(17);
    uint32_t mcast_end_y = get_compile_time_arg_val(18);
    constexpr bool output_is_sharded = get_compile_time_arg_val(19) != 0;
    constexpr bool output_is_rm = get_compile_time_arg_val(20) != 0;
    constexpr auto out_args = TensorAccessorArgs<21>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t core_idx = get_arg_val<uint32_t>(1);
    const bool is_hub = (core_idx == 0);

    const uint32_t arrive_sem_addr = get_semaphore(arrive_sem_id);
    const uint32_t ready_sem_addr = get_semaphore(ready_sem_id);
    volatile tt_l1_ptr uint32_t* arrive_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrive_sem_addr);
    volatile tt_l1_ptr uint32_t* ready_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_sem_addr);

    CircularBuffer cb_local_stat(cb_local_stat_id);
    CircularBuffer cb_gather(cb_gather_id);
    CircularBuffer cb_scale_src(cb_scale_src_id);
    CircularBuffer cb_scale(cb_scale_id);

    if (is_hub) {
        cb_gather.reserve_back(num_cores);
    } else {
        cb_scale.reserve_back(1);
    }

    // 1. Gather per-core sum(x^2) into hub's cb_gather.
    cb_local_stat.wait_front(1);
    const uint32_t gather_addr = cb_gather.get_write_ptr();
    if (is_hub) {
        invalidate_l1_cache();
        sharded_rmsnorm_detail::copy_l1_words(gather_addr, cb_local_stat.get_read_ptr(), stat_tile_bytes);
        cb_local_stat.pop_front(1);
        if constexpr (num_cores > 1) {
            noc_semaphore_wait(arrive_sem_ptr, num_cores - 1);
            noc_semaphore_set(arrive_sem_ptr, 0);
        }
        cb_gather.push_back(num_cores);
    } else {
        noc_async_write(
            cb_local_stat.get_read_ptr(),
            get_noc_addr(hub_noc_x, hub_noc_y, gather_addr + core_idx * stat_tile_bytes),
            stat_tile_bytes);
        noc_async_write_barrier();
        cb_local_stat.pop_front(1);
        noc_semaphore_inc(get_noc_addr(hub_noc_x, hub_noc_y, arrive_sem_addr), 1);
        noc_async_atomic_barrier();
    }

    // 2. Broadcast computed scale (rsqrt(mean(x^2) + eps)) from hub to all cores' cb_scale.
    if (is_hub) {
        cb_scale_src.wait_front(1);
        cb_scale.reserve_back(1);
        const uint32_t scale_src_addr = cb_scale_src.get_read_ptr();
        const uint32_t scale_addr = cb_scale.get_write_ptr();

        if constexpr (num_cores > 1) {
            if constexpr (is_rect) {
                if (noc_index == 1) {
                    std::swap(mcast_start_x, mcast_end_x);
                    std::swap(mcast_start_y, mcast_end_y);
                }
                const uint64_t mcast_scale_addr =
                    get_noc_multicast_addr(mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, scale_addr);
                noc_async_write_multicast(scale_src_addr, mcast_scale_addr, stat_tile_bytes, num_cores - 1);
                noc_async_write_barrier();

                *ready_sem_ptr = 1;
                const uint64_t mcast_sem_addr =
                    get_noc_multicast_addr(mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, ready_sem_addr);
                noc_semaphore_set_multicast(ready_sem_addr, mcast_sem_addr, num_cores - 1);
                noc_async_write_barrier();
                *ready_sem_ptr = 0;
            } else {
                for (uint32_t i = 1; i < num_cores; ++i) {
                    const uint32_t dst_x = get_arg_val<uint32_t>(2 + 2 * (i - 1));
                    const uint32_t dst_y = get_arg_val<uint32_t>(2 + 2 * (i - 1) + 1);
                    noc_async_write(scale_src_addr, get_noc_addr(dst_x, dst_y, scale_addr), stat_tile_bytes);
                }
                noc_async_write_barrier();
                for (uint32_t i = 1; i < num_cores; ++i) {
                    const uint32_t dst_x = get_arg_val<uint32_t>(2 + 2 * (i - 1));
                    const uint32_t dst_y = get_arg_val<uint32_t>(2 + 2 * (i - 1) + 1);
                    noc_semaphore_inc(get_noc_addr(dst_x, dst_y, ready_sem_addr), 1);
                }
                noc_async_atomic_barrier();
            }
        }

        invalidate_l1_cache();
        sharded_rmsnorm_detail::copy_l1_words(scale_addr, scale_src_addr, stat_tile_bytes);
        cb_scale.push_back(1);
        cb_scale_src.pop_front(1);
    } else {
        noc_semaphore_wait(ready_sem_ptr, 1);
        noc_semaphore_set(ready_sem_ptr, 0);
        cb_scale.push_back(1);
    }

    // 3. Emit output according to target layout and memory config.
    if constexpr (output_is_sharded && output_is_rm) {
        // cb_out directly aliases output.buffer(); compute already packed into it.
    } else if constexpr (!output_is_sharded && output_is_rm) {
        CircularBuffer cb_out(cb_out_id);
        cb_out.wait_front(w_tiles);
        const auto out_acc = TensorAccessor(out_args, out_addr);
        const uint32_t out_byte_offset = core_idx * w_tiles * row_tile_bytes;
        noc_async_write(cb_out.get_read_ptr(), out_acc.get_noc_addr(0, out_byte_offset), w_tiles * row_tile_bytes);
        noc_async_write_barrier();
        cb_out.pop_front(w_tiles);
    } else if constexpr (output_is_sharded && !output_is_rm) {
        constexpr uint32_t num_tiles_out = w_tiles / 2;
        sharded_rmsnorm_detail::scatter_row_to_tiles(
            cb_out_id, w_tiles, cb_tile_out_id, get_write_ptr(cb_tile_out_id), 0, num_tiles_out);
    } else {
        constexpr uint32_t kTile32x32Bytes = 32 * 32 * sizeof(uint16_t);
        const uint32_t tile_base = get_write_ptr(cb_tile_out_id);
        sharded_rmsnorm_detail::scatter_row_to_tiles(cb_out_id, w_tiles, cb_tile_out_id, tile_base, 0, w_tiles);
        const auto out_acc = TensorAccessor(out_args, out_addr);
        for (uint32_t t = 0; t < w_tiles; ++t) {
            noc_async_write(
                tile_base + t * kTile32x32Bytes, out_acc.get_noc_addr(core_idx * w_tiles + t, 0), kTile32x32Bytes);
        }
        noc_async_write_barrier();
        CircularBuffer(cb_out_id).pop_front(w_tiles);
    }
}
