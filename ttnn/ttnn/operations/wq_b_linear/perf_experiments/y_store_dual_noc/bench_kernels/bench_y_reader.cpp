// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Isolated y-store bench, NCRISC/NoC0 (the op's reader). Stands in for compute: fills cb_y with the core's expected
// y tiles from src, then all active cores sync (gather on the master, multicast release) so every store starts at
// once, then pushes cb_y. Then stores the reader's share: it only peeks cb_y (wait_front, no pop); the writer pops
// once y_done says these writes landed.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/ttnn/operations/wq_b_linear/perf_experiments/y_store_dual_noc/kernels/wq_b_linear_y_store.hpp"

void kernel_main() {
    constexpr uint32_t cb_y_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_m_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tensor_n_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t y_reader_share_tenths = get_compile_time_arg_val(3);
    constexpr auto y_split_order = static_cast<YSplitOrder>(get_compile_time_arg_val(4));
    constexpr uint32_t y_done_sem_id = get_compile_time_arg_val(5);
    constexpr uint32_t gather_sem_id = get_compile_time_arg_val(6);
    constexpr uint32_t release_sem_id = get_compile_time_arg_val(7);
    constexpr uint32_t num_active = get_compile_time_arg_val(8);
    constexpr uint32_t num_rect = get_compile_time_arg_val(9);
    constexpr auto y_args = TensorAccessorArgs<10>();
    constexpr auto src_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();

    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_start = get_arg_val<uint32_t>(1);
    const uint32_t core_n_tiles = get_arg_val<uint32_t>(2);
    const uint32_t n_stride = get_arg_val<uint32_t>(3);
    const uint32_t src_addr = get_arg_val<uint32_t>(4);
    const uint32_t master_x = get_arg_val<uint32_t>(5);
    const uint32_t master_y = get_arg_val<uint32_t>(6);
    const bool is_master = get_arg_val<uint32_t>(7) != 0;
    const uint32_t rect_x0 = get_arg_val<uint32_t>(8);
    const uint32_t rect_y0 = get_arg_val<uint32_t>(9);
    const uint32_t rect_x1 = get_arg_val<uint32_t>(10);
    const uint32_t rect_y1 = get_arg_val<uint32_t>(11);

    const uint32_t y_page_bytes = get_tile_size(cb_y_tiles);
    const auto y = TensorAccessor(y_args, y_addr, y_page_bytes);
    const auto src = TensorAccessor(src_args, src_addr, y_page_bytes);
    Noc noc;
    CircularBuffer cb_y(cb_y_tiles);
    const uint32_t y_tiles = block_m_tiles * core_n_tiles;

    cb_y.reserve_back(y_tiles);
    uint32_t offset = 0;
    for (uint32_t m = 0; m < block_m_tiles; ++m) {
        for (uint32_t n = 0; n < core_n_tiles; ++n) {
            noc.async_read(
                src,
                cb_y,
                y_page_bytes,
                {.page_id = m * tensor_n_tiles + n_start + n * n_stride},
                {.offset_bytes = offset});
            offset += y_page_bytes;
        }
    }
    noc.async_read_barrier();

    const uint32_t gather_addr = get_semaphore(gather_sem_id);
    auto* release = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(release_sem_id));
    if (is_master) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gather_addr), num_active - 1);
        noc_semaphore_set(release, 1);
        noc_semaphore_set_multicast_loopback_src(
            get_semaphore(release_sem_id),
            get_noc_multicast_addr(rect_x0, rect_y0, rect_x1, rect_y1, get_semaphore(release_sem_id)),
            num_rect);
        noc_async_write_barrier();
    } else {
        noc_semaphore_inc(get_noc_addr(master_x, master_y, gather_addr), 1);
        noc_async_atomic_barrier();
        noc_semaphore_wait(release, 1);
    }
    cb_y.push_back(y_tiles);

    const uint32_t reader_tiles = y_reader_tiles(y_tiles, y_reader_share_tenths);
    if (reader_tiles == 0) {
        return;
    }
    cb_y.wait_front(y_tiles);
    MaybeDeviceZoneScope("reader_y_store");
    store_y_share(
        noc,
        cb_y,
        y,
        y_page_bytes,
        block_m_tiles,
        core_n_tiles,
        tensor_n_tiles,
        n_start,
        n_stride,
        y_split_order,
        reader_tiles,
        true);
    noc.async_write_barrier();
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(y_done_sem_id)), 1);
}
