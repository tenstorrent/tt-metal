// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Isolated y-store bench, BRISC/NoC1 (the op's writer): waits for the y block and stores its share.

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
    constexpr auto y_args = TensorAccessorArgs<6>();

    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_start = get_arg_val<uint32_t>(1);
    const uint32_t core_n_tiles = get_arg_val<uint32_t>(2);
    const uint32_t n_stride = get_arg_val<uint32_t>(3);
    if (core_n_tiles == 0) {
        return;
    }

    const uint32_t y_page_bytes = get_tile_size(cb_y_tiles);
    const auto y = TensorAccessor(y_args, y_addr, y_page_bytes);
    Noc noc;
    CircularBuffer cb_y(cb_y_tiles);
    const uint32_t y_tiles = block_m_tiles * core_n_tiles;
    const uint32_t reader_tiles = y_reader_tiles(y_tiles, y_reader_share_tenths);
    {
        MaybeDeviceZoneScope("writer_y_wait");
        cb_y.wait_front(y_tiles);
    }
    MaybeDeviceZoneScope("writer_y_store");
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
        false);
    noc.async_write_barrier();
    if (reader_tiles > 0) {
        auto* y_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(y_done_sem_id));
        noc_semaphore_wait(y_done, 1);
        noc_semaphore_set(y_done, 0);
    }
    cb_y.pop_front(y_tiles);
}
