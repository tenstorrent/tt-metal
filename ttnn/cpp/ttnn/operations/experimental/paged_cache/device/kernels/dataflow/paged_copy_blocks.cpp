// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Copies or zeroes whole blocks of paged caches, page by page. Row i of the [num_pairs, 2] table is (src, dst):
// src == 0xFFFFFFFF zeroes block dst, src == dst skips the row, otherwise block dst becomes a copy of block src.
// This core handles items [first_item, first_item + num_items) of the (cache, row, page) item space.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_page = 0, cb_table = 1, cb_zero = 2;
constexpr uint32_t kZeroBlock = 0xFFFFFFFF;

void kernel_main() {
    constexpr uint32_t pages_per_block = get_compile_time_arg_val(0);
    constexpr uint32_t num_pairs = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t aligned_page_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(4);
    constexpr auto cache_a = TensorAccessorArgs<5>();
    constexpr auto table_a = TensorAccessorArgs<cache_a.next_compile_time_args_offset()>();

    const uint32_t first_item = get_arg_val<uint32_t>(0);
    const uint32_t num_items = get_arg_val<uint32_t>(1);
    const uint32_t table_addr = get_arg_val<uint32_t>(2);
    // rt 3 + c = buffer address of cache c

    Noc noc;
    CircularBuffer page(cb_page), table(cb_table), zero(cb_zero);

    const auto table_acc = TensorAccessor(table_a, table_addr, row_bytes);
    for (uint32_t i = 0; i < num_pairs; i++) {
        noc.async_read(table_acc, table, row_bytes, {.page_id = i}, {.offset_bytes = i * row_bytes});
    }
    const CoreLocalMem<volatile uint32_t> zeros(zero.get_write_ptr());
    for (uint32_t w = 0; w < aligned_page_bytes / sizeof(uint32_t); w++) {
        zeros[w] = 0;
    }
    noc.async_read_barrier();

    const CoreLocalMem<volatile uint32_t> rows(table.get_write_ptr());
    constexpr uint32_t row_words = row_bytes / sizeof(uint32_t);
    auto page_l1 = use<CircularBuffer::AddrSelector::WRITE_PTR>(page);
    auto zero_l1 = use<CircularBuffer::AddrSelector::WRITE_PTR>(zero);

    for (uint32_t item = first_item; item < first_item + num_items; item++) {
        const uint32_t cache = item / (num_pairs * pages_per_block);
        const uint32_t row = item / pages_per_block % num_pairs;
        const uint32_t p = item % pages_per_block;
        const uint32_t src = rows[row * row_words];
        const uint32_t dst = rows[row * row_words + 1];
        if (src == dst) {
            continue;
        }
        const auto cache_acc = TensorAccessor(cache_a, get_arg_val<uint32_t>(3 + cache), aligned_page_bytes);
        if (src == kZeroBlock) {
            noc.async_write(
                zero_l1, cache_acc, page_bytes, {.offset_bytes = 0}, {.page_id = dst * pages_per_block + p});
        } else {
            noc.async_read(cache_acc, page, page_bytes, {.page_id = src * pages_per_block + p}, {.offset_bytes = 0});
            noc.async_read_barrier();
            noc.async_write(
                page_l1, cache_acc, page_bytes, {.offset_bytes = 0}, {.page_id = dst * pages_per_block + p});
        }
        noc.async_write_barrier();
    }
}
