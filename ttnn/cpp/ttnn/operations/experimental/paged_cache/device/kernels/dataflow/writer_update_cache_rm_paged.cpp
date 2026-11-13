// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// Row-major interleaved paged cache: every cache row is one page, so each input row is written
// straight to page ``page_table[pos / block_size] * block_size + pos % block_size``.
void kernel_main() {
    Noc noc;

    const auto my_batch_idx = get_arg(args::my_batch_idx);

    constexpr auto row_bytes = get_arg(args::row_bytes);
    constexpr auto input_row_stride = get_arg(args::input_row_stride);
    constexpr auto block_size = get_arg(args::block_size);
    constexpr auto max_blocks_per_seq = get_arg(args::max_blocks_per_seq);
    constexpr auto index_stick_size_B = get_arg(args::index_stick_size_B);
    constexpr auto page_table_stick_size = get_arg(args::page_table_stick_size);
    constexpr auto num_rows = get_arg(args::num_rows);
    constexpr uint32_t max_position = max_blocks_per_seq * block_size;

    DataflowBuffer dfb_input(dfb::input);
    DataflowBuffer dfb_index(dfb::index);
    DataflowBuffer dfb_page_table(dfb::page_table);

    dfb_input.reserve_back(1);
    dfb_input.push_back(1);
    const uint32_t src_base = dfb_input.get_read_ptr();

    const auto index_reader = TensorAccessor(tensor::index);
    dfb_index.reserve_back(1);
    const uint32_t index_l1 = dfb_index.get_write_ptr();
    noc.async_read(index_reader, CoreLocalMem<uint32_t>(index_l1), index_stick_size_B, {.page_id = 0}, {});

    const auto page_table_reader = TensorAccessor(tensor::page_table);
    dfb_page_table.reserve_back(1);
    const uint32_t page_table_l1 = dfb_page_table.get_write_ptr();
    uint32_t page_table_row = my_batch_idx;
#ifdef PAGE_TABLE_ROW_TENSOR
    {
        // The row lands in the page-table buffer, which the row read below then overwrites.
        constexpr auto page_table_row_page_bytes = get_arg(args::page_table_row_page_bytes);
        const auto row_reader = TensorAccessor(tensor::page_table_row);
        noc.async_read(
            row_reader, CoreLocalMem<uint32_t>(page_table_l1), page_table_row_page_bytes, {.page_id = 0}, {});
        noc.async_read_barrier();
        page_table_row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_table_l1)[0];
    }
#endif
    noc.async_read(
        page_table_reader,
        CoreLocalMem<uint32_t>(page_table_l1),
        page_table_stick_size,
        {.page_id = page_table_row},
        {});
    noc.async_read_barrier();
    dfb_index.push_back(1);
    dfb_page_table.push_back(1);

    volatile tt_l1_ptr uint32_t* positions =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_l1) + my_batch_idx * num_rows;
    volatile tt_l1_ptr uint32_t* page_table = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_table_l1);

    const auto cache = TensorAccessor(tensor::cache);
    for (uint32_t s = 0; s < num_rows; ++s) {
        const uint32_t pos = positions[s];
        // -1 (skip) is also out of range.
        if (pos >= max_position) {
            continue;
        }
        const uint32_t page = page_table[pos / block_size] * block_size + pos % block_size;
        noc.async_write(
            CoreLocalMem<uint32_t>(src_base + s * input_row_stride), cache, row_bytes, {}, {.page_id = page});
    }
    noc.async_write_barrier();

    dfb_page_table.pop_front(1);
    dfb_index.pop_front(1);
    dfb_input.pop_front(1);
}
