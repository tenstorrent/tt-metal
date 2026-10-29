// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// expert rows, the placement program's place.cpp (data movement only). Reads the routing table the expert rows program
// wrote and writes moe_compute's outputs:
//   output 0  per-expert token counts, one row in every grid core's shard (same L1 address on every core)
//   output 1  one activation row per token with >= 1 local expert, ascending: [token, k per expert (top_k + 1 = not
//             routed; the ring writes top_k or top_k + 1 there, by tilize RISC), score bits per expert], then rows
//             [-1, top_k + 1 .., 0 ..]
//   output 2  page e: expert e's tokens ascending, one e_t_entry_words entry each [token, k, 0 ..], then end_of_list
//   output 3  the double buffer: the last two local experts' rows, half e % 2, with moe_compute's geometry: expert
//             e's rows split evenly over the height shards (earlier shards take the remainder), row i < half_rows of
//             height shard h at (e % 2) * half_rows * segment_bytes + i * segment_bytes, bytes
//             [w, w + 1) * segment_bytes of the row for width shard w.
// Runs on the height_shards x width_shards combine cores, core i = shard (i / width_shards, i % width_shards); each
// core places its own output-3 shard, and the metadata work is split over all of them.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "expert_rows_common.h"

namespace {

constexpr uint32_t tokens = get_named_compile_time_arg_val("tokens");
constexpr uint32_t top_k = get_named_compile_time_arg_val("top_k");
constexpr uint32_t local_experts = get_named_compile_time_arg_val("local_experts");
constexpr uint32_t height_shards = get_named_compile_time_arg_val("height_shards");
constexpr uint32_t segment_bytes = get_named_compile_time_arg_val("segment_bytes");
constexpr uint32_t half_rows = get_named_compile_time_arg_val("half_rows");
constexpr uint32_t count_words = get_named_compile_time_arg_val("count_words");
constexpr uint32_t e_t_page_bytes = get_named_compile_time_arg_val("e_t_page_bytes");
constexpr uint32_t activation_row_bytes = get_named_compile_time_arg_val("activation_row_bytes");
constexpr uint32_t grid_w = get_named_compile_time_arg_val("grid_w");
constexpr uint32_t grid_h = get_named_compile_time_arg_val("grid_h");
constexpr uint32_t program_cores = get_named_compile_time_arg_val("program_cores");
constexpr uint32_t row_bytes = get_named_compile_time_arg_val("row_bytes");
// the routing table block rows_reader.cpp wrote
constexpr expert_rows::TableLayout table_block = expert_rows::table_layout(tokens, top_k, local_experts);
// byte offsets in the scratch CB
constexpr uint32_t scratch_e_t = get_named_compile_time_arg_val("scratch_e_t");
constexpr uint32_t scratch_activation = get_named_compile_time_arg_val("scratch_activation");
constexpr uint32_t scratch_count_row = get_named_compile_time_arg_val("scratch_count_row");
constexpr uint32_t scratch_table = get_named_compile_time_arg_val("scratch_table");
constexpr uint32_t not_routed_k = top_k + 1;
constexpr uint32_t word_bytes = sizeof(uint32_t);
constexpr uint32_t e_t_entry_words = get_named_compile_time_arg_val("e_t_entry_words");
constexpr uint32_t end_of_list = 0xFFFFFFFFu;

// The routing table into scratch: its head, row counts and first rows (everything before the row tokens), then (with
// routed rows) the rows' tokens and k and the token-ordered entries. Returns the routed rows.
template <typename TableAccessor>
FORCE_INLINE uint32_t read_routing_table(const Noc& noc, const TableAccessor& table, uint32_t scratch) {
    const uint64_t table_noc_addr = table.get_noc_addr(0);
    noc.async_read(
        PrecomposedUnicastEndpoint{},
        CoreLocalMem<uint32_t>(scratch + scratch_table),
        table_block.row_tokens,
        {.noc_addr = table_noc_addr},
        {});
    noc.async_read_barrier();
    const uint32_t routed_rows = *reinterpret_cast<const tt_l1_ptr uint32_t*>(scratch + scratch_table);
    // every part of routed_rows u16 starts at a multiple of part_alignment
    const uint32_t part_bytes =
        expert_rows::align_up(routed_rows * expert_rows::u16_bytes, expert_rows::part_alignment);
    if (routed_rows) {
        const uint32_t parts[] = {
            table_block.row_tokens,
            table_block.row_k,
            table_block.entry_experts,
            table_block.entry_tokens,
            table_block.entry_scores,
            table_block.entry_k};
        for (uint32_t part = 0; part < sizeof(parts) / sizeof(parts[0]); ++part) {
            noc.async_read(
                PrecomposedUnicastEndpoint{},
                CoreLocalMem<uint32_t>(scratch + scratch_table + parts[part]),
                part_bytes,
                {.noc_addr = table_noc_addr + parts[part]},
                {});
        }
    }
    noc.async_read_barrier();
    return routed_rows;
}

// Output 0: the counts row to grid cores core_index, core_index + program_cores, ... of the row-major grid.
FORCE_INLINE void write_count_rows(
    const Noc& noc,
    uint32_t scratch,
    const tt_l1_ptr uint16_t* expert_row_counts,
    uint32_t core_index,
    uint32_t counts_addr,
    uint32_t virtual_x_arg,
    uint32_t virtual_y_arg) {
    volatile tt_l1_ptr uint32_t* count_row =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + scratch_count_row);
    for (uint32_t word = 0; word < count_words; ++word) {
        count_row[word] = word < local_experts ? expert_row_counts[word] : 0;
    }
    for (uint32_t core = core_index; core < grid_w * grid_h; core += program_cores) {
        const uint32_t x = get_arg_val<uint32_t>(virtual_x_arg + core % grid_w);
        const uint32_t y = get_arg_val<uint32_t>(virtual_y_arg + core / grid_w);
        noc.async_write(
            CoreLocalMem<uint32_t>(scratch + scratch_count_row),
            UnicastEndpoint{},
            count_words * word_bytes,
            {},
            {.noc_x = x, .noc_y = y, .addr = counts_addr});
    }
}

// Output 2: the page of each expert with expert % program_cores == core_index.
template <typename ETAccessor>
FORCE_INLINE void write_e_t_pages(
    const Noc& noc,
    const ETAccessor& e_t,
    uint32_t scratch,
    const tt_l1_ptr uint16_t* expert_row_counts,
    const tt_l1_ptr uint16_t* expert_first_rows,
    const tt_l1_ptr uint16_t* row_tokens,
    const tt_l1_ptr uint16_t* row_k,
    uint32_t core_index) {
    volatile tt_l1_ptr uint32_t* page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + scratch_e_t);
    for (uint32_t expert = core_index; expert < local_experts; expert += program_cores) {
        noc.async_write_barrier();  // the page buffer is reused per page
        const uint32_t row_count = expert_row_counts[expert];
        for (uint32_t entry = 0; entry <= row_count; ++entry) {
            const uint32_t row = expert_first_rows[expert] + entry;
            page[e_t_entry_words * entry] = entry < row_count ? uint32_t(row_tokens[row]) : end_of_list;
            page[e_t_entry_words * entry + 1] = entry < row_count ? uint32_t(row_k[row]) : 0;
            for (uint32_t word = 2; word < e_t_entry_words; ++word) {
                page[e_t_entry_words * entry + word] = 0;
            }
        }
        noc.async_write(
            CoreLocalMem<uint32_t>(scratch + scratch_e_t),
            e_t,
            (row_count + 1) * e_t_entry_words * word_bytes,
            {},
            {.page_id = expert});
    }
}

FORCE_INLINE volatile tt_l1_ptr uint32_t* activation_buffer(uint32_t scratch, uint32_t buffer) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + scratch_activation + buffer * activation_row_bytes);
}

// [token, not_routed_k per expert, 0 per expert, 0 alignment padding]
FORCE_INLINE void clear_activation_row(volatile tt_l1_ptr uint32_t* row, uint32_t token) {
    row[0] = token;
    for (uint32_t expert = 0; expert < local_experts; ++expert) {
        row[1 + expert] = not_routed_k;
        row[1 + local_experts + expert] = 0;
    }
    for (uint32_t word = 1 + 2 * local_experts; word < activation_row_bytes / word_bytes; ++word) {
        row[word] = 0;
    }
}

// Output 1, activated tokens ascending: row r is written by core r % program_cores. Returns the activated tokens.
template <typename ActivationAccessor>
FORCE_INLINE uint32_t write_activated_rows(
    const Noc& noc, const ActivationAccessor& activation, uint32_t scratch, uint32_t routed_rows, uint32_t core_index) {
    const tt_l1_ptr uint16_t* entry_experts =
        reinterpret_cast<const tt_l1_ptr uint16_t*>(scratch + scratch_table + table_block.entry_experts);
    const tt_l1_ptr uint16_t* entry_tokens =
        reinterpret_cast<const tt_l1_ptr uint16_t*>(scratch + scratch_table + table_block.entry_tokens);
    const tt_l1_ptr uint16_t* entry_scores =
        reinterpret_cast<const tt_l1_ptr uint16_t*>(scratch + scratch_table + table_block.entry_scores);
    const tt_l1_ptr uint16_t* entry_k =
        reinterpret_cast<const tt_l1_ptr uint16_t*>(scratch + scratch_table + table_block.entry_k);
    uint32_t row_index = 0, buffer = 0;
    uint32_t first_entry = 0;
    while (first_entry < routed_rows) {
        const uint32_t token = entry_tokens[first_entry];
        uint32_t end_entry = first_entry;
        while (end_entry < routed_rows && entry_tokens[end_entry] == token) {
            ++end_entry;
        }
        if (row_index % program_cores == core_index) {
            if (buffer == 0) {
                noc.async_write_barrier();  // every activation buffer is about to be refilled
            }
            volatile tt_l1_ptr uint32_t* row = activation_buffer(scratch, buffer);
            clear_activation_row(row, token);
            for (uint32_t entry = first_entry; entry < end_entry; ++entry) {
                row[1 + entry_experts[entry]] = entry_k[entry];
                row[1 + local_experts + entry_experts[entry]] = uint32_t(entry_scores[entry]);
            }
            noc.async_write(
                CoreLocalMem<uint32_t>(scratch + scratch_activation + buffer * activation_row_bytes),
                activation,
                activation_row_bytes,
                {},
                {.page_id = 0, .offset_bytes = row_index * activation_row_bytes});
            buffer = (buffer + 1) % expert_rows::activation_buffers;
        }
        ++row_index;
        first_entry = end_entry;
    }
    return row_index;
}

// Output 1, the empty rows from `first_row` on, of this core's share.
template <typename ActivationAccessor>
FORCE_INLINE void write_empty_activation_rows(
    const Noc& noc, const ActivationAccessor& activation, uint32_t scratch, uint32_t first_row, uint32_t core_index) {
    noc.async_write_barrier();
    clear_activation_row(activation_buffer(scratch, 0), end_of_list);
    for (uint32_t row = first_row; row < tokens; ++row) {
        if (row % program_cores == core_index) {
            noc.async_write(
                CoreLocalMem<uint32_t>(scratch + scratch_activation),
                activation,
                activation_row_bytes,
                {},
                {.page_id = 0, .offset_bytes = row * activation_row_bytes});
        }
    }
}

// Output 3: the last two experts' rows of this core's shard (height_shard, width_shard), at most half_rows each.
template <typename RowsAccessor>
FORCE_INLINE void place_double_buffer_shard(
    const Noc& noc,
    const RowsAccessor& row_buffer,
    const tt_l1_ptr uint16_t* expert_row_counts,
    const tt_l1_ptr uint16_t* expert_first_rows,
    uint32_t double_buffer,
    uint32_t height_shard,
    uint32_t width_shard) {
    const uint32_t first_expert = local_experts >= 2 ? local_experts - 2 : 0;
    for (uint32_t expert = first_expert; expert < local_experts; ++expert) {
        const uint32_t row_count = expert_row_counts[expert];
        const uint32_t base_rows = row_count / height_shards, remainder = row_count % height_shards;
        const uint32_t rows = base_rows + (height_shard < remainder ? 1 : 0);
        ASSERT(rows <= half_rows);  // moe_compute's token cap: tokens <= height_shards * half_rows
        const uint32_t shard_first = height_shard * base_rows + (height_shard < remainder ? height_shard : remainder);
        for (uint32_t shard_row = 0; shard_row < rows; ++shard_row) {
            noc.async_read(
                row_buffer,
                CoreLocalMem<uint32_t>(double_buffer + ((expert % 2) * half_rows + shard_row) * segment_bytes),
                segment_bytes,
                {.page_id = expert_first_rows[expert] + shard_first + shard_row,
                 .offset_bytes = width_shard * segment_bytes},
                {});
        }
    }
}

}  // namespace

void kernel_main() {
    CircularBuffer cb_scratch(get_named_compile_time_arg_val("cb_scratch"));
    constexpr auto rows_args = TensorAccessorArgs<0>();
    constexpr auto e_t_args = TensorAccessorArgs<rows_args.next_compile_time_args_offset()>();
    constexpr auto activation_args = TensorAccessorArgs<e_t_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<activation_args.next_compile_time_args_offset()>();

    uint32_t arg = 0;
    const uint32_t rows_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t e_t_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t activation_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t counts_addr = get_arg_val<uint32_t>(arg++);    // output 0: L1 address of every grid core's shard
    const uint32_t double_buffer = get_arg_val<uint32_t>(arg++);  // output 3: this core's shard
    const uint32_t table_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t height_shard = get_arg_val<uint32_t>(arg++);
    const uint32_t width_shard = get_arg_val<uint32_t>(arg++);
    const uint32_t core_index = get_arg_val<uint32_t>(arg++);
    const uint32_t virtual_x_arg = arg, virtual_y_arg = arg + grid_w;

    const Noc noc;
    cb_scratch.reserve_back(1);
    const uint32_t scratch = cb_scratch.get_write_ptr();
    const tt_l1_ptr uint16_t* expert_row_counts =
        reinterpret_cast<const tt_l1_ptr uint16_t*>(scratch + scratch_table + table_block.row_counts);
    const tt_l1_ptr uint16_t* expert_first_rows =
        reinterpret_cast<const tt_l1_ptr uint16_t*>(scratch + scratch_table + table_block.first_rows);
    const tt_l1_ptr uint16_t* row_tokens =
        reinterpret_cast<const tt_l1_ptr uint16_t*>(scratch + scratch_table + table_block.row_tokens);
    const tt_l1_ptr uint16_t* row_k =
        reinterpret_cast<const tt_l1_ptr uint16_t*>(scratch + scratch_table + table_block.row_k);

    const auto table = TensorAccessor(table_args, table_addr, table_block.bytes);
    const uint32_t routed_rows = read_routing_table(noc, table, scratch);

    write_count_rows(noc, scratch, expert_row_counts, core_index, counts_addr, virtual_x_arg, virtual_y_arg);
    const auto e_t = TensorAccessor(e_t_args, e_t_addr, e_t_page_bytes);
    write_e_t_pages(noc, e_t, scratch, expert_row_counts, expert_first_rows, row_tokens, row_k, core_index);
    const auto activation = TensorAccessor(activation_args, activation_addr, tokens * activation_row_bytes);
    const uint32_t activated = write_activated_rows(noc, activation, scratch, routed_rows, core_index);
    write_empty_activation_rows(noc, activation, scratch, activated, core_index);

    const auto row_buffer = TensorAccessor(rows_args, rows_addr, row_bytes);
    place_double_buffer_shard(
        noc, row_buffer, expert_row_counts, expert_first_rows, double_buffer, height_shard, width_shard);
    noc.async_read_barrier();
    noc.async_write_barrier();
}
