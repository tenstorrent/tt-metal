// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// expert rows metadata and placement (data movement only). Reads the routing table the expert-row program wrote and
// writes moe_compute's outputs:
//   slot 0  per-expert token counts, one row in every grid core's shard (same L1 address on every core)
//   slot 1  one activation row per token with >= 1 local expert, ascending: [token, k per expert (k + 1 = not
//           routed), score bits per expert], then rows [-1, k + 1 .., 0 ..]
//   slot 2  page e: expert e's tokens ascending, one u32 per 16 B entry, then -1
//   slot 3  the last two local experts' rows in the double buffer, half e % 2, with moe_compute's geometry: expert e's
//           rows split evenly over the height shards (earlier shards take the remainder), row i of height shard h at
//           (e % 2) * half_bytes + i * segment_bytes, bytes [w, w + 1) * segment_bytes of the row for width shard w.
// The metadata work is split over the program_cores cores of the program. ComputeOnly runs on the combine cores
// (height_shards x width_shards, core i = shard (i / width_shards, i % width_shards)) and each core places its own
// slot-3 shard. With the combine (place_dense = 0) it runs on the feeder cores, feed.cpp fills slot 3, and once every
// core's writes landed core 0 increments the combine sync core's metadata semaphore once (the combine reader waits for
// exactly 1).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

namespace {

constexpr uint32_t tokens = get_named_compile_time_arg_val("tokens");
constexpr uint32_t top_k = get_named_compile_time_arg_val("top_k");
constexpr uint32_t local_experts = get_named_compile_time_arg_val("local_experts");
constexpr uint32_t height_shards = get_named_compile_time_arg_val("height_shards");
constexpr uint32_t segment_bytes = get_named_compile_time_arg_val("segment_bytes");
constexpr uint32_t half_bytes = get_named_compile_time_arg_val("half_bytes");
constexpr uint32_t count_words = get_named_compile_time_arg_val("count_words");
constexpr uint32_t e_t_page_bytes = get_named_compile_time_arg_val("e_t_page_bytes");
constexpr uint32_t activation_row_bytes = get_named_compile_time_arg_val("activation_row_bytes");
constexpr uint32_t grid_w = get_named_compile_time_arg_val("grid_w");
constexpr uint32_t grid_h = get_named_compile_time_arg_val("grid_h");
constexpr uint32_t program_cores = get_named_compile_time_arg_val("program_cores");
constexpr uint32_t row_bytes = get_named_compile_time_arg_val("row_bytes");
constexpr bool place_dense = get_named_compile_time_arg_val("place_dense") == 1;
constexpr uint32_t done_semaphore_id = get_named_compile_time_arg_val("done_semaphore_id");
constexpr uint32_t metadata_semaphore_id = get_named_compile_time_arg_val("metadata_semaphore_id");
// routing table block (rows_reader.cpp): u32 routed rows; u16 counts / first rows [local_experts]; u16 rows
// [tokens top_k] in expert-row order; the same entries in token order: slot, token, score, k
constexpr uint32_t table_counts = get_named_compile_time_arg_val("table_counts");
constexpr uint32_t table_offsets = get_named_compile_time_arg_val("table_offsets");
constexpr uint32_t table_rows = get_named_compile_time_arg_val("table_rows");
constexpr uint32_t table_entry_experts = get_named_compile_time_arg_val("table_entry_experts");
constexpr uint32_t table_entry_tokens = get_named_compile_time_arg_val("table_entry_tokens");
constexpr uint32_t table_entry_scores = get_named_compile_time_arg_val("table_entry_scores");
constexpr uint32_t table_entry_k = get_named_compile_time_arg_val("table_entry_k");
constexpr uint32_t table_bytes = get_named_compile_time_arg_val("table_bytes");
// byte offsets in the scratch CB
constexpr uint32_t scratch_counts = get_named_compile_time_arg_val("scratch_counts");
constexpr uint32_t scratch_offsets = get_named_compile_time_arg_val("scratch_offsets");
constexpr uint32_t scratch_e_t = get_named_compile_time_arg_val("scratch_e_t");
constexpr uint32_t scratch_activation = get_named_compile_time_arg_val("scratch_activation");
constexpr uint32_t scratch_count_row = get_named_compile_time_arg_val("scratch_count_row");
constexpr uint32_t scratch_table = get_named_compile_time_arg_val("scratch_table");
constexpr uint32_t activation_buffers = 8;  // activation rows in flight before a write barrier
constexpr uint32_t not_routed_k = top_k + 1;

// The routing table into scratch: its head, then (with routed rows) the rows and the token-ordered entries; the
// per-expert counts and first rows widened to u32 meanwhile. Returns the routed rows.
template <typename TableAccessor>
FORCE_INLINE uint32_t read_routing_table(
    const Noc& noc,
    const TableAccessor& table,
    uint32_t scratch,
    tt_l1_ptr uint32_t* expert_row_counts,
    tt_l1_ptr uint32_t* expert_first_rows) {
    const uint64_t table_noc_addr = table.get_noc_addr(0);
    noc.async_read(
        PrecomposedUnicastEndpoint{},
        CoreLocalMem<uint32_t>(scratch + scratch_table),
        table_rows,
        {.noc_addr = table_noc_addr},
        {});
    noc.async_read_barrier();
    const uint32_t routed_rows = *reinterpret_cast<const tt_l1_ptr uint32_t*>(scratch + scratch_table);
    const uint32_t part_bytes = (routed_rows * 2 + 63) & ~63u;
    if (routed_rows) {
        const uint32_t parts[5] = {
            table_rows, table_entry_experts, table_entry_tokens, table_entry_scores, table_entry_k};
        for (uint32_t i = 0; i < 5; ++i) {
            noc.async_read(
                PrecomposedUnicastEndpoint{},
                CoreLocalMem<uint32_t>(scratch + scratch_table + parts[i]),
                part_bytes,
                {.noc_addr = table_noc_addr + parts[i]},
                {});
        }
    }
    const tt_l1_ptr uint16_t* table_expert_rows =
        reinterpret_cast<const tt_l1_ptr uint16_t*>(scratch + scratch_table + table_counts);
    const tt_l1_ptr uint16_t* table_expert_first_rows =
        reinterpret_cast<const tt_l1_ptr uint16_t*>(scratch + scratch_table + table_offsets);
    for (uint32_t e = 0; e < local_experts; ++e) {
        expert_row_counts[e] = table_expert_rows[e];
        expert_first_rows[e] = table_expert_first_rows[e];
    }
    noc.async_read_barrier();
    return routed_rows;
}

// Slot 0: the counts row to grid cores core_index, core_index + program_cores, ... of the row-major grid.
FORCE_INLINE void write_count_rows(
    const Noc& noc,
    uint32_t scratch,
    const tt_l1_ptr uint32_t* expert_row_counts,
    uint32_t core_index,
    uint32_t counts_addr,
    uint32_t virtual_x_arg,
    uint32_t virtual_y_arg) {
    volatile tt_l1_ptr uint32_t* count_row =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + scratch_count_row);
    for (uint32_t i = 0; i < count_words; ++i) {
        count_row[i] = i < local_experts ? expert_row_counts[i] : 0;
    }
    for (uint32_t core = core_index; core < grid_w * grid_h; core += program_cores) {
        const uint32_t x = get_arg_val<uint32_t>(virtual_x_arg + core % grid_w);
        const uint32_t y = get_arg_val<uint32_t>(virtual_y_arg + core / grid_w);
        noc.async_write(
            CoreLocalMem<uint32_t>(scratch + scratch_count_row),
            UnicastEndpoint{},
            count_words * 4,
            {},
            {.noc_x = x, .noc_y = y, .addr = counts_addr});
    }
}

// Slot 2: page e for the experts e with e % program_cores == core_index.
template <typename ETAccessor>
FORCE_INLINE void write_e_t_pages(
    const Noc& noc,
    const ETAccessor& e_t,
    uint32_t scratch,
    const tt_l1_ptr uint32_t* expert_row_counts,
    const tt_l1_ptr uint32_t* expert_first_rows,
    const tt_l1_ptr uint16_t* row_tokens,
    uint32_t core_index) {
    volatile tt_l1_ptr uint32_t* page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + scratch_e_t);
    for (uint32_t e = core_index; e < local_experts; e += program_cores) {
        noc.async_write_barrier();  // the page buffer is reused per page
        const uint32_t n = expert_row_counts[e];
        for (uint32_t i = 0; i <= n; ++i) {
            page[4 * i] = i < n ? uint32_t(row_tokens[expert_first_rows[e] + i]) : 0xFFFFFFFFu;
            page[4 * i + 1] = 0;
            page[4 * i + 2] = 0;
            page[4 * i + 3] = 0;
        }
        noc.async_write(CoreLocalMem<uint32_t>(scratch + scratch_e_t), e_t, (n + 1) * 16, {}, {.page_id = e});
    }
}

FORCE_INLINE volatile tt_l1_ptr uint32_t* activation_buffer(uint32_t scratch, uint32_t buffer) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + scratch_activation + buffer * activation_row_bytes);
}

// [token, not_routed_k per expert, 0 per expert, 0 alignment padding]
FORCE_INLINE void clear_activation_row(volatile tt_l1_ptr uint32_t* row, uint32_t token) {
    row[0] = token;
    for (uint32_t e = 0; e < local_experts; ++e) {
        row[1 + e] = not_routed_k;
        row[1 + local_experts + e] = 0;
    }
    for (uint32_t i = 1 + 2 * local_experts; i < activation_row_bytes / 4; ++i) {
        row[i] = 0;
    }
}

// Slot 1, activated tokens ascending: row r is written by core r % program_cores. Returns the activated tokens.
template <typename ActivationAccessor>
FORCE_INLINE uint32_t write_activated_rows(
    const Noc& noc, const ActivationAccessor& activation, uint32_t scratch, uint32_t routed_rows, uint32_t core_index) {
    const tt_l1_ptr uint16_t* entry_slots =
        reinterpret_cast<const tt_l1_ptr uint16_t*>(scratch + scratch_table + table_entry_experts);
    const tt_l1_ptr uint16_t* entry_tokens =
        reinterpret_cast<const tt_l1_ptr uint16_t*>(scratch + scratch_table + table_entry_tokens);
    const tt_l1_ptr uint16_t* entry_scores =
        reinterpret_cast<const tt_l1_ptr uint16_t*>(scratch + scratch_table + table_entry_scores);
    const tt_l1_ptr uint16_t* entry_k =
        reinterpret_cast<const tt_l1_ptr uint16_t*>(scratch + scratch_table + table_entry_k);
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
            for (uint32_t q = first_entry; q < end_entry; ++q) {
                row[1 + entry_slots[q]] = entry_k[q];
                row[1 + local_experts + entry_slots[q]] = uint32_t(entry_scores[q]);
            }
            noc.async_write(
                CoreLocalMem<uint32_t>(scratch + scratch_activation + buffer * activation_row_bytes),
                activation,
                activation_row_bytes,
                {},
                {.page_id = 0, .offset_bytes = row_index * activation_row_bytes});
            buffer = (buffer + 1) % activation_buffers;
        }
        ++row_index;
        first_entry = end_entry;
    }
    return row_index;
}

// Slot 1, the empty rows from `first_row` on, of this core's share.
template <typename ActivationAccessor>
FORCE_INLINE void write_empty_activation_rows(
    const Noc& noc, const ActivationAccessor& activation, uint32_t scratch, uint32_t first_row, uint32_t core_index) {
    noc.async_write_barrier();
    clear_activation_row(activation_buffer(scratch, 0), 0xFFFFFFFFu);
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

// Slot 3: the last two experts' rows of this core's shard (height_shard, width_shard).
template <typename RowsAccessor>
FORCE_INLINE void place_double_buffer_shard(
    const Noc& noc,
    const RowsAccessor& row_buffer,
    const tt_l1_ptr uint32_t* expert_row_counts,
    const tt_l1_ptr uint32_t* expert_first_rows,
    uint32_t double_buffer,
    uint32_t height_shard,
    uint32_t width_shard) {
    const uint32_t first_expert = local_experts >= 2 ? local_experts - 2 : 0;
    for (uint32_t e = first_expert; e < local_experts; ++e) {
        const uint32_t n = expert_row_counts[e];
        const uint32_t base_rows = n / height_shards, remainder = n % height_shards;
        const uint32_t shard_rows = base_rows + (height_shard < remainder ? 1 : 0);
        const uint32_t shard_first = height_shard * base_rows + (height_shard < remainder ? height_shard : remainder);
        for (uint32_t i = 0; i < shard_rows; ++i) {
            noc.async_read(
                row_buffer,
                CoreLocalMem<uint32_t>(double_buffer + (e % 2) * half_bytes + i * segment_bytes),
                segment_bytes,
                {.page_id = expert_first_rows[e] + shard_first + i, .offset_bytes = width_shard * segment_bytes},
                {});
        }
    }
}

// With the combine: every core's writes landed, then core 0 increments the combine sync core's metadata semaphore.
FORCE_INLINE void signal_metadata_ready(
    const Noc& noc, uint32_t core_index, uint32_t lead_x, uint32_t lead_y, uint32_t sync_x, uint32_t sync_y) {
    Semaphore<> done(done_semaphore_id);
    done.up(noc, lead_x, lead_y, 1);
    if (core_index == 0) {
        done.wait(program_cores);
        done.set(0);
        Semaphore<>(metadata_semaphore_id).up(noc, sync_x, sync_y, 1);
    }
    noc.async_atomic_barrier();
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
    const uint32_t counts_addr = get_arg_val<uint32_t>(arg++);    // slot 0: L1 address of every grid core's shard
    const uint32_t double_buffer = get_arg_val<uint32_t>(arg++);  // slot 3: this core's shard
    const uint32_t table_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t height_shard = get_arg_val<uint32_t>(arg++);
    const uint32_t width_shard = get_arg_val<uint32_t>(arg++);
    const uint32_t core_index = get_arg_val<uint32_t>(arg++);
    const uint32_t lead_x = get_arg_val<uint32_t>(arg++);  // core 0 of the program and the combine sync core
    const uint32_t lead_y = get_arg_val<uint32_t>(arg++);
    const uint32_t sync_x = get_arg_val<uint32_t>(arg++);
    const uint32_t sync_y = get_arg_val<uint32_t>(arg++);
    const uint32_t virtual_x_arg = arg, virtual_y_arg = arg + grid_w;

    const Noc noc;
    cb_scratch.reserve_back(1);
    const uint32_t scratch = cb_scratch.get_write_ptr();
    tt_l1_ptr uint32_t* expert_row_counts = reinterpret_cast<tt_l1_ptr uint32_t*>(scratch + scratch_counts);
    tt_l1_ptr uint32_t* expert_first_rows = reinterpret_cast<tt_l1_ptr uint32_t*>(scratch + scratch_offsets);
    const tt_l1_ptr uint16_t* row_tokens =
        reinterpret_cast<const tt_l1_ptr uint16_t*>(scratch + scratch_table + table_rows);

    const auto table = TensorAccessor(table_args, table_addr, table_bytes);
    const uint32_t routed_rows = read_routing_table(noc, table, scratch, expert_row_counts, expert_first_rows);

    write_count_rows(noc, scratch, expert_row_counts, core_index, counts_addr, virtual_x_arg, virtual_y_arg);
    const auto e_t = TensorAccessor(e_t_args, e_t_addr, e_t_page_bytes);
    write_e_t_pages(noc, e_t, scratch, expert_row_counts, expert_first_rows, row_tokens, core_index);
    const auto activation = TensorAccessor(activation_args, activation_addr, tokens * activation_row_bytes);
    const uint32_t activated = write_activated_rows(noc, activation, scratch, routed_rows, core_index);
    write_empty_activation_rows(noc, activation, scratch, activated, core_index);

    if constexpr (place_dense) {
        const auto row_buffer = TensorAccessor(rows_args, rows_addr, row_bytes);
        place_double_buffer_shard(
            noc, row_buffer, expert_row_counts, expert_first_rows, double_buffer, height_shard, width_shard);
    }
    noc.async_read_barrier();
    noc.async_write_barrier();
    if constexpr (!place_dense) {
        signal_metadata_ready(noc, core_index, lead_x, lead_y, sync_x, sync_y);
    }
}
