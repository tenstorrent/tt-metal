// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tile-row permutation TILE reshape writer. Assembles each output tile of a batch in an L1 ring from face-row runs
// of the batch's staged source tiles, then writes it once. Plan page layout: reshape_codegen_tile_rows_plan.hpp.
// The flush before a ring refill waits for every outstanding write, so a second ring would overlap nothing.
#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/debug/assert.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

using namespace tt::data_movement::common;

void kernel_main() {
    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_batch = get_arg_val<uint32_t>(0);
    const uint32_t end_batch = get_arg_val<uint32_t>(1);

    constexpr uint32_t tile_bytes = get_named_compile_time_arg_val("tile_bytes");
    constexpr uint32_t face_row_bytes = get_named_compile_time_arg_val("face_row_bytes");
    constexpr uint32_t plan_cb = get_named_compile_time_arg_val("plan_cb");
    constexpr uint32_t input_cb = get_named_compile_time_arg_val("input_cb");
    constexpr uint32_t ring_cb = get_named_compile_time_arg_val("ring_cb");
    constexpr uint32_t zero_cb = get_named_compile_time_arg_val("zero_cb");
    constexpr uint32_t in_slots = get_named_compile_time_arg_val("in_slots");
    constexpr uint32_t entry_words = get_named_compile_time_arg_val("entry_words");
    constexpr uint32_t write_group = get_named_compile_time_arg_val("write_group");
    constexpr uint32_t header_words = get_named_compile_time_arg_val("header_words");
    constexpr uint32_t entry_header_words = get_named_compile_time_arg_val("entry_header_words");
    constexpr uint32_t zero_slot = get_named_compile_time_arg_val("zero_slot");
    constexpr uint32_t slot_shift = get_named_compile_time_arg_val("slot_shift");
    constexpr uint32_t src_row_shift = get_named_compile_time_arg_val("src_row_shift");
    constexpr uint32_t dst_row_shift = get_named_compile_time_arg_val("dst_row_shift");
    constexpr uint32_t row_mask = get_named_compile_time_arg_val("row_mask");
    constexpr auto output_args = TensorAccessorArgs<0>();

    // 32x32 tiles of four 16x16 faces; the host rejects any other tile shape.
    constexpr uint32_t face_rows = 16;
    constexpr uint32_t face_bytes = face_rows * face_row_bytes;
    static_assert(tile_bytes == 4 * face_bytes, "tile is four 16x16 faces");
    static_assert(write_group > 0);
    static_assert(entry_words > entry_header_words, "an entry holds at least one run");
    static_assert(zero_slot >= in_slots, "zero_slot never names a real slot");
    static_assert(row_mask >= 2 * face_rows - 1, "row fields hold a tile row index");

    const auto output_accessor = TensorAccessor(output_args, output_addr);
    Noc noc;
    CircularBuffer plan_buffer(plan_cb);
    CircularBuffer input_buffer(input_cb);
    CircularBuffer ring(ring_cb);
    CircularBuffer zero_tile(zero_cb);

    // Both are private scratch that is never pushed, so the write pointer stays at the buffer base.
    const uint32_t zero_l1 = zero_tile.get_write_ptr();
    noc.async_write_zeros(zero_tile, tile_bytes);
    noc.write_zeros_l1_barrier();
    const uint32_t ring_l1 = ring.get_write_ptr();

    uint32_t group_size = 0;
    uint32_t group_pages[write_group];

    auto write_group_out = [&]() {
        noc.async_read_barrier();
        for (uint32_t k = 0; k < group_size; ++k) {
            noc.async_write<NocOptions::DEFAULT, tile_bytes>(
                CoreLocalMem<uint8_t>(ring_l1 + k * tile_bytes),
                output_accessor,
                tile_bytes,
                {.offset_bytes = 0},
                {.page_id = group_pages[k], .offset_bytes = 0});
        }
        group_size = 0;
    };

    for (uint32_t batch = start_batch; batch < end_batch; ++batch) {
        plan_buffer.wait_front(1);
        input_buffer.wait_front(in_slots);
        CoreLocalMem<volatile uint32_t> plan(plan_buffer.get_read_ptr());
        const uint32_t input_l1 = input_buffer.get_read_ptr();
        const uint32_t n_out = plan[1];

        for (uint32_t j = 0; j < n_out; ++j) {
            const uint32_t entry = header_words + in_slots + j * entry_words;
            if (group_size == 0) {
                noc.async_writes_flushed();
            }
            const uint32_t dst_tile = ring_l1 + group_size * tile_bytes;
            const uint32_t n_runs = plan[entry + 1];
            ASSERT(n_runs <= entry_words - entry_header_words);
            for (uint32_t r = 0; r < n_runs; ++r) {
                const uint32_t run = plan[entry + entry_header_words + r];
                const uint32_t slot = run >> slot_shift;
                const uint32_t src_row = (run >> src_row_shift) & row_mask;
                const uint32_t dst_row = (run >> dst_row_shift) & row_mask;
                const uint32_t bytes = (run & row_mask) * face_row_bytes;
                ASSERT(slot == zero_slot || slot < in_slots);
                const uint32_t src_tile = slot == zero_slot ? zero_l1 : input_l1 + slot * tile_bytes;
                const uint32_t src =
                    src_tile + (src_row >= face_rows ? 2 * face_bytes : 0) + (src_row % face_rows) * face_row_bytes;
                const uint32_t dst =
                    dst_tile + (dst_row >= face_rows ? 2 * face_bytes : 0) + (dst_row % face_rows) * face_row_bytes;
                tt_memmove<true, true, true, face_bytes>(noc, dst, src, bytes);
                tt_memmove<true, true, true, face_bytes>(noc, dst + face_bytes, src + face_bytes, bytes);
            }
            group_pages[group_size++] = plan[entry];
            if (group_size == write_group) {
                write_group_out();
            }
        }
        // The partial group's self-reads still source this batch's input slots.
        if (group_size != 0) {
            write_group_out();
        }
        input_buffer.pop_front(in_slots);
        plan_buffer.pop_front(1);
    }
    noc.async_write_barrier();
}
