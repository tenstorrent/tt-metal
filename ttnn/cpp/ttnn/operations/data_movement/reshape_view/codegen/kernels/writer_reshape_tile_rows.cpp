// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tile-row permutation TILE reshape writer. Assembles each output tile of a
// batch in an L1 ring from face-row runs of the batch's staged source tiles
// (local NOC self-reads), then writes each output tile once.
//
// Plan page (uint32): [n_in, n_out, in_page[in_slots], entry[out_slots]]
// with entry = [out_page, n_runs, run[max_runs]] (entry_words each) and
// run = slot << slot_shift | src_row << src_row_shift | dst_row << dst_row_shift
// | rows. A run never crosses tile row 16 on either side, so it is one
// contiguous span in each of the two faces of that half. slot == zero_slot
// copies from a pre-zeroed tile (TILE padding rows). The host planner
// (reshape_codegen_tile_rows_plan.hpp) owns this layout and passes every field
// position here as a compile-time arg.
//
// Output tiles are written per ring of write_group tiles after one read
// barrier; before the ring is refilled, writes_flushed guarantees its
// previous DRAM writes have left L1. The flush waits for every outstanding
// write, so a second ring would not overlap anything.
#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/debug/assert.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

using namespace tt::data_movement::common;

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_batch = get_arg_val<uint32_t>(1);
    const uint32_t end_batch = get_arg_val<uint32_t>(2);

    constexpr uint32_t tile_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t face_row_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t plan_cb = get_compile_time_arg_val(2);
    constexpr uint32_t input_cb = get_compile_time_arg_val(3);
    constexpr uint32_t ring_cb = get_compile_time_arg_val(4);
    constexpr uint32_t zero_cb = get_compile_time_arg_val(5);
    constexpr uint32_t in_slots = get_compile_time_arg_val(6);
    constexpr uint32_t entry_words = get_compile_time_arg_val(7);
    constexpr uint32_t write_group = get_compile_time_arg_val(8);
    constexpr uint32_t header_words = get_compile_time_arg_val(9);
    constexpr uint32_t entry_header_words = get_compile_time_arg_val(10);
    constexpr uint32_t zero_slot = get_compile_time_arg_val(11);
    constexpr uint32_t slot_shift = get_compile_time_arg_val(12);
    constexpr uint32_t src_row_shift = get_compile_time_arg_val(13);
    constexpr uint32_t dst_row_shift = get_compile_time_arg_val(14);
    constexpr uint32_t row_mask = get_compile_time_arg_val(15);
    constexpr auto output_args = TensorAccessorArgs<16>();

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

    zero_tile.reserve_back(1);
    const uint32_t zero_l1 = zero_tile.get_write_ptr();
    noc.async_write_zeros(zero_tile, tile_bytes);
    noc.write_zeros_l1_barrier();
    ring.reserve_back(write_group);
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
    ring.push_back(write_group);
    zero_tile.push_back(1);
}
