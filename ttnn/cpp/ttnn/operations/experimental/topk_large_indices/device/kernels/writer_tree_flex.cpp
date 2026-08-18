// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Column-parallel TREE writer, FLEX output variant: writer_tree.cpp /
// writer_tree_with_values.cpp's inter-core merge-tree traffic (unchanged),
// with the root's output emission swapped for the flex formats
// (FLEX_OUT_TILE / FLEX_INDEX_U16 / FLEX_WITH_VALUES defines) — see
// writer_flex.cpp and topk_large_indices_writer_flex_common.hpp.
//
// The column-parallel factory is only selected for single-row inputs
// (num_rows == 1, so every leading dim is 1): each output row is its own 2D
// slab — tile row == row, in-tile row 0, padding rows [1, 32).

#include "api/dataflow/noc_semaphore.h"

#include "topk_large_indices_writer_flex_common.hpp"

void kernel_main() {
    using namespace topk_large_indices_writer_flex;

    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    const uint32_t num_recv = get_arg_val<uint32_t>(1);
    // Up to 7 (x, y) partner pairs (P <= 128 -> <= 7 levels). Offsets must
    // match the factory's partner_coords(14) writer-args block.
    uint32_t partner_x[7];
    uint32_t partner_y[7];
    for (uint32_t m = 0; m < 7; ++m) {
        partner_x[m] = get_arg_val<uint32_t>(2 + 2 * m);
        partner_y[m] = get_arg_val<uint32_t>(3 + 2 * m);
    }
    const uint32_t do_ship = get_arg_val<uint32_t>(16);
    const uint32_t winner_x = get_arg_val<uint32_t>(17);
    const uint32_t winner_y = get_arg_val<uint32_t>(18);
    const uint32_t is_empty_ship = get_arg_val<uint32_t>(19);
    const uint32_t indices_addr = get_arg_val<uint32_t>(20);
    // Multi-rectangle: this rectangle's first output row (0 on a single-rect
    // program; the factory forbids tile_output with multiple rectangles, so
    // the TILE scatter paths below keep their rect-local tile_row).
    const uint32_t start_row_rt = get_arg_val<uint32_t>(22);
#ifdef FLEX_WITH_VALUES
    const uint32_t values_addr = get_arg_val<uint32_t>(21);
#endif

    constexpr uint32_t cb_ship_values = get_compile_time_arg_val(0);
    constexpr uint32_t cb_ship_indices = get_compile_time_arg_val(1);
    constexpr uint32_t cb_neginf_scratch = get_compile_time_arg_val(2);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(3);
    constexpr uint32_t ready_sem_id = get_compile_time_arg_val(4);
    constexpr uint32_t data_sem_id = get_compile_time_arg_val(5);
    constexpr uint32_t tiles_per_sequence = get_compile_time_arg_val(6);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t cb_indices_out = get_compile_time_arg_val(8);
    constexpr uint32_t cb_indices_scratch = get_compile_time_arg_val(9);
    constexpr uint32_t indices_page_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t source_slices_per_row = get_compile_time_arg_val(11);
    constexpr uint32_t output_slices_per_row = get_compile_time_arg_val(12);
    constexpr uint32_t cb_values_out = get_compile_time_arg_val(13);
    constexpr uint32_t cb_values_scratch = get_compile_time_arg_val(14);
    constexpr uint32_t values_page_bytes = get_compile_time_arg_val(15);
    constexpr uint32_t cb_pad_zero = get_compile_time_arg_val(16);
    constexpr uint32_t tiles_per_out_row = get_compile_time_arg_val(17);
    constexpr auto indices_args = TensorAccessorArgs<18>();
#ifdef FLEX_WITH_VALUES
    constexpr auto values_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
#endif

#ifdef FLEX_INDEX_U16
    constexpr uint32_t indices_elem_bytes = 2;
#else
    constexpr uint32_t indices_elem_bytes = 4;
#endif
    constexpr bool cb_row_major = (source_slices_per_row == 32);

    constexpr uint32_t sequence_tiles = 2 * tiles_per_sequence;
    constexpr uint32_t sequence_bytes = tiles_per_sequence * tile_bytes;

    Noc noc;
    Semaphore<> ready_sem(ready_sem_id);
    Semaphore<> data_sem(data_sem_id);
    UnicastEndpoint remote;
    CircularBuffer ship_values_cb(cb_ship_values);
    CircularBuffer ship_indices_cb(cb_ship_indices);
    CircularBuffer recv_cb(cb_recv);

    const uint32_t recv_values_base = recv_cb.get_write_ptr();
    const uint32_t recv_indices_base = recv_values_base + sequence_bytes;

    if (do_ship != 0 && is_empty_ship != 0) {
        CircularBuffer scratch_cb(cb_neginf_scratch);
        scratch_cb.reserve_back(sequence_tiles);
        const uint32_t scratch_base = scratch_cb.get_write_ptr();
        constexpr uint32_t sequence_words = sequence_bytes / sizeof(uint32_t);
        volatile tt_l1_ptr uint32_t* scratch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_base);
        for (uint32_t i = 0; i < sequence_words; ++i) {
            scratch[i] = 0xFF800000u;
        }
        for (uint32_t i = 0; i < sequence_words; ++i) {
            scratch[sequence_words + i] = 0xFFFFFFFFu;
        }
    }

    const auto indices = TensorAccessor(indices_args, indices_addr, indices_page_bytes);
    CircularBuffer indices_cb(cb_indices_out);
#ifdef FLEX_WITH_VALUES
    const auto values = TensorAccessor(values_args, values_addr, values_page_bytes);
    CircularBuffer values_cb(cb_values_out);
#endif

#ifdef FLEX_OUT_TILE
    if (do_ship == 0) {
        // Root: pre-issue every slab's padding rows (each output row is its
        // own [1, k] slab -> pad rows [1, 32) of tile row == row) before the
        // merge tree even starts — pad rows are disjoint from data rows, so
        // their NoC injection hides under the compute pipeline.
        const uint32_t zero_base = init_pad_slab(cb_pad_zero);
        for (uint32_t row = 0; row < num_rows; ++row) {
            issue_tile_row_pad<indices_elem_bytes>(noc, indices, zero_base, row, /*pad_from=*/1, tiles_per_out_row);
#ifdef FLEX_WITH_VALUES
            issue_tile_row_pad<2>(noc, values, zero_base, row, /*pad_from=*/1, tiles_per_out_row);
#endif
        }
    }
#endif

    for (uint32_t row = 0; row < num_rows; ++row) {
        for (uint32_t m = 0; m < num_recv; ++m) {
            recv_cb.reserve_back(sequence_tiles);
            data_sem.set(0);
            ready_sem.up(noc, partner_x[m], partner_y[m], 1);
            data_sem.wait(1);
            recv_cb.push_back(sequence_tiles);
        }

        if (do_ship != 0) {
            uint32_t src_values;
            uint32_t src_indices;
            if (is_empty_ship != 0) {
                CircularBuffer scratch_cb(cb_neginf_scratch);
                src_values = scratch_cb.get_write_ptr();
                src_indices = src_values + sequence_bytes;
            } else {
                ship_values_cb.wait_front(tiles_per_sequence);
                ship_indices_cb.wait_front(tiles_per_sequence);
                src_values = ship_values_cb.get_read_ptr();
                src_indices = ship_indices_cb.get_read_ptr();
            }

            ready_sem.wait(1);
            ready_sem.set(0);

            noc.async_write(
                CoreLocalMem<uint32_t>(src_values),
                remote,
                sequence_bytes,
                {.offset_bytes = 0},
                {.noc_x = winner_x, .noc_y = winner_y, .addr = recv_values_base});
            noc.async_write(
                CoreLocalMem<uint32_t>(src_indices),
                remote,
                sequence_bytes,
                {.offset_bytes = 0},
                {.noc_x = winner_x, .noc_y = winner_y, .addr = recv_indices_base});
            noc.async_write_barrier();

            data_sem.up(noc, winner_x, winner_y, 1);
            noc.async_atomic_barrier();

            if (is_empty_ship == 0) {
                ship_values_cb.pop_front(tiles_per_sequence);
                ship_indices_cb.pop_front(tiles_per_sequence);
            }
        } else {
            // Root: values row then indices row (compute pushes in that order).
#ifdef FLEX_OUT_TILE
            const uint32_t tile_row = row;  // single-row slabs: one tile row per output row
            constexpr uint32_t in_tile_r = 0;
#endif

#ifdef FLEX_WITH_VALUES
#ifdef FLEX_OUT_TILE
            values_cb.wait_front(1);
            issue_tile_row_scatter<source_slices_per_row, output_slices_per_row, 32, 2, cb_row_major>(
                noc, values, values_cb.get_read_ptr(), tile_row, in_tile_r, tiles_per_out_row);
            noc.async_writes_flushed();
            values_cb.pop_front(1);
#else
            if constexpr (cb_row_major) {
                issue_contiguous_row_write(values_cb, noc, values, row, values_page_bytes);
                noc.async_writes_flushed();
                values_cb.pop_front(1);
            } else {
                CircularBuffer values_scratch_cb(cb_values_scratch);
                issue_reordered_row_write<source_slices_per_row, output_slices_per_row, 32>(
                    values_cb, values_scratch_cb, noc, values, row, values_page_bytes);
                noc.async_writes_flushed();
                values_scratch_cb.pop_front(1);
            }
#endif
#endif  // FLEX_WITH_VALUES

#ifdef FLEX_INDEX_U16
            {
                CircularBuffer indices_scratch_cb(cb_indices_scratch);
                indices_cb.wait_front(1);
                indices_scratch_cb.reserve_back(1);
                narrow_row_to_u16<source_slices_per_row, output_slices_per_row>(
                    indices_cb.get_read_ptr(), indices_scratch_cb.get_write_ptr());
                indices_cb.pop_front(1);  // narrowed synchronously by this RISC
                indices_scratch_cb.push_back(1);
                indices_scratch_cb.wait_front(1);
#ifdef FLEX_OUT_TILE
                issue_tile_row_scatter<source_slices_per_row, output_slices_per_row, 32, 2, /*src_row_major=*/true>(
                    noc, indices, indices_scratch_cb.get_read_ptr(), tile_row, in_tile_r, tiles_per_out_row);
#else
                noc.async_write(
                    indices_scratch_cb,
                    indices,
                    indices_page_bytes,
                    {.offset_bytes = 0},
                    {.page_id = start_row_rt + row, .offset_bytes = 0});
#endif
                noc.async_writes_flushed();
                indices_scratch_cb.pop_front(1);
            }
#else
            // UINT32 indices; flex requires an opt-in, so this is TILE output.
            indices_cb.wait_front(1);
            issue_tile_row_scatter<source_slices_per_row, output_slices_per_row, 64, 4, cb_row_major>(
                noc, indices, indices_cb.get_read_ptr(), tile_row, in_tile_r, tiles_per_out_row);
            noc.async_writes_flushed();
            indices_cb.pop_front(1);
#endif
        }
    }

    noc.async_write_barrier();
}
