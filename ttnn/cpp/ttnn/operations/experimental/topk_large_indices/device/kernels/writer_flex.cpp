// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Row-parallel FLEX output writer: serves the opt-in output formats of
// topk_large_indices (tile_output=true and/or index_dtype=UINT16), selected by
// the program factory via the FLEX_OUT_TILE / FLEX_INDEX_U16 / FLEX_WITH_VALUES
// defines. The default ROW_MAJOR/UINT32 program keeps using writer.cpp /
// writer_with_values.cpp so its kernel binaries stay byte-identical.
//
// Consumes exactly the same compute-produced CB stream as the default writers
// (values page then indices page per row when FLEX_WITH_VALUES). See
// topk_large_indices_writer_flex_common.hpp for the TILE scatter/padding and
// UINT16 narrowing mechanics.
//
// TILE row addressing: TILE padding is per 2D slice — every [rows_2d, k] slab
// of the (flattened) output pads its rows to a multiple of 32 independently —
// so the writer tracks (slice_idx, in_slice_row) across its contiguous global
// row range and the core that owns a slab's LAST logical row zero-fills that
// slab's padding rows.

#include "topk_large_indices_writer_flex_common.hpp"

void kernel_main() {
    using namespace topk_large_indices_writer_flex;

    const uint32_t indices_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_row = get_arg_val<uint32_t>(1);
    const uint32_t num_rows = get_arg_val<uint32_t>(2);
#ifdef FLEX_WITH_VALUES
    const uint32_t values_addr = get_arg_val<uint32_t>(3);
#endif
#ifdef FLEX_OUT_TILE
    const uint32_t rows_2d = get_arg_val<uint32_t>(4);
    const uint32_t start_in_slice = get_arg_val<uint32_t>(5);
    const uint32_t start_slice = get_arg_val<uint32_t>(6);
#endif

    constexpr uint32_t cb_indices = get_compile_time_arg_val(0);
    constexpr uint32_t cb_indices_scratch = get_compile_time_arg_val(1);
    constexpr uint32_t indices_page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t source_slices_per_row = get_compile_time_arg_val(3);
    constexpr uint32_t output_slices_per_row = get_compile_time_arg_val(4);
    constexpr uint32_t cb_values = get_compile_time_arg_val(5);
    constexpr uint32_t cb_values_scratch = get_compile_time_arg_val(6);
    constexpr uint32_t values_page_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t cb_pad_zero = get_compile_time_arg_val(8);
    constexpr uint32_t tiles_per_out_row = get_compile_time_arg_val(9);
    constexpr auto indices_args = TensorAccessorArgs<10>();
#ifdef FLEX_WITH_VALUES
    constexpr auto values_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
#endif

#ifdef FLEX_INDEX_U16
    constexpr uint32_t indices_elem_bytes = 2;
#else
    constexpr uint32_t indices_elem_bytes = 4;
#endif
    constexpr bool cb_row_major = (source_slices_per_row == 32);

    const auto indices = TensorAccessor(indices_args, indices_addr, indices_page_bytes);
    CircularBuffer indices_cb(cb_indices);
#ifdef FLEX_WITH_VALUES
    const auto values = TensorAccessor(values_args, values_addr, values_page_bytes);
    CircularBuffer values_cb(cb_values);
#endif
    Noc noc;

#ifdef FLEX_OUT_TILE
    const uint32_t zero_base = init_pad_slab(cb_pad_zero);
    const uint32_t tiles_per_slice_h = (rows_2d + 31) >> 5;
    const uint32_t pad_from = rows_2d & 31;  // 0 = slabs fill their tiles exactly, no padding
    uint32_t slice_idx = start_slice;
    uint32_t in_slice_row = start_in_slice;

    if (pad_from != 0) {
        // Zero-fill the padding rows of every 2D slab whose LAST logical row
        // this core owns. Pad rows are disjoint from every data row, so they
        // are pre-issued here — before the first compute-produced row is even
        // awaited — hiding their NoC injection under the compute pipeline
        // instead of tailing the critical path.
        const uint32_t end_row = start_row + num_rows;  // exclusive
        const uint32_t last_tile_row_in_slab = (rows_2d - 1) >> 5;
        for (uint32_t slab = start_slice;; ++slab) {
            const uint32_t slab_last_row = slab * rows_2d + (rows_2d - 1);
            if (slab_last_row >= end_row) {
                break;
            }
            const uint32_t pad_tile_row = slab * tiles_per_slice_h + last_tile_row_in_slab;
            issue_tile_row_pad<indices_elem_bytes>(noc, indices, zero_base, pad_tile_row, pad_from, tiles_per_out_row);
#ifdef FLEX_WITH_VALUES
            issue_tile_row_pad<2>(noc, values, zero_base, pad_tile_row, pad_from, tiles_per_out_row);
#endif
        }
    }
#endif

    for (uint32_t local_row = 0; local_row < num_rows; ++local_row) {
        const uint32_t row = start_row + local_row;
        (void)row;
#ifdef FLEX_OUT_TILE
        const uint32_t tile_row = slice_idx * tiles_per_slice_h + (in_slice_row >> 5);
        const uint32_t in_tile_r = in_slice_row & 31;
#endif

#ifdef FLEX_WITH_VALUES
        // Compute pushes values then indices each row; consume in the same order.
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
                {.page_id = row, .offset_bytes = 0});
#endif
            noc.async_writes_flushed();
            indices_scratch_cb.pop_front(1);
        }
#else
        // UINT32 indices; the flex writer is only selected with at least one
        // opt-in, so this is necessarily the TILE-output path.
        indices_cb.wait_front(1);
        issue_tile_row_scatter<source_slices_per_row, output_slices_per_row, 64, 4, cb_row_major>(
            noc, indices, indices_cb.get_read_ptr(), tile_row, in_tile_r, tiles_per_out_row);
        noc.async_writes_flushed();
        indices_cb.pop_front(1);
#endif

#ifdef FLEX_OUT_TILE
        if (++in_slice_row == rows_2d) {
            in_slice_row = 0;
            ++slice_idx;
        }
#endif
    }

    noc.async_write_barrier();
}
