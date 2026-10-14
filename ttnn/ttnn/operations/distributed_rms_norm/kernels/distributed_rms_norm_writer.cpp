// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// distributed_rms_norm — worker writer (BRISC, NoC1).
//   publish_records_block : cb_partial (fp32 tile per tile row; row 0 = the 32 per-row partial sums after the
//                           in-DEST transpose) -> one 128 B record per tile row into my device's slot of every
//                           sending port of my link; one barrier, then one local-arrival increment per port.
//   store_block           : cb_output_tiles -> DRAM, out_batch_tiles per barrier (host knob out_batch_tiles_knob;
//                           it divides the core's W chunk, so a batch never straddles a tile row or the CB end).
//
// Raw-API justification: a record is a sub-tile row (two 64 B face rows) landing in a port core's L1 scratch at a
// host-computed offset; no dataflow helper places partial tiles into a remote L1 slot.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_partial = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t sem_local_arrivals = get_compile_time_arg_val(2);
    constexpr uint32_t record_bytes = get_compile_time_arg_val(3);  // host RECORD_BYTES: 32 fp32 partial sums
    constexpr auto out_args = TensorAccessorArgs<4>();

    constexpr uint32_t half_record = record_bytes / 2;
    constexpr uint32_t f32_face_bytes = 1024;
    constexpr uint32_t max_ports = 2;

    size_t a = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t row_start = get_arg_val<uint32_t>(a++);
    const uint32_t core_rows = get_arg_val<uint32_t>(a++);
    const uint32_t block_rows = get_arg_val<uint32_t>(a++);
    const uint32_t w_start = get_arg_val<uint32_t>(a++);
    const uint32_t cw = get_arg_val<uint32_t>(a++);
    const uint32_t wt_total = get_arg_val<uint32_t>(a++);
    const uint32_t my_slot_addr = get_arg_val<uint32_t>(a++);  // scratch + p * slot_bytes (same on every port)
    const uint32_t rec_base = get_arg_val<uint32_t>(a++);
    const uint32_t s_w = get_arg_val<uint32_t>(a++);
    const uint32_t w_idx = get_arg_val<uint32_t>(a++);
    const uint32_t out_batch = get_arg_val<uint32_t>(a++);  // tiles per writer barrier (divides cw)
    const uint32_t num_ports = get_arg_val<uint32_t>(a++);  // sending ports of my link (1 or 2)
    uint32_t port_x[max_ports], port_y[max_ports];
    for (uint32_t i = 0; i < num_ports; ++i) {
        port_x[i] = get_arg_val<uint32_t>(a++);
        port_y[i] = get_arg_val<uint32_t>(a++);
    }

    if (core_rows == 0) {
        return;  // idle core of the worker rectangle
    }

    const uint32_t partial_tile_bytes = get_tile_size(cb_partial);
    const uint32_t out_tile_bytes = get_tile_size(cb_output_tiles);
    const auto out = TensorAccessor(out_args, out_addr, out_tile_bytes);
    const uint32_t local_arrivals = get_semaphore(sem_local_arrivals);

    for (uint32_t r0 = 0; r0 < core_rows; r0 += block_rows) {
        const uint32_t rows = (core_rows - r0) < block_rows ? (core_rows - r0) : block_rows;

        // ---- publish_records_block ----
        cb_wait_front(cb_partial, rows);
        const uint32_t rp = get_read_ptr(cb_partial);
        // record block of tile row j: [half 0 of w = 0..s_w-1][half 1 of w = 0..s_w-1] (see the reader's gather)
        for (uint32_t j = 0; j < rows; ++j) {
            const uint32_t src = rp + j * partial_tile_bytes;
            const uint32_t block = my_slot_addr + (rec_base + (r0 + j) * s_w) * record_bytes;
            const uint32_t dst0 = block + w_idx * half_record;          // rows 0-15 of the tile row
            const uint32_t dst1 = block + (s_w + w_idx) * half_record;  // rows 16-31
            for (uint32_t i = 0; i < num_ports; ++i) {
                noc_async_write(src, get_noc_addr(port_x[i], port_y[i], dst0), half_record);
                noc_async_write(src + f32_face_bytes, get_noc_addr(port_x[i], port_y[i], dst1), half_record);
            }
        }
        noc_async_write_barrier();
        for (uint32_t i = 0; i < num_ports; ++i) {
            noc_semaphore_inc(get_noc_addr(port_x[i], port_y[i], local_arrivals), 1);
        }
        cb_pop_front(cb_partial, rows);

        // ---- store_block: out_batch tiles of my W chunk per barrier ----
        for (uint32_t j = 0; j < rows; ++j) {
            const uint32_t row_tile0 = (row_start + r0 + j) * wt_total + w_start;
            for (uint32_t t0 = 0; t0 < cw; t0 += out_batch) {
                cb_wait_front(cb_output_tiles, out_batch);
                uint32_t rd = get_read_ptr(cb_output_tiles);
                for (uint32_t t = 0; t < out_batch; ++t) {
                    noc_async_write_page(row_tile0 + t0 + t, out, rd);
                    rd += out_tile_bytes;
                }
                noc_async_write_barrier();
                cb_pop_front(cb_output_tiles, out_batch);
            }
        }
    }
    noc_async_atomic_barrier();
}
