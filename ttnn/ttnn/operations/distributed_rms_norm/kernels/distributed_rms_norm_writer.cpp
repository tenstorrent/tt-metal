// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// distributed_rms_norm — worker writer (BRISC, NoC1).
//   publish_records_block : cb_partial (fp32 tile per tile row; row 0 = the 32 per-row partial sums after the
//                           in-DEST transpose) -> one 128 B record per tile row into my device's slot of every
//                           sending port of my link; one barrier, then one local-arrival increment per port.
//   store_block           : cb_output_tiles -> DRAM, one tile row (out_batch_tiles = core_w_tiles) per barrier.
//
// Raw-API justification: a record is a sub-tile row (two 64 B face rows) landing in a port core's L1 scratch at a
// host-computed offset; no dataflow helper places partial tiles into a remote L1 slot.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_partial = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t sem_local_arrivals = get_compile_time_arg_val(2);
    constexpr auto out_args = TensorAccessorArgs<3>();

    constexpr uint32_t record_bytes = 128;
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
        for (uint32_t j = 0; j < rows; ++j) {
            const uint32_t src = rp + j * partial_tile_bytes;
            const uint32_t dst = my_slot_addr + (rec_base + (r0 + j) * s_w + w_idx) * record_bytes;
            for (uint32_t i = 0; i < num_ports; ++i) {
                noc_async_write(src, get_noc_addr(port_x[i], port_y[i], dst), half_record);  // rows 0-15
                noc_async_write(
                    src + f32_face_bytes, get_noc_addr(port_x[i], port_y[i], dst + half_record), half_record);
            }
        }
        noc_async_write_barrier();
        for (uint32_t i = 0; i < num_ports; ++i) {
            noc_semaphore_inc(get_noc_addr(port_x[i], port_y[i], local_arrivals), 1);
        }
        cb_pop_front(cb_partial, rows);

        // ---- store_block: one tile row of my W chunk per barrier ----
        for (uint32_t j = 0; j < rows; ++j) {
            cb_wait_front(cb_output_tiles, cw);
            uint32_t rd = get_read_ptr(cb_output_tiles);
            const uint32_t tile0 = (row_start + r0 + j) * wt_total + w_start;
            for (uint32_t t = 0; t < cw; ++t) {
                noc_async_write_page(tile0 + t, out, rd);
                rd += out_tile_bytes;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_output_tiles, cw);
        }
    }
    noc_async_atomic_barrier();
}
