// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// distributed_rms_norm — worker writer (BRISC, NoC1).
//   publish_records_block : cb_partial (fp32 tile per tile row; row 0 = the 32 per-row partial sums after the
//                           in-DEST transpose) -> one 128 B record per tile row into my device's slot of every
//                           sending port of my link; one barrier, then one local-arrival increment per port.
//                           R5 pre-combine (sharded WIDTH): the records go instead to the inbox of the combiner
//                           owning each tile row, then one local-combine increment per combiner; a combiner later
//                           publishes its combined records (cb_lc_out, one record per tile row) to the ports.
//   store_block           : cb_output_tiles -> DRAM, out_batch_tiles per barrier (host knob out_batch_tiles_knob;
//                           it divides every normalize chunk, so a batch never straddles a tile row or the CB end).
//                           Walk order = compute's: stats batch, normalize chunk, tile row, tile. Absent when the
//                           output is a native shard (compute packs into it).
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
    constexpr bool out_native = get_compile_time_arg_val(4) != 0;   // cb_output_tiles backed on the output shard
    constexpr bool local_combine = get_compile_time_arg_val(5) != 0;  // R5 device-local pre-combine
    constexpr uint32_t cb_lc_out = get_compile_time_arg_val(6);
    constexpr uint32_t sem_local_combine = get_compile_time_arg_val(7);
    constexpr bool x_sharded = get_compile_time_arg_val(8) != 0;  // sharded-path runtime args present
    constexpr uint32_t round_lookahead = get_compile_time_arg_val(9);  // rounds published ahead (host knob)
    constexpr uint32_t local_arrival_sems = 1 + round_lookahead;       // round b -> counter b % local_arrival_sems
    constexpr auto out_args = TensorAccessorArgs<10>();

    constexpr uint32_t half_record = record_bytes / 2;
    constexpr uint32_t f32_face_bytes = 1024;
    constexpr uint32_t max_ports = 2;

    // common args (one copy per device; see the reader) + per-core args (only what differs between cores)
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t block_rows = get_common_arg_val<uint32_t>(1);
    const uint32_t wt_total = get_common_arg_val<uint32_t>(2);
    const uint32_t my_slot_addr = get_common_arg_val<uint32_t>(3);  // scratch + p * slot_bytes (same on every port)
    const uint32_t s_w = get_common_arg_val<uint32_t>(4);           // records per tile row of the block I publish into
    const uint32_t num_ports = get_common_arg_val<uint32_t>(5);     // sending ports of a link (1 or 2)
    constexpr uint32_t link_table = 6;  // per link l: [round_recs, num_ports x (noc x, noc y)]
    size_t a = 0;
    const uint32_t row_start = get_arg_val<uint32_t>(a++);
    const uint32_t core_rows = get_arg_val<uint32_t>(a++);
    const uint32_t w_start = get_arg_val<uint32_t>(a++);
    const uint32_t cw = get_arg_val<uint32_t>(a++);
    const uint32_t rec_base = get_arg_val<uint32_t>(a++);
    const uint32_t w_idx = get_arg_val<uint32_t>(a++);      // my W chunk
    const uint32_t out_batch = get_arg_val<uint32_t>(a++);  // tiles per writer barrier (divides cw)
    const uint32_t my_link = get_arg_val<uint32_t>(a++);
    const uint32_t link_idx = link_table + my_link * (1 + 2 * num_ports);
    const uint32_t round_recs = get_common_arg_val<uint32_t>(link_idx);  // records per round in my link's region
    uint32_t port_x[max_ports], port_y[max_ports];
    for (uint32_t i = 0; i < num_ports; ++i) {
        port_x[i] = get_common_arg_val<uint32_t>(link_idx + 1 + 2 * i);
        port_y[i] = get_common_arg_val<uint32_t>(link_idx + 2 + 2 * i);
    }
    // sharded-path args (R9/R10); the interleaved path keeps its Phase 0 arg list and these defaults
    uint32_t stat_rows = block_rows, norm_chunk = cw, lc_rows = 0, lc_start = 0, inbox_addr = 0, lc_per = 1, n_comb = 0;
    if constexpr (x_sharded) {
        stat_rows = get_arg_val<uint32_t>(a++);   // tile rows per stats batch (knob)
        norm_chunk = get_arg_val<uint32_t>(a++);  // W tiles per normalize chunk (== cw unless gamma streams)
        lc_rows = get_arg_val<uint32_t>(a++);     // tile rows I pre-combine (0: not a combiner)
        lc_start = get_arg_val<uint32_t>(a++);    // first of them
        inbox_addr = get_arg_val<uint32_t>(a++);
        lc_per = get_arg_val<uint32_t>(a++);  // tile rows per combiner
        n_comb = get_arg_val<uint32_t>(a++);  // combiners of my row group, then their (x, y)
    }
    const uint32_t comb_idx = a;

    if (core_rows == 0) {
        return;  // idle core of the worker rectangle
    }

    const uint32_t partial_tile_bytes = get_tile_size(cb_partial);
    const uint32_t out_tile_bytes = get_tile_size(cb_output_tiles);
    const auto out = TensorAccessor(out_args, out_addr, out_tile_bytes);

    // publish one record (two 64 B face-row halves of the partial tile's row 0) into a [half 0 of w][half 1 of w]
    // block of rw records at (x, y)
    auto send_record = [&](uint32_t src, uint32_t x, uint32_t y, uint32_t block, uint32_t rw, uint32_t w) {
        noc_async_write(src, get_noc_addr(x, y, block + w * half_record), half_record);
        noc_async_write(src + f32_face_bytes, get_noc_addr(x, y, block + (rw + w) * half_record), half_record);
    };

    const uint32_t num_blocks = (core_rows + block_rows - 1) / block_rows;
    auto rows_of = [&](uint32_t b) {
        const uint32_t r0 = b * block_rows;
        return (core_rows - r0) < block_rows ? (core_rows - r0) : block_rows;
    };

    auto publish = [&](uint32_t round) {
        const uint32_t rows = rows_of(round);
        const uint32_t round_base = rec_base + round * round_recs;
        const uint32_t local_arrivals = get_semaphore(sem_local_arrivals + round % local_arrival_sems);

        // ---- publish_records_block, one stats batch at a time ----
        for (uint32_t b0 = 0; b0 < rows; b0 += stat_rows) {
            const uint32_t nb = (rows - b0) < stat_rows ? (rows - b0) : stat_rows;
            cb_wait_front(cb_partial, nb);
            const uint32_t rp = get_read_ptr(cb_partial);
            for (uint32_t j = 0; j < nb; ++j) {
                const uint32_t src = rp + j * partial_tile_bytes;
                const uint32_t row = round * block_rows + b0 + j;
                if constexpr (local_combine) {
                    // record of my W chunk for this tile row -> the inbox of the combiner owning the row
                    const uint32_t c = row / lc_per;
                    const uint32_t cx = get_arg_val<uint32_t>(comb_idx + 2 * c);
                    const uint32_t cy = get_arg_val<uint32_t>(comb_idx + 2 * c + 1);
                    send_record(src, cx, cy, inbox_addr + (row - c * lc_per) * s_w * record_bytes, s_w, w_idx);
                } else {
                    const uint32_t block = my_slot_addr + (round_base + (b0 + j) * s_w) * record_bytes;
                    for (uint32_t i = 0; i < num_ports; ++i) {
                        send_record(src, port_x[i], port_y[i], block, s_w, w_idx);
                    }
                }
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_partial, nb);
        }
        noc_async_write_barrier();
        if constexpr (local_combine) {
            for (uint32_t c = 0; c < n_comb; ++c) {
                noc_semaphore_inc(
                    get_noc_addr(
                        get_arg_val<uint32_t>(comb_idx + 2 * c),
                        get_arg_val<uint32_t>(comb_idx + 2 * c + 1),
                        get_semaphore(sem_local_combine)),
                    1);
            }
            // combiner: publish the combined record of each of my tile rows (one record per row on the exchange)
            if (lc_rows > 0) {
                cb_wait_front(cb_lc_out, lc_rows);
                const uint32_t lp = get_read_ptr(cb_lc_out);
                for (uint32_t j = 0; j < lc_rows; ++j) {
                    const uint32_t block = my_slot_addr + (round_base + lc_start + j) * record_bytes;
                    for (uint32_t i = 0; i < num_ports; ++i) {
                        send_record(lp + j * partial_tile_bytes, port_x[i], port_y[i], block, 1, 0);
                    }
                }
                noc_async_write_barrier();
                for (uint32_t i = 0; i < num_ports; ++i) {
                    noc_semaphore_inc(get_noc_addr(port_x[i], port_y[i], local_arrivals), 1);
                }
                cb_pop_front(cb_lc_out, lc_rows);
            }
        } else {
            for (uint32_t i = 0; i < num_ports; ++i) {
                noc_semaphore_inc(get_noc_addr(port_x[i], port_y[i], local_arrivals), 1);
            }
        }
    };

    // ---- store_block: out_batch tiles per barrier, in compute's (batch, chunk, row, tile) order ----
    auto store = [&](uint32_t round) {
        [[maybe_unused]] const uint32_t r0 = round * block_rows;
        const uint32_t rows = rows_of(round);
        if constexpr (!out_native) {
            for (uint32_t b0 = 0; b0 < rows; b0 += stat_rows) {
                const uint32_t nb = (rows - b0) < stat_rows ? (rows - b0) : stat_rows;
                for (uint32_t c0 = 0; c0 < cw; c0 += norm_chunk) {
                    const uint32_t wc = (cw - c0) < norm_chunk ? (cw - c0) : norm_chunk;
                    for (uint32_t j = 0; j < nb; ++j) {
                        const uint32_t row_tile0 = (row_start + r0 + b0 + j) * wt_total + w_start + c0;
                        for (uint32_t t0 = 0; t0 < wc; t0 += out_batch) {
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
            }
        }
    };

    // R2 schedule: round b+lookahead's records are published before round b's store waits on normalize, so the
    // exchange of the later round runs under round b's writes
    for (uint32_t round = 0; round < round_lookahead && round < num_blocks; ++round) {
        publish(round);
    }
    for (uint32_t round = 0; round < num_blocks; ++round) {
        if (round + round_lookahead < num_blocks) {
            publish(round + round_lookahead);
        }
        store(round);
    }
    noc_async_atomic_barrier();
}
