// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include <tt-metalium/constants.hpp>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/chronological_selections/device/kernels/chronology.hpp"

constexpr uint32_t rows_cb = get_compile_time_arg_val(0);
constexpr uint32_t spans_cb = get_compile_time_arg_val(1);
constexpr uint32_t scalar_cb = get_compile_time_arg_val(2);
constexpr uint32_t row_bytes = get_compile_time_arg_val(3);
constexpr uint32_t input_row_tiles = get_compile_time_arg_val(4);
constexpr uint32_t width_tiles = get_compile_time_arg_val(5);
constexpr uint32_t tiles_per_core = get_compile_time_arg_val(6);
constexpr uint32_t sp_rank = get_compile_time_arg_val(7);
constexpr uint32_t sp_size = get_compile_time_arg_val(8);
constexpr uint32_t local_rows = get_compile_time_arg_val(9);
constexpr bool has_actual_end = get_compile_time_arg_val(10);
constexpr uint32_t fabric_x = get_compile_time_arg_val(11);
constexpr uint32_t fabric_y = get_compile_time_arg_val(12);
constexpr uint32_t gathered_semaphore = get_compile_time_arg_val(13);
constexpr auto input_args = TensorAccessorArgs<14>();
constexpr auto start_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
constexpr auto end_args = TensorAccessorArgs<start_args.next_compile_time_args_offset()>();

constexpr uint32_t packed_rows = kda_chronology::selection::packed_history_rows;
// A BF16 tile holds four 16x16 faces; one tile row is a 32-byte face row in each of two faces.
constexpr uint32_t face_rows = 16;
constexpr uint32_t face_row_bytes = 16 * sizeof(uint16_t);
constexpr uint32_t face_bytes = face_rows * face_row_bytes;
constexpr uint32_t tile_row_bytes = 2 * face_row_bytes;
// DRAM reads start on this alignment, so each face row is fetched within its aligned span.
constexpr uint32_t span_bytes = 64;

// Gather this worker's column tiles of the outgoing and local final history rows, derived from the chronology,
// into the fabric worker's staged rows, then announce them there.
void kernel_main() {
    const uint32_t input_address = get_arg_val<uint32_t>(0);
    const uint32_t start_address = get_arg_val<uint32_t>(1);
    const uint32_t end_address = get_arg_val<uint32_t>(2);
    const uint32_t first_tile = get_arg_val<uint32_t>(3);

    const uint32_t scalar = get_write_ptr(scalar_cb);
    auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scalar);
    const auto start = TensorAccessor(start_args, start_address);
    noc_async_read(start.get_noc_addr(0), scalar, sizeof(uint32_t));
    noc_async_read_barrier();
    const uint32_t actual_start = words[0];
    kda_chronology::Topology topology{};
    if constexpr (has_actual_end) {
        const auto end = TensorAccessor(end_args, end_address);
        noc_async_read(end.get_noc_addr(0), scalar, sizeof(uint32_t));
        noc_async_read_barrier();
        topology = kda_chronology::derive_interval(actual_start, words[0], sp_rank, sp_size, local_rows);
    } else {
        topology = kda_chronology::derive(actual_start, sp_rank, sp_size, local_rows);
    }
    uint32_t selected[packed_rows];
    kda_chronology::selection_history_rows(
        topology, kda_chronology::selection::outgoing_and_local_final_history, sp_size, local_rows, selected);

    const auto input = TensorAccessor(input_args, input_address);
    const uint32_t rows = get_write_ptr(rows_cb);
    const uint32_t spans = get_write_ptr(spans_cb);
    const uint32_t end_tile = first_tile + tiles_per_core < width_tiles ? first_tile + tiles_per_core : width_tiles;
    const uint32_t tiles = end_tile - first_tile;
    // Fetch every face row's aligned span, then pick the face rows out of them.
    for (uint32_t index = 0; index < packed_rows; ++index) {
        const uint32_t row = selected[index];
        const uint32_t tile_row = row / tt::constants::TILE_HEIGHT;
        const uint32_t row_in_tile = row % tt::constants::TILE_HEIGHT;
        for (uint32_t tile = 0; tile < tiles; ++tile) {
            for (uint32_t column = 0; column < 2; ++column) {
                const uint32_t offset =
                    (row_in_tile / face_rows * 2 + column) * face_bytes + row_in_tile % face_rows * face_row_bytes;
                noc_async_read(
                    input.get_noc_addr(tile_row * input_row_tiles + first_tile + tile, offset & ~(span_bytes - 1)),
                    spans + ((index * tiles_per_core + tile) * 2 + column) * span_bytes,
                    span_bytes);
            }
        }
    }
    noc_async_read_barrier();
    for (uint32_t index = 0; index < packed_rows; ++index) {
        const uint32_t offset = selected[index] % tt::constants::TILE_HEIGHT % face_rows * face_row_bytes;
        for (uint32_t tile = 0; tile < tiles; ++tile) {
            for (uint32_t column = 0; column < 2; ++column) {
                noc_async_read(
                    get_noc_addr(
                        spans + ((index * tiles_per_core + tile) * 2 + column) * span_bytes +
                        (offset & (span_bytes - 1))),
                    rows + index * row_bytes + (first_tile + tile) * tile_row_bytes + column * face_row_bytes,
                    face_row_bytes);
            }
        }
    }
    noc_async_read_barrier();
    for (uint32_t index = 0; index < packed_rows; ++index) {
        const uint32_t columns = rows + index * row_bytes + first_tile * tile_row_bytes;
        noc_async_write(columns, get_noc_addr(fabric_x, fabric_y, columns), tiles * tile_row_bytes);
    }
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(fabric_x, fabric_y, get_semaphore(gathered_semaphore)), 1);
    noc_async_atomic_barrier();
}
