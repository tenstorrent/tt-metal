// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t kTileBytes = 32 * 32 * 4;  // fp32
constexpr uint32_t kFaceBytes = 16 * 16 * 4;
constexpr uint32_t kFaceRowBytes = 16 * 4;

void zero_l1(uint32_t addr, uint32_t bytes) {
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    while (bytes > 0) {
        const uint32_t len = bytes < MEM_ZEROS_SIZE ? bytes : MEM_ZEROS_SIZE;
        noc_async_read(zeros, addr, len);
        addr += len;
        bytes -= len;
    }
}

// Byte offsets of row `row` in the left (cols 0-15) and right (cols 16-31) faces of a 32x32 fp32 tile.
uint32_t left_face_row(uint32_t row) { return (row < 16 ? 0 : 2 * kFaceBytes) + (row % 16) * kFaceRowBytes; }
uint32_t right_face_row(uint32_t row) { return left_face_row(row) + kFaceBytes; }

}  // namespace

// Cross-core reduction and output store.
//
// Each of the 3 * Rt partial tiles holds this core's per-row partial sums in row 0. Core c writes
// that row into row c of the same tile on every core, so each core's gather tile ends up with
// one row per source core; the compute kernel column-reduces it. Rows >= num_cores are zeroed
// locally (no other core writes them). Then the untilized output blocks are copied, valid rows
// only, into the output shard.
void kernel_main() {
    constexpr uint32_t cb_partial = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gather = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out_rm = get_compile_time_arg_val(2);
    constexpr uint32_t Rt = get_compile_time_arg_val(3);
    constexpr uint32_t Wt = get_compile_time_arg_val(4);
    constexpr uint32_t rows = get_compile_time_arg_val(5);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t num_cores = get_compile_time_arg_val(7);
    constexpr uint32_t semaphore_id = get_compile_time_arg_val(8);
    constexpr uint32_t out_row_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t num_partials = 3 * Rt;

    const uint32_t c = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t home_x = get_arg_val<uint32_t>(2);
    const uint32_t home_y = get_arg_val<uint32_t>(3);
    const uint32_t home_offset = get_arg_val<uint32_t>(4);

    const uint32_t semaphore_addr = get_semaphore(semaphore_id);
    volatile tt_l1_ptr uint32_t* semaphore = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr);

    // Every core allocates the gather CB at the same L1 address, so the local pointer is also the
    // remote destination.
    cb_reserve_back(cb_gather, num_partials);
    const uint32_t gather_addr = get_write_ptr(cb_gather);
    if constexpr (num_cores < 32) {
        for (uint32_t t = 0; t < num_partials; ++t) {
            const uint32_t tile = gather_addr + t * kTileBytes;
            for (uint32_t row = num_cores; row < 32;) {
                // Rows sharing a face pair are contiguous: zero the rest of this half-tile at once.
                const uint32_t end = row < 16 ? 16 : 32;
                const uint32_t bytes = (end - row) * kFaceRowBytes;
                zero_l1(tile + left_face_row(row), bytes);
                zero_l1(tile + right_face_row(row), bytes);
                row = end;
            }
        }
        noc_async_read_barrier();
    }

    cb_wait_front(cb_partial, num_partials);
    const uint32_t partial_addr = get_read_ptr(cb_partial);
    const uint32_t dst_left = left_face_row(c);
    const uint32_t dst_right = right_face_row(c);
    for (uint32_t i = 0; i < num_cores; ++i) {
        const uint32_t x = get_common_arg_val<uint32_t>(2 * i);
        const uint32_t y = get_common_arg_val<uint32_t>(2 * i + 1);
        for (uint32_t t = 0; t < num_partials; ++t) {
            const uint32_t src = partial_addr + t * kTileBytes;
            const uint32_t dst = gather_addr + t * kTileBytes;
            noc_async_write(src, get_noc_addr(x, y, dst + dst_left), kFaceRowBytes);
            noc_async_write(src + kFaceBytes, get_noc_addr(x, y, dst + dst_right), kFaceRowBytes);
        }
    }
    noc_async_write_barrier();
    for (uint32_t i = 0; i < num_cores; ++i) {
        const uint32_t x = get_common_arg_val<uint32_t>(2 * i);
        const uint32_t y = get_common_arg_val<uint32_t>(2 * i + 1);
        noc_semaphore_inc(get_noc_addr(x, y, semaphore_addr), 1);
    }
    noc_async_atomic_barrier();
    cb_pop_front(cb_partial, num_partials);

    noc_semaphore_wait(semaphore, num_cores);
    noc_semaphore_set(semaphore, 0);
    cb_push_back(cb_gather, num_partials);

    // Valid rows only, into this slice's columns of its home output shard.
    for (uint32_t rt = 0; rt < Rt; ++rt) {
        const uint32_t valid = (rows - rt * 32) < 32 ? (rows - rt * 32) : 32;
        cb_wait_front(cb_out_rm, Wt);
        const uint32_t src = get_read_ptr(cb_out_rm);
        if constexpr (out_row_bytes == row_bytes) {
            noc_async_write(src, get_noc_addr(home_x, home_y, out_addr + rt * 32 * row_bytes), valid * row_bytes);
        } else {
            for (uint32_t r = 0; r < valid; ++r) {
                const uint32_t dst = out_addr + (rt * 32 + r) * out_row_bytes + home_offset;
                noc_async_write(src + r * row_bytes, get_noc_addr(home_x, home_y, dst), row_bytes);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out_rm, Wt);
    }
}
