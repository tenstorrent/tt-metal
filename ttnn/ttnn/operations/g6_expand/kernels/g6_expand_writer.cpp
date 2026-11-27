// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_expand writer (BRISC, NoC1) — regime R1 of op_design.md.
//
// This core owns output tiles [start_t, start_t + n_t) in column-major order
// (col = t / Ht, ht = t % Ht). Each column segment of that range is served by one
// broadcast tile from cb_bcast_tiles: await it, write it whole (2048 B) — or, with half_tile,
// as two 1 KiB writes of the faces-0/1 slot (faces 2/3 == faces 0/1) — to every output
// page (img*Ht + ht)*Wt + wt of the segment with no barrier between writes, flush, pop.
// One write barrier before return (BB-03).
// replicate_split (CT): this RISC also fetches row 0 of face 1 of every column (NoC1) into the
// slot the reader is filling (single block per core => slot j = CB read ptr + j * slot_bytes),
// builds face 1 (and face 3 when !half_tile) by RISC-V stores, and publishes each column on the
// local counter semaphore `sem_face1_built` before awaiting the reader's face 0 (cb push).

namespace {

template <uint32_t face_row_bytes>
FORCE_INLINE void fill_rows(uint32_t src_row_addr, uint32_t dst_addr, uint32_t num_rows) {
    constexpr uint32_t words = face_row_bytes / sizeof(uint32_t);
    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_row_addr);
    uint32_t row[words];
#pragma GCC unroll 8
    for (uint32_t w = 0; w < words; ++w) {
        row[w] = src[w];
    }
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_addr);
    for (uint32_t r = 0; r < num_rows; ++r) {
#pragma GCC unroll 8
        for (uint32_t w = 0; w < words; ++w) {
            dst[w] = row[w];
        }
        dst += words;
    }
}

}  // namespace

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_bcast_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);
    constexpr uint32_t page_size = get_compile_time_arg_val(3);
    constexpr uint32_t reader_write_eighths = get_compile_time_arg_val(4);
    constexpr uint32_t half_tile = get_compile_time_arg_val(5);  // slot holds faces 0/1 only
    constexpr uint32_t replicate_split = get_compile_time_arg_val(6);
    constexpr uint32_t sem_face1_built_id = get_compile_time_arg_val(7);
    constexpr uint32_t face_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t face_row_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t row_fetch_bytes = get_compile_time_arg_val(10);
    constexpr auto output_args = TensorAccessorArgs<11>();
    constexpr auto input_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr uint32_t half_bytes = page_size / 2;
    constexpr uint32_t slot_bytes = half_tile ? half_bytes : page_size;
    constexpr uint32_t face_rows = face_bytes / face_row_bytes;

    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_t = get_arg_val<uint32_t>(0);
    const uint32_t n_t = get_arg_val<uint32_t>(1);
    if (n_t == 0) {
        return;
    }

    const auto output_accessor = TensorAccessor(output_args, output_addr, page_size);

    const uint32_t end_t = start_t + n_t;
    [[maybe_unused]] const uint32_t first_col = start_t / Ht;
    [[maybe_unused]] uint32_t slot_base = 0;
    [[maybe_unused]] volatile tt_l1_ptr uint32_t* sem_face1_built = nullptr;
    if constexpr (replicate_split) {
        DeviceZoneScopedN("G6E_WR_FETCH");
        const uint32_t input_addr = get_common_arg_val<uint32_t>(1);
        const auto input_accessor = TensorAccessor(input_args, input_addr, page_size);
        sem_face1_built = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_face1_built_id));
        slot_base = get_read_ptr(cb_bcast_tiles);
        const uint32_t last_col = (end_t - 1) / Ht;
        for (uint32_t col = first_col; col <= last_col; ++col) {
            const uint32_t slot = slot_base + (col - first_col) * slot_bytes;
            noc_async_read(input_accessor.get_noc_addr(col, face_bytes), slot + face_bytes, row_fetch_bytes);
        }
        noc_async_read_barrier();
    }
    uint32_t t = start_t;
    while (t < end_t) {
        const uint32_t col = t / Ht;
        const uint32_t col_end = (col + 1) * Ht;
        const uint32_t seg_end = end_t < col_end ? end_t : col_end;
        // The reader writes the last (seg_len * reader_write_eighths) / 8 tiles of the segment.
        const uint32_t my_end = seg_end - ((seg_end - t) * reader_write_eighths) / 8;
        const uint32_t img = col / Wt;
        const uint32_t wt = col - img * Wt;
        uint32_t page = (img * Ht + (t - col * Ht)) * Wt + wt;

        if constexpr (replicate_split) {
            DeviceZoneScopedN("G6E_WR_REPLICATE");
            const uint32_t j = col - first_col;
            const uint32_t slot = slot_base + j * slot_bytes;
            fill_rows<face_row_bytes>(slot + face_bytes, slot + face_bytes + face_row_bytes, face_rows - 1);
            if constexpr (!half_tile) {
                fill_rows<face_row_bytes>(slot + face_bytes, slot + 3 * face_bytes, face_rows);
            }
            *sem_face1_built = j + 1;  // local publish (reader polls it before its write share)
        }
        {
            DeviceZoneScopedN("G6E_WAIT_BCAST");
            cb_wait_front(cb_bcast_tiles, 1);
        }
        {
            DeviceZoneScopedN("G6E_WRITE");
            const uint32_t src = get_read_ptr(cb_bcast_tiles);
            for (; t < my_end; ++t, page += Wt) {
                if constexpr (half_tile) {
                    noc_async_write(src, output_accessor.get_noc_addr(page, 0), half_bytes);
                    noc_async_write(src, output_accessor.get_noc_addr(page, half_bytes), half_bytes);
                } else {
                    noc_async_write(src, output_accessor.get_noc_addr(page), page_size);
                }
            }
            t = seg_end;
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_bcast_tiles, 1);
    }
    {
        DeviceZoneScopedN("G6E_DRAIN");
        noc_async_write_barrier();
    }
}
