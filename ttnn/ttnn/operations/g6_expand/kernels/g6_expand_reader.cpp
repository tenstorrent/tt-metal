// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_expand reader (NCRISC, NoC0) — regime R1 of op_design.md.
//
// For every input page (column) that this core's column-major output range touches:
//   fetch_row0_block : land row 0 of faces 0 and 1 (row_fetch_bytes each, at offsets 0 and
//                      face_bytes) straight into the column's slot of cb_bcast_tiles; all fetches
//                      of a block are in flight under ONE read barrier.
//                      Knob `row_fetch_requests` (CT): 2 = two row_fetch_bytes reads; 1 = one
//                      read of face_bytes + row_fetch_bytes (face 0 + row 0 of face 1).
//   replicate_row0   : in place, face0 row0 -> face0 rows 1..15 and face1 row0 -> face1 rows 1..15
//                      (bytes only, bit-exact; overwrites any fetched padding rows). Faces 2/3 are
//                      copies of faces 0/1: built in the slot (half_tile == 0) or never built
//                      (half_tile == 1: the slot is 1 KiB and every output page is written as two
//                      1 KiB writes from it). Mechanism knob `cpu_rows` (CT): RISC-V builds rows
//                      [0, min(cpu_rows, 16)) of faces 0/1, local NoC L1->L1 doubling copies grow
//                      them to 16 rows (batched over the block under one barrier per level, or —
//                      knob `noc_fanout` — every chunk copied from the prefix under ONE barrier); faces
//                      2/3 by RISC-V when cpu_rows >= 32, else one local NoC copy.
//   publish          : push one page per column when no NoC copy is involved, else per block.
//   write_share      : (knob reader_write_eighths, CT) the reader also writes the last
//                      (seg_len * reader_write_eighths) / 8 tiles of every column segment on NoC0,
//                      straight from the slot it built (perf lamp "write issue on one RISC"); the
//                      writer derives the same split. Writes are flushed before the slot could be
//                      re-reserved and barriered before return (BB-03).
// No helper applies: the op is pure byte movement (compute-path broadcast would change bits; R5).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

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

void kernel_main() {
    constexpr uint32_t cb_bcast_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t col_block = get_compile_time_arg_val(2);
    constexpr uint32_t page_size = get_compile_time_arg_val(3);
    constexpr uint32_t row_fetch_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t face_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t face_row_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t cpu_rows = get_compile_time_arg_val(7);
    constexpr uint32_t reader_write_eighths = get_compile_time_arg_val(8);
    constexpr uint32_t Wt = get_compile_time_arg_val(9);
    constexpr uint32_t half_tile = get_compile_time_arg_val(10);
    constexpr uint32_t row_fetch_requests = get_compile_time_arg_val(11);
    constexpr uint32_t noc_fanout = get_compile_time_arg_val(12);
    // Knob `replicate_split` (CT): the writer (BRISC) fetches + builds face 1 (and 3) of every slot
    // while this RISC does face 0 (and 2); it publishes each finished column on `sem_face1_built`
    // (a local counter, no NoC). Requires the pure-RISC-V build and a single block per core.
    constexpr uint32_t replicate_split = get_compile_time_arg_val(13);
    constexpr uint32_t sem_face1_built_id = get_compile_time_arg_val(14);
    constexpr auto input_args = TensorAccessorArgs<15>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    constexpr uint32_t face_rows = face_bytes / face_row_bytes;
    // Slot = one CB page: the whole tile, or (half_tile) only faces 0/1 — the tile's two
    // identical 1 KiB halves are then written from the same slot (faces 2/3 == faces 0/1).
    constexpr uint32_t half_bytes = 2 * face_bytes;
    constexpr uint32_t slot_bytes = half_tile ? half_bytes : page_size;
    // Rows [1, cpu_face_rows) of faces 0/1 are RISC-V stores; local NoC doubling grows them to
    // face_rows. Faces 2/3 (only when !half_tile): RISC-V when cpu_rows >= 2*face_rows, else one
    // local NoC copy. No NoC copy at all => publish per column (finer publish).
    constexpr uint32_t cpu_face_rows = cpu_rows < face_rows ? cpu_rows : face_rows;
    constexpr bool cpu_faces23 = cpu_rows >= 2 * face_rows;
    constexpr bool noc_doubling = cpu_face_rows < face_rows;
    constexpr bool noc_faces23 = !half_tile && !cpu_faces23 && !replicate_split;
    constexpr bool per_col_publish = !noc_doubling && !noc_faces23;
    static_assert(cpu_rows >= 1 && (cpu_faces23 || (cpu_rows & (cpu_rows - 1)) == 0));
    static_assert(row_fetch_requests == 1 || row_fetch_requests == 2);
    static_assert(!replicate_split || !noc_doubling, "replicate_split needs cpu_rows >= face_rows");
    [[maybe_unused]] volatile tt_l1_ptr uint32_t* sem_face1_built =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_face1_built_id));

    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    [[maybe_unused]] const uint32_t output_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t start_t = get_arg_val<uint32_t>(0);
    const uint32_t n_t = get_arg_val<uint32_t>(1);
    if (n_t == 0) {
        return;
    }

    const auto input_accessor = TensorAccessor(input_args, input_addr, page_size);
    [[maybe_unused]] const auto output_accessor = TensorAccessor(output_args, output_addr, page_size);
    const uint32_t end_t = start_t + n_t;

    const uint32_t first_col = start_t / Ht;
    const uint32_t last_col = (start_t + n_t - 1) / Ht;
    const uint32_t cols_this_core = last_col - first_col + 1;

    for (uint32_t c0 = 0; c0 < cols_this_core; c0 += col_block) {
        const uint32_t block_cols = (cols_this_core - c0) < col_block ? (cols_this_core - c0) : col_block;
        {
            DeviceZoneScopedN("G6E_RESERVE");  // blocking wait: own zone (BB-21)
            cb_reserve_back(cb_bcast_tiles, block_cols);
        }
        const uint32_t block_base = get_write_ptr(cb_bcast_tiles);
        {
            DeviceZoneScopedN("G6E_FETCH");
            for (uint32_t j = 0; j < block_cols; ++j) {
                const uint32_t col = first_col + c0 + j;
                const uint32_t slot = block_base + j * slot_bytes;
                if constexpr (replicate_split) {
                    // face 1 is the writer's: touch only face 0.
                    noc_async_read(input_accessor.get_noc_addr(col, 0), slot, row_fetch_bytes);
                } else if constexpr (row_fetch_requests == 1) {
                    // One request: face 0 whole + row 0 of face 1 (rows other than row 0 are
                    // overwritten by the replicate, so x's padding never reaches out).
                    noc_async_read(input_accessor.get_noc_addr(col, 0), slot, face_bytes + row_fetch_bytes);
                } else {
                    noc_async_read(input_accessor.get_noc_addr(col, 0), slot, row_fetch_bytes);
                    noc_async_read(input_accessor.get_noc_addr(col, face_bytes), slot + face_bytes, row_fetch_bytes);
                }
            }
            noc_async_read_barrier();
        }
        {
            DeviceZoneScopedN("G6E_REPLICATE");
            for (uint32_t j = 0; j < block_cols; ++j) {
                const uint32_t slot = block_base + j * slot_bytes;
                if constexpr (cpu_face_rows > 1) {
                    fill_rows<face_row_bytes>(slot, slot + face_row_bytes, cpu_face_rows - 1);
                    if constexpr (!replicate_split) {
                        fill_rows<face_row_bytes>(
                            slot + face_bytes, slot + face_bytes + face_row_bytes, cpu_face_rows - 1);
                    }
                }
                if constexpr (!half_tile && (cpu_faces23 || replicate_split)) {
                    fill_rows<face_row_bytes>(slot, slot + 2 * face_bytes, face_rows);
                    if constexpr (!replicate_split) {
                        fill_rows<face_row_bytes>(slot + face_bytes, slot + 3 * face_bytes, face_rows);
                    }
                }
                if constexpr (per_col_publish) {
                    cb_push_back(cb_bcast_tiles, 1);
                }
            }
            if constexpr (noc_doubling && noc_fanout) {
                // Fan-out: every remaining chunk of rows copies the final CPU-built prefix, so all
                // copies of the block are in flight under ONE barrier (no level-by-level waits).
                constexpr uint32_t prefix_bytes = cpu_face_rows * face_row_bytes;
                for (uint32_t j = 0; j < block_cols; ++j) {
                    const uint32_t slot = block_base + j * slot_bytes;
                    const uint64_t src0 = get_noc_addr(slot);
                    const uint64_t src1 = get_noc_addr(slot + face_bytes);
                    for (uint32_t off = prefix_bytes; off < face_bytes; off += prefix_bytes) {
                        noc_async_read(src0, slot + off, prefix_bytes);
                        noc_async_read(src1, slot + face_bytes + off, prefix_bytes);
                    }
                }
                noc_async_read_barrier();
            } else if constexpr (noc_doubling) {
                for (uint32_t rows = cpu_face_rows; rows < face_rows; rows *= 2) {
                    const uint32_t bytes = rows * face_row_bytes;
                    for (uint32_t j = 0; j < block_cols; ++j) {
                        const uint32_t slot = block_base + j * slot_bytes;
                        noc_async_read(get_noc_addr(slot), slot + bytes, bytes);
                        noc_async_read(get_noc_addr(slot + face_bytes), slot + face_bytes + bytes, bytes);
                    }
                    noc_async_read_barrier();
                }
            }
            if constexpr (noc_faces23) {
                for (uint32_t j = 0; j < block_cols; ++j) {
                    const uint32_t slot = block_base + j * slot_bytes;
                    noc_async_read(get_noc_addr(slot), slot + half_bytes, half_bytes);
                }
                noc_async_read_barrier();
            }
        }
        if constexpr (!per_col_publish) {
            cb_push_back(cb_bcast_tiles, block_cols);
        }
        if constexpr (reader_write_eighths > 0) {
            DeviceZoneScopedN("G6E_RD_WRITE");
            for (uint32_t j = 0; j < block_cols; ++j) {
                const uint32_t col = first_col + c0 + j;
                const uint32_t seg_begin = start_t > col * Ht ? start_t : col * Ht;
                const uint32_t seg_end = end_t < (col + 1) * Ht ? end_t : (col + 1) * Ht;
                const uint32_t mine = ((seg_end - seg_begin) * reader_write_eighths) / 8;
                const uint32_t img = col / Wt;
                const uint32_t wt = col - img * Wt;
                const uint32_t slot = block_base + j * slot_bytes;
                uint32_t page = (img * Ht + (seg_end - mine - col * Ht)) * Wt + wt;
                if constexpr (replicate_split) {
                    if (mine > 0) {
                        noc_semaphore_wait_min(sem_face1_built, c0 + j + 1);  // face 1 of this slot built
                    }
                }
                for (uint32_t k = 0; k < mine; ++k, page += Wt) {
                    if constexpr (half_tile) {
                        noc_async_write(slot, output_accessor.get_noc_addr(page, 0), half_bytes);
                        noc_async_write(slot, output_accessor.get_noc_addr(page, half_bytes), half_bytes);
                    } else {
                        noc_async_write(slot, output_accessor.get_noc_addr(page), page_size);
                    }
                }
            }
            noc_async_writes_flushed();  // slot may be re-reserved by the next block
        }
    }
    if constexpr (reader_write_eighths > 0) {
        DeviceZoneScopedN("G6E_RD_DRAIN");
        noc_async_write_barrier();
    }
}
