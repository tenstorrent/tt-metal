// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <utility>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/scratchpad.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/data_movement/concat/device/kernels/dataflow/concat_reader_common.hpp"

// Builds the output tiles of a TILE concat along the rows (dim -2) whose inputs' row counts need not be
// whole tiles, in output page order, into the bound Dataflow Buffer.
//
// Output row r of a block (the dims before -2) is row r - offset_i of input i, where offset_i is the sum of
// the earlier inputs' logical rows; rows past the last input's are padding and written as zeros. A run of
// rows that comes from one source tile is copied from a scratch copy of that tile, face row by face row: in
// a 32x32 tile, row k is the k % 16-th row of faces 2 * (k / 16) and 2 * (k / 16) + 1, each 16 elements.
// Every count that follows the rows is a runtime argument, so one program serves every row count.
//
// Runtime args: start_tile, num_tiles (this core's range of output tiles), out_tile_rows (output tile rows
// per block). Varargs: in_rows[N] (each input's logical rows), then in_tile_rows[N] (each input's tile
// rows per block).
void kernel_main() {
    const uint32_t start_tile = get_arg(args::start_tile);
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t out_tile_rows = get_arg(args::out_tile_rows);
    constexpr uint32_t width_tiles = get_arg(args::width_tiles);
    constexpr uint32_t face_row_bytes = get_arg(args::face_row_bytes);

    constexpr uint32_t num_tensors = std::tuple_size_v<decltype(tensor::inputs)>;
    constexpr auto tensor_indices = std::make_integer_sequence<uint32_t, num_tensors>();
    constexpr uint32_t in_tile_rows_offset = num_tensors;
    constexpr uint32_t tile_height = 32;
    constexpr uint32_t face_height = 16;
    constexpr uint32_t face_bytes = face_height * face_row_bytes;
    constexpr uint32_t face_row_words = face_row_bytes / sizeof(uint32_t);

    uint32_t out_rows = 0;
    for (uint32_t i = 0; i < num_tensors; ++i) {
        out_rows += get_vararg(i);
    }

    DataflowBuffer dfb_out(dfb::out);
    const uint32_t tile_bytes = dfb_out.get_tile_size();
    // Two source tiles, so that within one output tile whose rows come from two consecutive tiles of one
    // input, each is read once. Consecutive output tiles are in other columns, so across output tiles a
    // shifted source tile is read twice.
    Scratchpad<uint8_t> scratch(scratch::src);
    const uint32_t scratch_base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&*scratch.local_mem()));
    uint32_t slot_tensor[2] = {num_tensors, num_tensors};
    uint32_t slot_page[2] = {0, 0};
    uint32_t next_slot = 0;
    Noc noc;

    auto read_tile = [&](uint32_t tensor_idx, uint32_t page, uint32_t dst_addr) {
        concat_reader::with_input(tensor_idx, tensor_indices, [&](auto I) {
            const auto accessor = TensorAccessor(std::get<decltype(I)::value>(tensor::inputs));
            noc.async_read(accessor, CoreLocalMem<uint8_t>(dst_addr), tile_bytes, {.page_id = page}, {});
        });
        noc.async_read_barrier();
    };
    // The L1 address of a scratch copy of the tile, reading it unless a slot already holds it.
    auto source_tile = [&](uint32_t tensor_idx, uint32_t page) -> uint32_t {
        for (uint32_t s = 0; s < 2; ++s) {
            if (slot_tensor[s] == tensor_idx && slot_page[s] == page) {
                return scratch_base + s * tile_bytes;
            }
        }
        const uint32_t s = next_slot;
        next_slot ^= 1;
        slot_tensor[s] = tensor_idx;
        slot_page[s] = page;
        read_tile(tensor_idx, page, scratch_base + s * tile_bytes);
        return scratch_base + s * tile_bytes;
    };
    auto row_offset = [](uint32_t row) {
        return (row / face_height) * 2 * face_bytes + (row % face_height) * face_row_bytes;
    };

    const uint32_t tiles_per_block = out_tile_rows * width_tiles;
    for (uint32_t t = start_tile; t < start_tile + num_tiles; ++t) {
        const uint32_t block = t / tiles_per_block;
        const uint32_t tile_row = (t % tiles_per_block) / width_tiles;
        const uint32_t col = t % width_tiles;

        dfb_out.reserve_back(1);
        const uint32_t out_addr = dfb_out.get_write_ptr();
        uint32_t k = 0;
        while (k < tile_height) {
            const uint32_t row = tile_row * tile_height + k;
            if (row >= out_rows) {
                // Padding rows past the last input's.
                for (; k < tile_height; ++k) {
                    for (uint32_t half = 0; half < 2; ++half) {
                        volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                            out_addr + row_offset(k) + half * face_bytes);
                        for (uint32_t w = 0; w < face_row_words; ++w) {
                            dst[w] = 0;
                        }
                    }
                }
                break;
            }
            uint32_t tensor_idx = 0;
            uint32_t src_row = row;
            while (src_row >= get_vararg(tensor_idx)) {
                src_row -= get_vararg(tensor_idx);
                ++tensor_idx;
            }
            const uint32_t src_in_tile = src_row % tile_height;
            uint32_t run = tile_height - k;
            if (tile_height - src_in_tile < run) {
                run = tile_height - src_in_tile;
            }
            if (get_vararg(tensor_idx) - src_row < run) {
                run = get_vararg(tensor_idx) - src_row;
            }
            const uint32_t page =
                (block * get_vararg(in_tile_rows_offset + tensor_idx) + src_row / tile_height) * width_tiles + col;
            if (k == 0 && src_in_tile == 0 && run == tile_height) {
                // A whole source tile in place: read it straight into the output slot.
                read_tile(tensor_idx, page, out_addr);
                k = tile_height;
                continue;
            }
            const uint32_t src_addr = source_tile(tensor_idx, page);
            for (uint32_t j = 0; j < run; ++j) {
                for (uint32_t half = 0; half < 2; ++half) {
                    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        out_addr + row_offset(k + j) + half * face_bytes);
                    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        src_addr + row_offset(src_in_tile + j) + half * face_bytes);
                    for (uint32_t w = 0; w < face_row_words; ++w) {
                        dst[w] = src[w];
                    }
                }
            }
            k += run;
        }
        dfb_out.push_back(1);
    }
}
