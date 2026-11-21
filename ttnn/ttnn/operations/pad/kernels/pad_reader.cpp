// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// pad reader (explicit sync on both rings; the host lists both in disable_dfb_implicit_sync_for, and every
// NoC transfer of this kernel carries the user transaction id trid 0).
//
// Block operations (op_design.md, Block schedule):
//   build_pad_row_template      iff the core has pad rows: W_o x value in one dfb_pad_row entry per writer
//                               thread (this thread builds copies p, p+R, ...), pushed before any data read.
//   load_data_block             reserve block_rows entries of dfb_out_rows; one tagged read per input row,
//                               landing at byte offset head_bytes of its entry; no barrier inside the block.
//   fill_row_margins_first_lap  while the block is on the ring's first lap: fill the head [0, bw) and tail
//                               [bw+W, W_o) of each entry (bytes disjoint from the reads in flight). Later laps
//                               reuse the slot's margins: reads never touch them and the writer only reads.
//   publish_data_block          one tagged read barrier, push the block's rows.
//
// Thread p of R takes this core's data rows p, p+R, ..., block_rows of them per reserve/push (one op). On the
// STRIDED ring an op's entries sit stride_size apart in one tile counter's slice, and a thread rotates its
// tile counters after each push: with W = m*R writer threads, op k of reader p goes to writer p + (k mod m)*R
// (pad_writer.cpp follows the same dealing); with R >= W every op of reader p goes to writer p mod W.
//
//   fill_tile_padding_of_block  tile layout only: a page is a tile, copied whole (head = tail = 0); after the
//                               barrier, `value` over the input's tile padding in the block's last-row /
//                               last-column tiles (those places are added elements of the output).
//
// Helper use: the template and margins are filled with dataflow_kernel_lib::fill_l1_range (element-granular
// ranges, element-wide stores at unaligned edges). The reads are raw NoC calls: the kernel library has no
// row-assembly dataflow helper.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"

// fill_tile_padding (tile layout): `value` over the places of one tile at or past hv valid rows or wv valid
// columns. The tile is a grid of faces (face_h x face_w elements, row-major inside a face, faces row-major);
// per face: the rows at or past hv are one contiguous range, the columns at or past wv one range per valid row.
template <uint32_t elem_bytes, uint32_t tile_h, uint32_t tile_w, uint32_t face_h, uint32_t face_w>
FORCE_INLINE void fill_tile_padding(uint32_t tile, uint32_t hv, uint32_t wv, uint32_t pad_bits) {
    constexpr uint32_t row_bytes = face_w * elem_bytes;
    constexpr uint32_t face_bytes = face_h * row_bytes;
    uint32_t face = tile;
    for (uint32_t fr = 0; fr < tile_h; fr += face_h) {
        const uint32_t rv = hv <= fr ? 0u : (hv - fr < face_h ? hv - fr : face_h);  // valid rows of this face
        for (uint32_t fc = 0; fc < tile_w; fc += face_w, face += face_bytes) {
            const uint32_t cv = wv <= fc ? 0u : (wv - fc < face_w ? wv - fc : face_w);  // valid columns
            if (rv < face_h) {
                dataflow_kernel_lib::fill_l1_range<elem_bytes>(face + rv * row_bytes, (face_h - rv) * row_bytes, pad_bits);
            }
            if (cv == 0) {
                if (rv) {
                    dataflow_kernel_lib::fill_l1_range<elem_bytes>(face, rv * row_bytes, pad_bits);
                }
            } else if (cv < face_w) {
                for (uint32_t r = 0; r < rv; ++r) {
                    dataflow_kernel_lib::fill_l1_range<elem_bytes>(
                        face + r * row_bytes + cv * elem_bytes, (face_w - cv) * elem_bytes, pad_bits);
                }
            }
        }
    }
}

void kernel_main() {
    constexpr uint32_t R = get_arg(args::reader_threads);
    constexpr uint32_t W = get_arg(args::writer_threads);
    constexpr uint32_t ops_per_tc_round = W > R ? W / R : 1;  // this thread's tile counters on dfb_out_rows
    constexpr uint32_t elem_bytes = get_arg(args::elem_bytes);
    constexpr bool tile_layout = get_arg(args::tile_layout) != 0;
    constexpr uint32_t tile_h = get_arg(args::tile_h);
    constexpr uint32_t tile_w = get_arg(args::tile_w);
    constexpr uint32_t face_h = get_arg(args::face_h);
    constexpr uint32_t face_w = get_arg(args::face_w);

    const uint32_t data_start = get_arg(args::data_start);
    const uint32_t data_count = get_arg(args::data_count);
    const uint32_t pad_count = get_arg(args::pad_count);
    const uint32_t block_rows = get_arg(args::block_rows);
    const uint32_t ring_blocks = get_arg(args::ring_blocks);
    const uint32_t in_row_bytes = get_arg(args::in_row_bytes);
    const uint32_t out_row_bytes = get_arg(args::out_row_bytes);
    const uint32_t head_bytes = get_arg(args::head_bytes);
    const uint32_t tail_bytes = get_arg(args::tail_bytes);
    const uint32_t pad_bits = get_arg(args::pad_bits);
    // Tile layout: the input page grid's inner dims (tile rows Hp, tile columns Tp) and the valid rows / columns
    // of its last tile row / column (== tile_h / tile_w when aligned: nothing to fill).
    const uint32_t Hp = get_arg(args::Hp);
    const uint32_t Tp = get_arg(args::Tp);
    const uint32_t h_last = get_arg(args::h_last);
    const uint32_t w_last = get_arg(args::w_last);

    const uint32_t p = get_my_thread_id();
    constexpr NocOptVals user_txn = {.trid = 0};

    DataflowBuffer dfb_out_rows(dfb::out_rows);
    DataflowBuffer dfb_pad_row(dfb::pad_row);
    Noc noc;
    const auto acc_in = TensorAccessor(tensor::input);

    // build_pad_row_template: one copy per writer thread, copy k pushed by reader k mod R (its pushes rotate
    // to writers k, k+R, ...). Gated on pad_count exactly as the writer's wait/pop.
    if (pad_count > 0) {
        for (uint32_t k = p; k < W; k += R) {
            dfb_pad_row.reserve_back(1);
            dataflow_kernel_lib::fill_l1_range<elem_bytes>(dfb_pad_row.get_write_ptr(), out_row_bytes, pad_bits);
            dfb_pad_row.push_back(1);
        }
    }

    const uint32_t my_rows = data_count / R + (p < data_count % R ? 1u : 0u);
    const uint32_t tail_offset = head_bytes + in_row_bytes;
    const uint32_t stride_bytes = dfb_out_rows.get_stride_size();  // entry spacing within one op
    uint32_t row = p;  // core-local data row of the next read
    for (uint32_t done = 0, block_idx = 0; done < my_rows; done += block_rows, ++block_idx) {
        const uint32_t rows_this_block = my_rows - done < block_rows ? my_rows - done : block_rows;

        // load_data_block
        const uint32_t block_first_row = row;
        dfb_out_rows.reserve_back(block_rows);
        for (uint32_t j = 0; j < rows_this_block; ++j, row += R) {
            noc.async_read<NocOptions::TXN_ID>(
                acc_in,
                dfb_out_rows,
                in_row_bytes,
                {.page_id = data_start + row},
                {.offset_bytes = j * stride_bytes + head_bytes},
                user_txn);
        }

        // fill_row_margins_first_lap: in the shadow of the reads in flight.
        // First lap of the tile counter this op lands on: op k is that counter's (k / ops_per_tc_round)-th.
        if (block_idx < ring_blocks * ops_per_tc_round && (head_bytes | tail_bytes) != 0) {
            const uint32_t base = dfb_out_rows.get_write_ptr();
            for (uint32_t j = 0; j < rows_this_block; ++j) {
                const uint32_t entry = base + j * stride_bytes;
                if (head_bytes) {
                    dataflow_kernel_lib::fill_l1_range<elem_bytes>(entry, head_bytes, pad_bits);
                }
                if (tail_bytes) {
                    dataflow_kernel_lib::fill_l1_range<elem_bytes>(entry + tail_offset, tail_bytes, pad_bits);
                }
            }
        }

        // publish_data_block: every push but the core's last is a whole block, so block starts stay aligned
        // to the ring and a block never wraps inside one reservation.
        noc.async_read_barrier<NocOptions::TXN_ID>(user_txn);

        // fill_tile_padding_of_block (tile layout): a read copies the whole input tile, its tile padding
        // included, so the tiles of the last tile row / column are fixed up after the barrier, every lap.
        if constexpr (tile_layout) {
            if (h_last < tile_h || w_last < tile_w) {
                const uint32_t base = dfb_out_rows.get_write_ptr();
                for (uint32_t j = 0, g = data_start + block_first_row; j < rows_this_block; ++j, g += R) {
                    const uint32_t t = g % Tp;
                    const uint32_t hv = (g / Tp) % Hp == Hp - 1 ? h_last : tile_h;
                    const uint32_t wv = t == Tp - 1 ? w_last : tile_w;
                    if (hv < tile_h || wv < tile_w) {
                        fill_tile_padding<elem_bytes, tile_h, tile_w, face_h, face_w>(
                            base + j * stride_bytes, hv, wv, pad_bits);
                    }
                }
            }
        }
        dfb_out_rows.push_back(rows_this_block);
    }
    // Every transfer carried the user id, so the hart's tag register is already handed back.
}
