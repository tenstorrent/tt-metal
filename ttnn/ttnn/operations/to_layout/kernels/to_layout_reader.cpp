// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// to_layout reader: one block operation per unit of work (op_design.md, Block schedule).
//
// A unit is a band of block_height_tiles tile-rows x block_width_tiles (bw) tile-columns:
//   unit u -> band = u / n_col_blocks, col_block = u % n_col_blocks.
// Leading dims are folded into images; every image is tile-padded on its own:
//   tile_row -> image = tile_row / Ht, h = tile_row % Ht, valid_rows = min(32, H - 32 h).
//
// regime TILIZE   (load_row_block, explicit sync on staged_input = cb_row_block):
//   one reserve per block; the valid stick chunks of each tile-row are read straight into the ring
//   (src: stick page image*H + 32h + r, offset col_block*chunk_bytes; dst offset r*chunk_bytes), the
//   padding rows of an image's last tile-row are zeroed in place, one barrier, one push. Every NoC
//   transfer carries the user transaction id 0 (target page, Ring protocols).
// regime UNTILIZE (load_tile_block, implicit sync on staged_input = cb_tile_block_in):
//   one tagged read per tile page; the DM0 interrupt handler posts the credits.
//
// No helper: kernel_lib's read_sticks_for_tilize is the Metal 1 CB API with untagged transfers
// (op_design.md, Helpers considered and rejected).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

namespace {
constexpr uint32_t REGIME_TILIZE = 0;
constexpr uint32_t REGIME_UNTILIZE = 1;
constexpr uint32_t TILE_HEIGHT = 32;
}  // namespace

void kernel_main() {
    const uint32_t unit_start = get_arg(args::unit_start);
    const uint32_t num_units = get_arg(args::num_units);

    constexpr uint32_t R = get_arg(args::reader_threads);
    constexpr uint32_t regime = get_arg(args::regime);
    static_assert(regime == REGIME_TILIZE || regime == REGIME_UNTILIZE, "to_layout_reader: unknown regime");
    constexpr uint32_t bw = get_arg(args::block_width_tiles);
    constexpr uint32_t bht = get_arg(args::block_height_tiles);
    constexpr uint32_t n_col_blocks = get_arg(args::n_col_blocks);
    constexpr uint32_t tensor_w_tiles = get_arg(args::tensor_w_tiles);
    constexpr uint32_t total_tile_rows = get_arg(args::total_tile_rows);
    constexpr uint32_t H = get_arg(args::image_height);
    constexpr uint32_t Ht = get_arg(args::image_height_tiles);
    constexpr uint32_t chunk_bytes = get_arg(args::chunk_bytes);
    constexpr uint32_t band_row_bytes = TILE_HEIGHT * chunk_bytes;  // one tile-row of the block = bw entries
    constexpr NocOptVals user_txn = {.trid = 0};

    DataflowBuffer staged(dfb::staged_input);
    Noc noc;
    const auto acc = TensorAccessor(tensor::input);

    // The plan's count, not get_num_threads(): were a thread not to run, its units would be missing.
    for (uint32_t i = get_my_thread_id(); i < num_units; i += R) {
        const uint32_t u = unit_start + i;
        const uint32_t band = u / n_col_blocks;
        const uint32_t col_block = u % n_col_blocks;
        const uint32_t first_tile_row = band * bht;
        const uint32_t rows_left = total_tile_rows - first_tile_row;
        const uint32_t band_tile_rows = rows_left < bht ? rows_left : bht;  // ragged last band

        if constexpr (regime == REGIME_TILIZE) {
            staged.reserve_back(band_tile_rows * bw);
            for (uint32_t tr = 0; tr < band_tile_rows; ++tr) {
                const uint32_t tile_row = first_tile_row + tr;
                const uint32_t image = tile_row / Ht;
                const uint32_t h = tile_row % Ht;
                const uint32_t rows_in_image = H - h * TILE_HEIGHT;
                const uint32_t valid_rows = rows_in_image < TILE_HEIGHT ? rows_in_image : TILE_HEIGHT;
                const uint32_t first_stick = image * H + h * TILE_HEIGHT;
                const uint32_t dst_base = tr * band_row_bytes;
                for (uint32_t r = 0; r < valid_rows; ++r) {
                    noc.async_read<NocOptions::TXN_ID>(
                        acc,
                        staged,
                        chunk_bytes,
                        {.page_id = first_stick + r, .offset_bytes = col_block * chunk_bytes},
                        {.offset_bytes = dst_base + r * chunk_bytes},
                        user_txn);
                }
                if (valid_rows < TILE_HEIGHT) {
                    // Padding rows of the image's last tile-row: zeroed in L1, never read (TILE padding = 0).
                    noc.async_write_zeros(
                        staged,
                        (TILE_HEIGHT - valid_rows) * chunk_bytes,
                        {.offset_bytes = dst_base + valid_rows * chunk_bytes});
                    noc.write_zeros_l1_barrier();
                }
            }
            noc.async_read_barrier<NocOptions::TXN_ID>(user_txn);
            staged.push_back(band_tile_rows * bw);
        } else {
            for (uint32_t tr = 0; tr < band_tile_rows; ++tr) {
                const uint32_t first_page = (first_tile_row + tr) * tensor_w_tiles + col_block * bw;
                for (uint32_t j = 0; j < bw; ++j) {
                    noc.async_read<NocOptions::TXN_ID>(acc, staged, {.page_id = first_page + j}, {});
                }
            }
        }
    }

    // Hand the hart back with the user id in its tag registers: a program that runs next with untagged
    // transfers would otherwise inherit a ring's id (target page, Ring protocols). async_write_zeros
    // reprograms the write command buffer's id, so the write tag is reset too.
    noc_async_read_set_trid(0, noc.get_noc_id());
    noc_async_write_set_trid(0, noc.get_noc_id());
}
