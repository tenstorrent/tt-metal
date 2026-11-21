// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// to_layout writer: one block operation per unit of work (op_design.md, Block schedule). Unit and tile-row
// geometry as in to_layout_reader.cpp.
//
// regime TILIZE   (store_tile_block, implicit sync on staged_output = cb_tile_block):
//   one tagged write per tile page, in the order the compute kernel packs them (tile-row by tile-row,
//   bw tiles each); the DM0 interrupt handler acks the entries.
// regime UNTILIZE (store_row_block, explicit sync on staged_output = cb_row_block_out):
//   one wait per block, the valid stick chunks of each tile-row written out (src offset r*chunk_bytes,
//   dst stick page image*H + 32h + r at offset col_block*chunk_bytes), one barrier, one pop. Padding rows
//   are popped, never written. Every NoC transfer carries the user transaction id 0.
//
// No helper: kernel_lib's write_sticks_after_untilize is the Metal 1 CB API with untagged transfers
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

    constexpr uint32_t W = get_arg(args::writer_threads);
    constexpr uint32_t regime = get_arg(args::regime);
    static_assert(regime == REGIME_TILIZE || regime == REGIME_UNTILIZE, "to_layout_writer: unknown regime");
    constexpr uint32_t bw = get_arg(args::block_width_tiles);
    constexpr uint32_t bht = get_arg(args::block_height_tiles);
    constexpr uint32_t n_col_blocks = get_arg(args::n_col_blocks);
    constexpr uint32_t tensor_w_tiles = get_arg(args::tensor_w_tiles);
    constexpr uint32_t total_tile_rows = get_arg(args::total_tile_rows);
    constexpr uint32_t H = get_arg(args::image_height);
    constexpr uint32_t Ht = get_arg(args::image_height_tiles);
    constexpr uint32_t chunk_bytes = get_arg(args::chunk_bytes);
    constexpr uint32_t band_row_bytes = TILE_HEIGHT * chunk_bytes;
    constexpr NocOptVals user_txn = {.trid = 0};

    DataflowBuffer staged(dfb::staged_output);
    Noc noc;
    const auto acc = TensorAccessor(tensor::output);

    for (uint32_t i = get_my_thread_id(); i < num_units; i += W) {
        const uint32_t u = unit_start + i;
        const uint32_t band = u / n_col_blocks;
        const uint32_t col_block = u % n_col_blocks;
        const uint32_t first_tile_row = band * bht;
        const uint32_t rows_left = total_tile_rows - first_tile_row;
        const uint32_t band_tile_rows = rows_left < bht ? rows_left : bht;

        if constexpr (regime == REGIME_TILIZE) {
            for (uint32_t tr = 0; tr < band_tile_rows; ++tr) {
                const uint32_t first_page = (first_tile_row + tr) * tensor_w_tiles + col_block * bw;
                for (uint32_t j = 0; j < bw; ++j) {
                    noc.async_write<NocOptions::TXN_ID>(staged, acc, {}, {.page_id = first_page + j});
                }
            }
        } else {
            staged.wait_front(band_tile_rows * bw);
            for (uint32_t tr = 0; tr < band_tile_rows; ++tr) {
                const uint32_t tile_row = first_tile_row + tr;
                const uint32_t image = tile_row / Ht;
                const uint32_t h = tile_row % Ht;
                const uint32_t rows_in_image = H - h * TILE_HEIGHT;
                const uint32_t valid_rows = rows_in_image < TILE_HEIGHT ? rows_in_image : TILE_HEIGHT;
                const uint32_t first_stick = image * H + h * TILE_HEIGHT;
                const uint32_t src_base = tr * band_row_bytes;
                for (uint32_t r = 0; r < valid_rows; ++r) {
                    noc.async_write<NocOptions::TXN_ID>(
                        staged,
                        acc,
                        chunk_bytes,
                        {.offset_bytes = src_base + r * chunk_bytes},
                        {.page_id = first_stick + r, .offset_bytes = col_block * chunk_bytes},
                        user_txn);
                }
            }
            noc.async_write_barrier<NocOptions::TXN_ID>(user_txn);
            staged.pop_front(band_tile_rows * bw);
        }
    }

    // Hand the hart back with the user id in its tag register (target page, Ring protocols).
    noc_async_write_set_trid(0, noc.get_noc_id());
}
