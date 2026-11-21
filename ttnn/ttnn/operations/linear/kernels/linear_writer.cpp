// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Writer of linear (store_block): writer hart w of W drains NEOs w, w+W, ... (C/W of them) from the STRIDED ring
// `cb_output_tiles`, one tile each in strict rotation. A NEO's block_h x block_w tiles arrive row-major; tile j
// goes to output page (first_row + j / block_w) * n_tiles + first_col(neo) + j % block_w. A padded NEO's block
// (NEO 0's, recomputed) is stored again to the same pages with the same bytes.
//
// The result may be interleaved in DRAM or width-sharded in L1 on any grid: the accessor resolves each page to
// wherever it lives (rules R7 and R14). Implicit sync: one tagged write per ring entry.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_rounds = get_arg(args::num_rounds);
    const uint32_t round_start = get_arg(args::round_start);
    constexpr uint32_t block_h = get_arg(args::block_h);
    constexpr uint32_t block_w = get_arg(args::block_w);
    constexpr uint32_t n_tiles = get_arg(args::n_tiles);
    constexpr uint32_t col_blocks = get_arg(args::col_blocks);
    constexpr uint32_t rounds_per_row_block = get_arg(args::rounds_per_row_block);
    constexpr uint32_t C = get_arg(args::compute_threads);
    constexpr uint32_t W = get_arg(args::writer_threads);
    constexpr uint32_t group = C / W;
    constexpr uint32_t block_tiles = block_h * block_w;

    const uint32_t w_id = get_my_thread_id();

    DataflowBuffer out(dfb::cb_output_tiles);
    Noc noc;
    const auto acc_out = TensorAccessor(tensor::out);

    for (uint32_t i = 0; i < num_rounds; ++i) {
        const uint32_t round = round_start + i;
        const uint32_t first_row = (round / rounds_per_row_block) * block_h;
        const uint32_t first_col_block = (round % rounds_per_row_block) * C;
        for (uint32_t j = 0; j < block_tiles; ++j) {
            const uint32_t row_page = (first_row + j / block_w) * n_tiles + j % block_w;
            for (uint32_t g = 0; g < group; ++g) {
                const uint32_t col_block = first_col_block + w_id + g * W;
                const uint32_t col = (col_block < col_blocks ? col_block : first_col_block) * block_w;
                noc.async_write<NocOptions::TXN_ID>(out, acc_out, {}, {.page_id = row_page + col});
            }
        }
    }
    // Hand the hart back with the user id in its tag register (see linear_reader.cpp).
    noc_async_write_set_trid(0, noc.get_noc_id());
}
