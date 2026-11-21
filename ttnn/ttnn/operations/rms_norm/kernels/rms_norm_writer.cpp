// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm writer: hart w of W drains NEOs w, w+W, ... (C/W of them) through the STRIDED `output_tiles` ring,
// one tile each in strict rotation: the mirror of the reader's input dealing. A unit a NEO recomputed in the
// last round (or a row past the tensor's end in a ragged unit) is stored again to the same pages, same bytes.
//
// Implicit sync: one tagged write per tile, page row*Wt + w; the last statement hands the tag register back.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_units = get_arg(args::num_units);
    const uint32_t unit_start = get_arg(args::unit_start);
    const uint32_t total_rows = get_arg(args::total_rows);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t rows_per_block = get_arg(args::rows_per_block);
    constexpr uint32_t W = get_arg(args::writer_threads);
    constexpr uint32_t C = get_arg(args::compute_threads);
    constexpr uint32_t group = C / W;
    static_assert(C % W == 0, "W must divide C");

    const uint32_t w_id = get_my_thread_id();
    const uint32_t rounds = (num_units + C - 1) / C;

    Noc noc;
    DataflowBuffer output_tiles(dfb::output_tiles);
    const auto acc_output = TensorAccessor(tensor::output);

    auto row_of = [&](uint32_t neo, uint32_t k, uint32_t r) __attribute__((always_inline)) {
        const uint32_t local = k * C + neo;
        const uint32_t first_row = (unit_start + (local < num_units ? local : k * C)) * rows_per_block;
        const uint32_t row = first_row + r;
        return row < total_rows ? row : first_row;
    };

    for (uint32_t k = 0; k < rounds; ++k) {  // store_row_block, one unit per NEO per round
        for (uint32_t r = 0; r < rows_per_block; ++r) {
            for (uint32_t w = 0; w < Wt; ++w) {
                for (uint32_t j = 0; j < group; ++j) {
                    noc.async_write<NocOptions::TXN_ID>(
                        output_tiles, acc_output, {}, {.page_id = row_of(w_id + j * W, k, r) * Wt + w});
                }
            }
        }
    }
    noc_async_write_set_trid(0, noc.get_noc_id());
}
