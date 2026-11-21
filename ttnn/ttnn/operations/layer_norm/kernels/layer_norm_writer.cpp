// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm writer. Hart w of W drains NEOs w, w+W, ... (C/W of them) through the STRIDED output_tiles ring,
// one tile each in strict rotation: the mirror of the reader's input dealing. A row a NEO recomputed in the
// last round is stored again to the same pages with the same bytes. Implicit sync: one tagged write per
// entry; the write tag is handed back at the end.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
#include "layer_norm_rows.hpp"

void kernel_main() {
    const uint32_t num_rows = get_arg(args::num_rows);
    const uint32_t row_start = get_arg(args::row_start);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t W = get_arg(args::writer_threads);
    constexpr uint32_t C = get_arg(args::compute_threads);
    constexpr uint32_t group = C / W;  // NEOs this hart drains

    const uint32_t w_id = get_my_thread_id();
    const uint32_t rounds = layer_norm_rows::rounds(num_rows, C);

    DataflowBuffer output_tiles(dfb::output_tiles);
    Noc noc;
    const auto acc_out = TensorAccessor(tensor::out);

    for (uint32_t round = 0; round < rounds; ++round) {
        for (uint32_t w = 0; w < Wt; ++w) {
            for (uint32_t j = 0; j < group; ++j) {
                noc.async_write<NocOptions::TXN_ID>(
                    output_tiles, acc_out, {}, {.page_id = layer_norm_rows::row_of(row_start, num_rows, C, w_id + j * W, round) * Wt + w});
            }
        }
    }
    noc_async_write_set_trid(0, noc.get_noc_id());
}
