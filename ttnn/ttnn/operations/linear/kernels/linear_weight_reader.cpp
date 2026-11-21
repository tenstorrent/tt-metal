// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Weight reader of linear (load_round_bias, load_weight_chunk): one hart fills the rings the NEOs take in
// BLOCKED blocks, one block per NEO, in NEO order.
//
//   cb_bias_tiles    per round, before its first weight chunk: NEO t's block_w bias tiles (has a bias only)
//   cb_weight_tiles  per round and K chunk: NEO t's [block_k x block_w] weight block, row-major
//
// NEO t's column block in round `round` is colblock(t) = col_round * C + t, or NEO 0's when that is past the
// last column block (a padded slot: the NEO recomputes NEO 0's block and the writer stores it again).
//
// Implicit sync: one tagged read per ring entry; the DM0 interrupt handler posts the credits.

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
    const uint32_t k_tile_start = get_arg(args::k_tile_start);
    const uint32_t k_tiles = get_arg(args::k_tiles);
    constexpr uint32_t block_w = get_arg(args::block_w);
    constexpr uint32_t block_k = get_arg(args::block_k);
    constexpr uint32_t n_tiles = get_arg(args::n_tiles);  // tiles per weight tile-row (page stride)
    constexpr uint32_t col_blocks = get_arg(args::col_blocks);
    constexpr uint32_t rounds_per_row_block = get_arg(args::rounds_per_row_block);
    constexpr uint32_t C = get_arg(args::compute_threads);
    const uint32_t num_k_chunks = k_tiles / block_k;

    DataflowBuffer weight(dfb::cb_weight_tiles);
    Noc noc;
    const auto acc_b = TensorAccessor(tensor::b);
#ifdef LINEAR_HAS_BIAS
    DataflowBuffer bias(dfb::cb_bias_tiles);
    const auto acc_bias = TensorAccessor(tensor::bias);
#endif

    for (uint32_t i = 0; i < num_rounds; ++i) {
        const uint32_t first_col_block = ((round_start + i) % rounds_per_row_block) * C;
        auto first_col = [&](uint32_t neo) __attribute__((always_inline)) {
            const uint32_t col_block = first_col_block + neo;
            return (col_block < col_blocks ? col_block : first_col_block) * block_w;
        };
#ifdef LINEAR_HAS_BIAS
        // [N] and [1, ..., 1, N] are the same tiles: one tile-row, page n holds columns 32n..32n+31 in row 0.
        for (uint32_t t = 0; t < C; ++t) {
            const uint32_t col = first_col(t);
            for (uint32_t c = 0; c < block_w; ++c) {
                noc.async_read<NocOptions::TXN_ID>(acc_bias, bias, {.page_id = col + c}, {});
            }
        }
#endif
        for (uint32_t kc = 0; kc < num_k_chunks; ++kc) {
            const uint32_t first_k = k_tile_start + kc * block_k;
            for (uint32_t t = 0; t < C; ++t) {
                const uint32_t col = first_col(t);
                for (uint32_t k = 0; k < block_k; ++k) {
                    const uint32_t row_page = (first_k + k) * n_tiles + col;
                    for (uint32_t c = 0; c < block_w; ++c) {
                        noc.async_read<NocOptions::TXN_ID>(acc_b, weight, {.page_id = row_page + c}, {});
                    }
                }
            }
        }
    }
    // Hand the hart back with the user id in its tag register (see linear_reader.cpp).
    noc_async_read_set_trid(0, noc.get_noc_id());
}
