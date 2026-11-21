// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Activation reader of linear (load_act_chunk): one hart fills the ALL ring `cb_act_tiles`, which every NEO of
// the cluster sees whole. Per round and per K chunk it delivers the [block_h x block_k] activation chunk of the
// round's row block, row-major, once for all C NEOs (they all multiply the same rows).
//
// The activation may be interleaved in DRAM or in L1, or width-sharded in L1 on any grid: the accessor resolves
// each page to wherever it lives, so the same code serves every placement (spec GUIDANCE, rules R5 and R14).
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
    constexpr uint32_t block_h = get_arg(args::block_h);
    constexpr uint32_t block_k = get_arg(args::block_k);
    constexpr uint32_t a_k_tiles = get_arg(args::a_k_tiles);  // tiles per activation tile-row (page stride)
    constexpr uint32_t rounds_per_row_block = get_arg(args::rounds_per_row_block);
    const uint32_t num_k_chunks = k_tiles / block_k;

    DataflowBuffer act(dfb::cb_act_tiles);
    Noc noc;
    const auto acc_a = TensorAccessor(tensor::a);

    for (uint32_t i = 0; i < num_rounds; ++i) {
        const uint32_t first_row = ((round_start + i) / rounds_per_row_block) * block_h;
        for (uint32_t kc = 0; kc < num_k_chunks; ++kc) {
            const uint32_t first_k = k_tile_start + kc * block_k;
            for (uint32_t r = 0; r < block_h; ++r) {
                const uint32_t row_page = (first_row + r) * a_k_tiles + first_k;
                for (uint32_t k = 0; k < block_k; ++k) {
                    noc.async_read<NocOptions::TXN_ID>(acc_a, act, {.page_id = row_page + k}, {});
                }
            }
        }
    }
    // Hand the hart back with the user id in its tag register: the register is sticky (target page, ring protocols).
    noc_async_read_set_trid(0, noc.get_noc_id());
}
