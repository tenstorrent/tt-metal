// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/kernel_lib/tilize_helpers.hpp"

constexpr uint32_t num_experts = get_arg(args::num_experts);
constexpr uint32_t emb_dim_dfb_tiles = get_arg(args::emb_dim_dfb_tiles);
// dispatch_table_num_pages is only meaningful when USE_DISPATCH_TABLE_SKIP
// is defined; it carries zero from the program factory otherwise.
constexpr uint32_t dispatch_table_num_pages = get_arg(args::dispatch_table_num_pages);

void kernel_main() {
    constexpr uint32_t TOKENS_PER_CHUNK = 32;
    uint32_t token_start_idx = get_arg(args::token_start_idx);
    uint32_t num_chunks = get_arg(args::num_chunks);
    constexpr uint32_t total_token_tiles = TOKENS_PER_CHUNK * emb_dim_dfb_tiles;

    DataflowBuffer dfb_combine_input(dfb::combine_input);
    DataflowBuffer dfb_weights(dfb::weights);
#ifdef USE_DISPATCH_TABLE_SKIP
    DataflowBuffer dfb_dispatch_table(dfb::dispatch_table);
    DataflowBuffer dfb_indices(dfb::indices);
#endif
    DataflowBuffer dfb_rowmajor(dfb::rowmajor);

#ifdef USE_DISPATCH_TABLE_SKIP
    // Wait for writer to finish pre-loading dispatch table (loaded once for all chunks)
    dfb_dispatch_table.wait_front(dispatch_table_num_pages);
#endif

    compute_kernel_hw_startup(dfb::combine_input, dfb::weights, dfb::output);

    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
#ifdef USE_DISPATCH_TABLE_SKIP
        // Wait for writer to load this chunk's indices (one chunk at a time)
        dfb_indices.wait_front(TOKENS_PER_CHUNK);
#endif

        dfb_rowmajor.reserve_back(total_token_tiles);

        // Process one expert at a time: both input (c_0) and weight (c_1) are streamed
        // one expert at a time by reader and writer respectively.
        for (uint32_t i = 0; i < TOKENS_PER_CHUNK; ++i) {
            mul_bcast_scalar_init(dfb::combine_input, dfb::weights);

            // first_active tracks whether we've picked the accumulator-initializing
            // expert yet: the DeepSeek path looks for a locally-mapped expert via
            // the dispatch table; the GPT-OSS path looks for a non-zero routing
            // weight. A single pass skips inactive experts; if none qualified,
            // the last expert is forced through to initialise the accumulator.
            bool first_active = true;
            for (uint32_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
                dfb_combine_input.wait_front(emb_dim_dfb_tiles);
                dfb_weights.wait_front(1);

                bool skip_expert = false;
                bool must_zero_init = false;
#ifdef USE_DISPATCH_TABLE_SKIP
                {
                    // Indices arrive from DRAM as uint16 and stay uint16 in the DFB; read as a zero-extended
                    // uint16 element, so the writer needs no in-place uint16->int32 expansion.
                    uint32_t expert_id = dfb_indices.read_tile_value<uint16_t>(i, expert_idx);
                    // Check dispatch table: -1 (0xFFFFFFFF) means non-local
                    uint32_t chip_id = dfb_dispatch_table.read_tile_value(0, expert_id);
                    bool is_local = (chip_id != 0xFFFFFFFF);

                    // On the last expert, if none were local, we must process it to
                    // initialize the accumulator. Writer guarantees the weight is zero
                    // for this case, so multiply produces zeros.
                    bool is_last = (expert_idx == num_experts - 1);
                    must_zero_init = is_last && first_active;
                    skip_expert = !is_local && !must_zero_init;
                }
#else
                {
                    // Read weight value — if zero, skip (but always process at least one
                    // expert so the accumulator gets initialized with valid data)
                    uint32_t weight_val = dfb_weights.read_tile_value(0, 0);
                    skip_expert = (weight_val == 0) && !first_active;
                }
#endif

                if (skip_expert) {
                    dfb_combine_input.pop_front(emb_dim_dfb_tiles);
                    dfb_weights.pop_front(1);
                    continue;
                }

                if (!first_active) {
                    pack_reconfig_l1_acc(1);  // accumulate
                } else {
                    pack_reconfig_l1_acc(0);  // overwrite
                    first_active = false;
                }

                tile_regs_acquire();

                for (uint32_t j = 0; j < emb_dim_dfb_tiles; j++) {
                    mul_tiles_bcast<BroadcastType::SCALAR>(dfb::combine_input, dfb::weights, j, 0, j);
                }

                tile_regs_commit();
                tile_regs_wait();

                for (uint32_t j = 0; j < emb_dim_dfb_tiles; j++) {
                    pack_tile<true>(j, dfb::rowmajor, i * emb_dim_dfb_tiles + j);
                }

                tile_regs_release();

                dfb_combine_input.pop_front(emb_dim_dfb_tiles);
                dfb_weights.pop_front(1);
            }
            pack_reconfig_l1_acc(0);
        }

#ifdef USE_DISPATCH_TABLE_SKIP
        // Release this chunk's indices so writer can load the next chunk's
        dfb_indices.pop_front(TOKENS_PER_CHUNK);
#endif

        dfb_rowmajor.push_back(total_token_tiles);

        compute_kernel_lib::tilize<total_token_tiles, dfb::rowmajor, dfb::output>(1);
    }
}
