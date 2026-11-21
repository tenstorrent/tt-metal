// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute kernel of linear: one NEO of C. It sees its own BLOCKED blocks of the weight and bias rings, every
// entry of the ALL activation ring, and packs into its slice of the STRIDED output ring, so it reads like a
// single-threaded kernel over its own block of each round.
//
// Per round (one [block_h x block_w] output block, its accumulator held in DEST for the whole of K):
//   preload_bias_block  DEST[r * block_w + c] = bias tile c, row 0 broadcast down 32 rows   (has a bias only)
//   matmul_chunk        DEST += act chunk [block_h x block_k] x weight block [block_k x block_w], per K chunk
//   pack_block          DEST -> cb_output_tiles, block_h * block_w tiles, once
//
// Raw compute API instead of the kernel library, as op_design.md (Helpers considered and rejected) records:
//   - the matmul is written with raw compute API calls, not with matmul_block() (matmul_block_helpers.hpp).
//   - so is the bias preload, not with add_bias_bcast_rows() (bias_add_helpers.hpp).
//   - the eltwise chain acquires DEST itself (chain.inl:3366-3389), so it cannot add a row-broadcast bias to,
//     or pack, an accumulator a raw matmul left in DEST; folding the bias into DEST first removes the product
//     ring a chain would need.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_rounds = get_arg(args::num_rounds);
    const uint32_t k_tiles = get_arg(args::k_tiles);
    constexpr uint32_t block_h = get_arg(args::block_h);
    constexpr uint32_t block_w = get_arg(args::block_w);
    constexpr uint32_t block_k = get_arg(args::block_k);
    constexpr uint32_t block_tiles = block_h * block_w;
    constexpr uint32_t act_chunk_tiles = block_h * block_k;
    constexpr uint32_t weight_block_tiles = block_k * block_w;
    const uint32_t num_k_chunks = k_tiles / block_k;

    DataflowBuffer act(dfb::cb_act_tiles);
    DataflowBuffer weight(dfb::cb_weight_tiles);
    DataflowBuffer out(dfb::cb_output_tiles);

    // Matmul maps in0 (activation) to SrcB and in1 (weight) to SrcA. The output ring never changes, so the
    // pack target set here holds for the whole kernel.
    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::cb_act_tiles, dfb::cb_weight_tiles, dfb::cb_output_tiles);

#ifdef LINEAR_HAS_BIAS
    DataflowBuffer bias(dfb::cb_bias_tiles);
#else
    // Without a bias the matmul configuration is never disturbed: once per kernel.
    matmul_block_init(dfb::cb_act_tiles, dfb::cb_weight_tiles, 0, block_w, block_h, block_k);
#endif

    for (uint32_t round = 0; round < num_rounds; ++round) {
        tile_regs_acquire();
#ifdef LINEAR_HAS_BIAS
        // preload_bias_block: an exact copy (no multiply), so independent of the fidelity.
        unary_bcast_init<BroadcastType::ROW>(dfb::cb_bias_tiles);
        bias.wait_front(block_w);
        for (uint32_t r = 0; r < block_h; ++r) {
            for (uint32_t c = 0; c < block_w; ++c) {
                unary_bcast<BroadcastType::ROW>(dfb::cb_bias_tiles, c, r * block_w + c);
            }
        }
        bias.pop_front(block_w);
        // The preload reprogrammed the unpacker and the math engine: back to the matmul MOP.
        matmul_block_init(dfb::cb_act_tiles, dfb::cb_weight_tiles, 0, block_w, block_h, block_k);
#endif
        // matmul_chunk, per K chunk, in K order (the floating-point order is fixed).
        for (uint32_t kc = 0; kc < num_k_chunks; ++kc) {
            act.wait_front(act_chunk_tiles);
            weight.wait_front(weight_block_tiles);
            for (uint32_t k = 0; k < block_k; ++k) {
                // in0 block row-major with block_k tiles per row; in1 block row-major with block_w per row.
                matmul_block(
                    dfb::cb_act_tiles, dfb::cb_weight_tiles, k, k * block_w, 0, 0, block_w, block_h, block_k);
            }
            act.pop_front(act_chunk_tiles);
            weight.pop_front(weight_block_tiles);
        }
        tile_regs_commit();

        // pack_block: the accumulator straight into the destination ring.
        out.reserve_back(block_tiles);
        tile_regs_wait();
        pack_block(0, dfb::cb_output_tiles, block_tiles);
        tile_regs_release();
        out.push_back(block_tiles);
    }
}
