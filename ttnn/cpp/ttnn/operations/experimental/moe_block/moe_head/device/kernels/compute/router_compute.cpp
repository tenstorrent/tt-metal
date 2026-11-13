// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The router cores' compute kernel: the logits' TP partials summed, the scores (sigmoid, or sqrt(softplus)) + bias, the
// top-K on the SFPU, then the routing rows (router_common.hpp routing_rows; under ROUTER_ROUTE_WORDS also the dispatch
// prep's route words, router_common.hpp route_words).
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/row_block.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/router_common.hpp"

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/topk.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_binary_sfpu.h"
#ifdef ROUTER_SQRTSOFTPLUS
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/eltwise_unary/sqrt.h"
#endif
#include "api/debug/kernel_profiler.hpp"

// Compile-time args (head_router_program.cpp, the router's compute kernel)
constexpr uint32_t expert_tiles = get_compile_time_arg_val(0);  // (score tiles a token column: router candidates / 32)
constexpr uint32_t tp_size = get_compile_time_arg_val(1);       // (the chips summing a logits partial each)
constexpr uint32_t top_k = get_compile_time_arg_val(2);

// Circular buffers (head_program_common.hpp RouterCb)
constexpr uint32_t cb_logits = 2;          // fp32 logits^T, a block per TP position (the owner's: summed in place)
constexpr uint32_t cb_arrivals = 3;        // the other chips' logits partials, a block per arrival slot
constexpr uint32_t cb_arrival_order = 4;   // the arrivals' slots, in arrival order (a uint32 page each)
constexpr uint32_t cb_bias = 5;            // the score bias^T
constexpr uint32_t cb_scores = 6;          // the biased scores' top-K keys
constexpr uint32_t cb_pack_to_unpack = 8;  // a one-page sync: what PACK wrote is in L1 before UNPACK reads it
constexpr uint32_t cb_best = 9;            // the top-K keys of every token column
constexpr uint32_t cb_expert_index = 11;   // the expert ids (uint16), a tile per score tile
constexpr uint32_t cb_ready = 12;          // the reader's page: the partials are in; this chip's index, TP position
// (the top-K keys: the expert id in the score's low mantissa bits - as many as the ids need)
constexpr uint32_t expert_id_bits_mask = routing_rows::id_bits_mask_of(expert_tiles);

#ifdef TRISC_MATH
namespace ckernel::sfpu {
// a score tile's top-K keys: the fp32 score with its low mantissa bits (id_mask) replaced by the expert id (the uint16
// id tile two DEST tiles on): ttnn.topk's fused keys at fp32 - 9 bits instead of bf16, raw INT32 words
template <uint32_t id_mask>
inline void router_topk_key() {
    TOPK_SFPENCC_ALL_LANES_ON();
    sfpi::vConstIntPrgm0 = id_mask;
    for (int i = 0; i < 32; i++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, 128);
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG1, 0);  // the id (the high half of a uint16 in 32-bit DEST: junk)
        TTI_SFPOR(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);
        TTI_SFPXOR(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);  // the score's low bits cleared
        TTI_SFPOR(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
}
}  // namespace ckernel::sfpu
#endif

// This kernel's args (head_router_program.cpp router_compute_rt), read once by kernel_main: num_ops, the ops (owner
// block << 8 | 2: a fin), per owner block the arrivals to add (the fin: tp_size - 1, from the zone's slots 0 ..);
// then the tail: the top-K's done words and the rows' L1. Every tile here is TRANSPOSED: logits^T [expert_tiles][32
// tokens] - a token is a column.
struct RouterComputeArgs {
    const volatile tt_l1_ptr uint32_t* table;  // the args from [0] (the ops and arrivals tables are read in place)
    uint32_t num_ops;                          // [0]
    // the tail, after the arrivals table ([1 + num_ops + tp_size] on)
    uint32_t done_words;         // [+0] the top-K's done words (16 B apart; word 112: the partials are in)
    uint32_t best_page_l1;       // [+1] the cb_best page
    uint32_t rows_l1;            // [+2] the joint routing rows
    uint32_t scale_bits;         // [+3] the route scale's fp32 bits
    uint32_t logits_l1;          // [+4] cb_logits' base (the scores: packed over their logits)
    uint32_t route_words_l1;     // [+5] the route words' L1 (ROUTER_ROUTE_WORDS)
    uint32_t first_token;        // [+6] my row tile's first token (ROUTER_ROUTE_WORDS)
    uint32_t math_sync_address;  // common [0] + ROUTER_MATH_SYNC_OFFSET: the PACK -> MATH word
    uint32_t op(uint32_t index) const { return table[1 + index]; }
    uint32_t arrivals(uint32_t owner_block) const { return table[1 + num_ops + owner_block]; }
};
RouterComputeArgs read_router_compute_args(
    const volatile tt_l1_ptr uint32_t* args, const RouterCommonArgs& common_args) {
    RouterComputeArgs result;
    result.table = args;
    result.num_ops = args[0];
    const volatile tt_l1_ptr uint32_t* tail = args + 1 + result.num_ops + tp_size;
    result.done_words = tail[0];
    result.best_page_l1 = tail[1];
    result.rows_l1 = tail[2];
    result.scale_bits = tail[3];
    result.logits_l1 = tail[4];
    result.route_words_l1 = tail[5];
    result.first_token = tail[6];
    result.math_sync_address = common_args.arena_base + ROUTER_MATH_SYNC_OFFSET;
    return result;
}
#ifdef ROUTER_CHIP_FROM_READY_PAGE
uint32_t router_tp_position = 0;  // (this chip's TP position, from the cb_ready page)
#endif

// One row block of this kernel's work: kernel_main below runs it once per block (row_block.hpp).
void process_row_block(const RouterComputeArgs& args) {
    // (the PACK -> MATH word: zero before anything - L1 scratch, PACK sets it only after the top-K)
    MATH((*reinterpret_cast<volatile uint32_t*>(args.math_sync_address) = 0));
    pack_reconfig_data_format(cb_logits);
#ifdef ROUTER_ITERATIONS_ON_ROUTER
    constexpr uint32_t router_iterations = ROUTER_ITERATIONS_ON_ROUTER;  // (my row tile of each)
#else
    constexpr uint32_t router_iterations = 1;
#endif
    for (uint32_t iteration = 0; iteration < router_iterations; ++iteration) {
        cb_wait_front(cb_ready, 1);  // (the engine's partial logits are in my cb_logits - UNPACK; PACK on word 112)
#ifdef ROUTER_CHIP_FROM_READY_PAGE
        if (iteration == 0) {  // (this chip's index and TP position: words 1, 2 of the ready page - every thread the
                               // same; one program for the mesh, common/chip_program.hpp)
            route_words::my_chip = read_tile_value(cb_ready, 0, 1);
            router_tp_position = read_tile_value(cb_ready, 0, 2);
        }
#endif
#ifdef PADDED_CHUNK
        // (a padded forward: is my row tile of this iteration live - the BRISC's bits, in the ready page: every thread
        // the same; an all-padding tile: every CB and word as ever, no sums, scores, top-K, rows or route words - its
        // route words are zero)
        const bool live = (read_tile_value(cb_ready, 0, 0) >> iteration) & 1;
#else
        constexpr bool live = true;
#endif
        if constexpr (router_iterations > 1 || row_block::BLOCKS > 1) {
            cb_pop_front(cb_ready, 1);
        }
        PACK(({
            volatile uint32_t* partials_word = reinterpret_cast<volatile uint32_t*>(args.done_words + 112);
            while (*partials_word == 0) {
            }
            *partials_word = 0;  // (re-arm)
        }));
        pack_reconfig_l1_acc(0);
        DeviceZoneScopedN("SE-RFIN");
        cb_reserve_back(cb_pack_to_unpack, 1);
        cb_push_back(cb_pack_to_unpack, 1);
        cb_wait_front(cb_pack_to_unpack, 1);  // (UNPACK: the partials are in L1)
        cb_pop_front(cb_pack_to_unpack, 1);
        cb_wait_front(cb_bias, expert_tiles);
        for (uint32_t op = 0; op < args.num_ops; ++op) {
#ifdef ROUTER_CHIP_FROM_READY_PAGE
            const uint32_t owner_block = router_tp_position;  // (my owner block)
#else
            const uint32_t owner_block = args.op(op) >> 8;
#endif
            // (the one fin: the owner's - tp_size - 1 sources)
            const uint32_t arrivals = args.arrivals(owner_block);
            cb_reserve_back(cb_scores, expert_tiles);
            cb_wait_front(cb_expert_index, expert_tiles);
            pack_reconfig_data_format(cb_logits);
            // the sources' partials in cb_arrival_order's order (the slot - the reader's fixed order, so the same sum
            // every run): all but the last summed into my block of cb_logits as they come (fp32 on the SFPU, packed
            // back in place), the last in the passes below - after the last arrival only its add is left
            uint32_t last_slot = 0;
            for (uint32_t arrival = 0; arrival < arrivals; ++arrival) {
                cb_wait_front(cb_arrival_order, 1);
                const uint32_t slot = read_tile_value(cb_arrival_order, 0, 0);
                cb_pop_front(cb_arrival_order, 1);
                if (arrival + 1 == arrivals) {
                    last_slot = slot;
                    break;
                }
                if (!live) {
                    continue;
                }
                for (uint32_t first = 0; first < expert_tiles; first += 2) {
                    const uint32_t tiles = expert_tiles - first < 2 ? expert_tiles - first : 2;
                    tile_regs_acquire();
                    reconfig_data_format_srca(cb_logits);
                    copy_init(cb_logits);
                    for (uint32_t j = 0; j < tiles; ++j) {
                        copy_tile(cb_logits, owner_block * expert_tiles + first + j, j);
                    }
                    reconfig_data_format_srca(cb_logits, cb_arrivals);
                    copy_init(cb_arrivals);
                    for (uint32_t j = 0; j < tiles; ++j) {
                        copy_tile(cb_arrivals, slot * expert_tiles + first + j, 2 + j);
                    }
                    add_binary_tile_init();
                    for (uint32_t j = 0; j < tiles; ++j) {
                        add_binary_tile(j, 2 + j, j);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t j = 0; j < tiles; ++j) {
                        pack_tile<true>(j, cb_logits, owner_block * expert_tiles + first + j);
                    }
                    tile_regs_release();
                }
                // (PACK -> UNPACK: the sums are in cb_logits before the next pass reads them)
                cb_reserve_back(cb_pack_to_unpack, 1);
                cb_push_back(cb_pack_to_unpack, 1);
                cb_wait_front(cb_pack_to_unpack, 1);
                cb_pop_front(cb_pack_to_unpack, 1);
            }
            {
                DeviceZoneScopedN("SE-RFARR");
            }  // (a mark: every arrival in - the fin from here to the top-K)
            // (all in DEST at fp32: the partials, the bias unpacked straight to DEST, the adds on the SFPU - the FPU's
            // srcA / srcB hold TF32, a 10-bit mantissa: the sums and the bias add through them moved scores by up to
            // ~1e-3 and the picks of close experts with them) (a score tile a pass: DEST 0 its score, 1 its key, 2 / 3
            // the operands - the score is kept: packed back over its logits in cb_logits, where the joint rows read
            // the picks' weights exactly; the key's low bits are the id)
            for (uint32_t tile = 0; tile < expert_tiles && live; ++tile) {
                tile_regs_acquire();
                reconfig_data_format_srca(cb_logits);
                copy_init(cb_logits);
                copy_tile(cb_logits, owner_block * expert_tiles + tile, 0);
                if (arrivals) {  // (+ the last source's partial)
                    reconfig_data_format_srca(cb_logits, cb_arrivals);
                    copy_init(cb_arrivals);
                    copy_tile(cb_arrivals, last_slot * expert_tiles + tile, 2);
                    add_binary_tile_init();
                    add_binary_tile(0, 2, 0);
                }
                {  // the owner: scores = sigmoid(logits) (ROUTER_SQRTSOFTPLUS: sqrt(softplus)), biased = scores + bias
                   // (bias^T: per row)
#ifdef ROUTER_SQRTSOFTPLUS
                    softplus_tile_init();  // (INP_FLOAT32: the fp32 polynomial, as stock ttnn.softplus on fp32; beta 1,
                                           // threshold 20)
                    softplus_tile(0, 0x3F800000u, 0x3F800000u, 0x41A00000u);
                    sqrt_tile_init();
                    sqrt_tile(0);
#else
                    sigmoid_tile_init();
                    sigmoid_tile(0);
#endif
                    reconfig_data_format_srca(cb_arrivals, cb_bias);
                    copy_init(cb_bias);
                    copy_tile(cb_bias, tile, 2);
                    add_binary_tile_init();
                    add_binary_tile(0, 2, 1);
                    // -> the top-K's keys: the expert ids into the low mantissa bits (router_topk_key: the id tile two
                    // on, DEST 3)
                    reconfig_data_format_srca(cb_bias, cb_expert_index);
                    copy_init(cb_expert_index);
                    copy_tile(cb_expert_index, tile, 3);
                    MATH((_llk_math_eltwise_unary_sfpu_params_(
                        ckernel::sfpu::router_topk_key<expert_id_bits_mask>, 1, VectorMode::RC_custom)));
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile<true>(1, cb_scores, tile);
                pack_tile<true>(0, cb_logits, owner_block * expert_tiles + tile);
                tile_regs_release();
            }
            cb_push_back(cb_scores, expert_tiles);
            // the owner's 32 tokens' top-K on the SFPU (ttnn.topk's bitonic network), each token a column, fp32
            // straight to DEST (no srcA rounding): the running top K of every column in DEST tile 0, each next 32
            // experts copied into tile 1 as keys (router_topk_key: the expert id in the low mantissa bits - one word a
            // datum, no id tiles riding the swaps); the 64-wide column sorted into K-runs (bitonic phases 0 .. log K -
            // 1), then merged run against run down to one sorted K-run on top (log(64 / K) merge + rebuild rounds) -
            // no full 64-wide sorts, nothing packed until the end
            DeviceZoneScopedN("SE-RTOPK");
            cb_wait_front(cb_scores, expert_tiles);
            // (the network's run: top_k's power of two, at least 4 - top_k 6 sorts the top 8, the rows take its first
            // top_k)
            constexpr uint32_t sort_run = top_k <= 4 ? 4 : top_k <= 8 ? 8 : top_k <= 16 ? 16 : 32;
            constexpr uint32_t log2_sort_run = sort_run == 8 ? 3 : sort_run == 16 ? 4 : sort_run == 32 ? 5 : 2;
            constexpr uint32_t merge_rounds = 6 - log2_sort_run;  // (log2(64 / sort_run) + 1)
            constexpr auto tie_order = ckernel::TopkTieOrder::Unset;
            if (live) {
                tile_regs_acquire();
                if constexpr (expert_tiles == 1) {  // (one tile of candidates: no running top K is parked in DEST
                                                    // tile 2 - the column's other half is -inf keys, below every
                                                    // score; top_k <= 32 never reaches them)
                    fill_tile_init();
                    fill_tile_int<DataFormat::Int32>(0, 0xFF800000u);
                }
                reconfig_data_format_srca(cb_scores);
                copy_init(cb_scores);
                ckernel::topk_tile_init<true, false>();
                for (uint32_t tile = 0; tile < expert_tiles; tile += 2) {
                    // (a last tile alone: against the running top K, a full round)
                    const bool alone = tile + 1 == expert_tiles;
                    // (a pair after the first: its run folded into the running top K)
                    const bool fold = tile > 0 && !alone;
                    if (alone) {
                        if constexpr (expert_tiles > 1) {
                            copy_dest_values<DataFormat::Int32>(2, 0);  // (the running top K, parked at tile - 2)
                        }
                        copy_tile(cb_scores, tile, 1);
                    } else {
                        copy_tile(cb_scores, tile, 0);
                        copy_tile(cb_scores, tile + 1, 1);
                    }
                    ckernel::topk_local_sort<false, DST_ACCUM_MODE, true, false, tie_order>(0, 0, log2_sort_run - 1);
                    for (uint32_t merge = 0; merge + 1 < merge_rounds; ++merge) {
                        ckernel::topk_merge<false, false, DST_ACCUM_MODE, true, false, tie_order>(0, merge, sort_run);
                        ckernel::topk_rebuild<false, DST_ACCUM_MODE, true, false, tie_order>(
                            0, 0, merge, sort_run, log2_sort_run, false);
                    }
                    if (!fold) {  // the top K of the 64 a column on top: tile 0's run, descending
                        ckernel::topk_merge<false, false, DST_ACCUM_MODE, true, false, tie_order>(
                            0, merge_rounds - 1, sort_run);
                        ckernel::topk_rebuild<false, DST_ACCUM_MODE, true, false, tie_order>(
                            0, 0, merge_rounds - 1, sort_run, log2_sort_run, true);
                    } else {  // the pair's top K into tile 1's run, ascending; the running top K (descending) into
                              // tile 0 -> one bitonic merge (max on top) + rebuild of K
                        ckernel::topk_merge<true, false, DST_ACCUM_MODE, true, false, tie_order>(
                            0, merge_rounds - 1, sort_run);
                        ckernel::topk_rebuild<false, DST_ACCUM_MODE, true, false, tie_order>(
                            0, 0, merge_rounds - 1, sort_run, log2_sort_run, false);
                        copy_dest_values<DataFormat::Int32>(2, 0);
                        ckernel::topk_merge<false, false, DST_ACCUM_MODE, true, false, tie_order>(
                            0, merge_rounds - 1, sort_run);
                        ckernel::topk_rebuild<false, DST_ACCUM_MODE, true, false, tie_order>(
                            0, 0, merge_rounds - 1, sort_run, log2_sort_run, true);
                    }
                    if (tile + 2 < expert_tiles) {
                        // (parked in DEST tile 2: fused keys leave the id tiles unused)
                        copy_dest_values<DataFormat::Int32>(0, 2);
                    }
                }
                tile_regs_commit();
            }  // (live)
            if constexpr (router_iterations > 1 || row_block::BLOCKS > 1) {
                cb_pop_front(
                    cb_scores, expert_tiles);  // (copied into DEST: the next iteration's scores reuse the pages)
            }
            cb_reserve_back(cb_best, 1);
            if (live) {
                tile_regs_wait();
                pack_reconfig_data_format(cb_best);
                pack_tile(0, cb_best);
                tile_regs_release();
            }
            cb_push_back(cb_best, 1);
            const uint32_t done_words = args.done_words;
            PACK(({  // (the last best pair packed: the joint rows can go - BRISC and NCRISC told)
                *reinterpret_cast<volatile uint32_t*>(done_words) = 1;
                *reinterpret_cast<volatile uint32_t*>(done_words + 64) = 1;
            }));
            {  // the joint rows' last two shares (tokens 16 .. 23 the MATH thread, 24 .. 31 the PACK thread; soft
               // float): the cb_best page is in L1 once PACK passed its push - PACK tells MATH (math_sync_address)
                const uint32_t best_page_l1 = args.best_page_l1, rows_l1 = args.rows_l1;
#ifdef ROUTER_CHIP_FROM_READY_PAGE
                // (my block of cb_logits: the scores)
                const uint32_t scores_l1 = args.logits_l1 + router_tp_position * ROUTER_SCORE_STRIDE;
#else
                const uint32_t scores_l1 = args.logits_l1;
#endif
                const uint32_t scale_bits = args.scale_bits;
                PACK(({
                    *reinterpret_cast<volatile uint32_t*>(args.math_sync_address) = 1;
                    if (live) {
                        routing_rows::build<top_k, expert_tiles>(best_page_l1, scores_l1, rows_l1, scale_bits, 24, 32);
                    }
                    *reinterpret_cast<volatile uint32_t*>(done_words + 48) = 1;
                }));
                MATH(({
                    volatile uint32_t* pack_done = reinterpret_cast<volatile uint32_t*>(args.math_sync_address);
                    while (*pack_done == 0) {
                    }
                    if constexpr (router_iterations > 1 || row_block::BLOCKS > 1) {
                        *pack_done = 0;  // (re-arm: PACK sets it again only after the next iteration's top-K)
                    }
                    if (live) {
                        routing_rows::build<top_k, expert_tiles>(best_page_l1, scores_l1, rows_l1, scale_bits, 16, 24);
                    }
                    *reinterpret_cast<volatile uint32_t*>(done_words + 32) = 1;
                }));
            }
#ifdef ROUTER_ROUTE_WORDS
            // dispatch prep: the route words of my 32 owned tokens from the joint rows (BRISC's word 80: all shares
            // written), into L1 for the BRISC's write (word 96); off the routing chain (the UNPACK thread is idle here)
#ifdef ROUTER_ITERATIONS_ON_ROUTER
            // (my row tile of this iteration)
            const uint32_t first_token =
                args.first_token + iteration * tt::constants::TILE_HEIGHT * ROUTER_ROW_TILES_PER_ITERATION;
#else
            const uint32_t first_token = args.first_token;
#endif
            const uint32_t route_words_l1 = args.route_words_l1, rows_l1_for_words = args.rows_l1;
            UNPACK(({
                volatile uint32_t* rows_written = reinterpret_cast<volatile uint32_t*>(done_words + 80);
                while (*rows_written == 0) {
                }
                *rows_written = 0;
                volatile uint32_t* route_words_out = reinterpret_cast<volatile uint32_t*>(route_words_l1);
                for (uint32_t column = 0; column < tt::constants::TILE_WIDTH && live; ++column) {
                    route_words::write(
                        route_words_out,
                        column,
                        reinterpret_cast<volatile const uint16_t*>(rows_l1_for_words + column * 4 * top_k),
                        first_token + column);
                }
                *reinterpret_cast<volatile uint32_t*>(done_words + 96) = 1;
            }));
#endif
        }
        // (the cb_best page read by every share of the rows: the next iteration packs it again)
        if constexpr (router_iterations > 1 || row_block::BLOCKS > 1) {
            cb_wait_front(cb_best, 1);
            cb_pop_front(cb_best, 1);
        }
    }  // (router iterations)
}

void kernel_main() {
    moe_block_args::load();  // (this core's args: the page the reader hands over)
    const RouterComputeArgs args = read_router_compute_args(
        moe_block_args::arg_pointer(0), read_router_common_args(moe_block_args::common_pointer(0)));
    // (once, bf16 sources as the logits matmul had: every reconfiguration after this is an explicit *_init /
    // reconfig_data_format call)
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_pack_to_unpack, cb_pack_to_unpack, cb_logits);
    for (uint32_t block = 0; block < row_block::BLOCKS; ++block) {
        row_block::begin(block);
        process_row_block(args);
        row_block::end(block);
    }
}
