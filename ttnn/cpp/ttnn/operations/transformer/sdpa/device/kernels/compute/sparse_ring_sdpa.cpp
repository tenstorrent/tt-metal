// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sparse ring SDPA compute on the precision recipes (host: sparse_ring_sdpa_program_factory.cpp). The recipe core loop
// (recipe/recipe_streaming.hpp) is dense SDPA's; this kernel is the sparse ring's outer loop. A core keeps the Q blocks
// of its current pass ("slots", one 64-row block each) resident with one recurrent state per slot, and applies each K/V
// piece the reader streams (once per ring iteration, heads interleaved) to every slot of the piece's head that keeps a
// block of it: one recipe step per (slot, piece) with the block mask (recipe_ring_mask 3: a K tile's valid columns are
// its block's real tokens if the slot keeps the block, none otherwise). With the pooled route the pass's first active
// iteration adds the pooled entries, one step per slot with its column-bias tiles as the additive mask
// (recipe_mask_flagged). The pass's last active iteration normalizes every slot into the output CB.
//
// Slot states: the recipe state banks (recipe_ring_cbs.hpp) hold one region per slot. O (CB 9) and l (CB 13) are
// balanced within a step (pushed and popped per chunk, the bytes staying in place), so before a slot's step the kernel
// makes the slot's region the whole CB (window_cb). Each slot's maxima, which stay pushed between steps, have their own
// CB pair (the host's slot_max_cb). The BF16-state recipes only (STANDARD, FAST): the FP32 ones keep O and l pushed
// between chunks. K and V stay in their CBs over a piece's steps (recipe_kv_hold); the kernel pops them after the
// piece.

#include <cstdint>

// The recipe step and the normalization reconfigure many times: the LLK Src zero-flag configurator stays out of line.
#define LLK_ZEROFLAG_OUTLINE 1

#if defined(WATCHER_ENABLED)
#pragma GCC optimize("Os")
#elif defined(TRISC_MATH)
#pragma GCC optimize("O2")
#endif

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/tile_constants.h"
#include "compute_common.hpp"
#include "recipe/recipe_tail.hpp"
#include "recipe/recipe_sfpu.hpp"
#include "recipe/recipe_streaming.hpp"
#include "recipe/recipe_ring.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/fused_op_indexer.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/chunked_prefill_utils.hpp"
#if defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)
#include "internal/tt-1xx/cache.h"  // invalidate_l1_cache on every compute thread (plan records read from L1)
#endif

namespace {
using namespace recipe_ring_cb;

// Positional compile-time args (the dense ring op's layout, shared with the reader and writer).
constexpr uint32_t NH = get_compile_time_arg_val(0);
constexpr uint32_t DHt = get_compile_time_arg_val(1);
constexpr uint32_t vDHt = get_compile_time_arg_val(2);
constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(3);
constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(4);
constexpr uint32_t kv_local_padded_Nt = get_compile_time_arg_val(6);
constexpr uint32_t num_local_k_chunks = get_compile_time_arg_val(12);
constexpr uint32_t num_q_chunks = get_compile_time_arg_val(14);
constexpr uint32_t ring_size = get_compile_time_arg_val(15);
constexpr uint32_t scale_fp32 = get_compile_time_arg_val(26);
constexpr uint32_t chunk_size_t = get_compile_time_arg_val(34);

constexpr uint32_t kBlockTiles = get_named_compile_time_arg_val("block_tiles");
constexpr uint32_t kBlocksPerPiece = Sk_chunk_t / kBlockTiles;
constexpr uint32_t resident_slots = get_named_compile_time_arg_val("resident_slots");
constexpr uint32_t num_q_passes = get_named_compile_time_arg_val("num_q_passes");
constexpr uint32_t kBW = get_named_compile_time_arg_val("plan_block_words");
constexpr uint32_t plan_rec_base = get_named_compile_time_arg_val("plan_rec_base");
constexpr uint32_t cb_sparse_plan = get_named_compile_time_arg_val("cb_sparse_plan");
constexpr bool pooled_route = get_named_compile_time_arg_val("pooled_route") == 1;
constexpr uint32_t pooled_chunks = get_named_compile_time_arg_val("pooled_chunks");
constexpr uint32_t pooled_last_tiles = get_named_compile_time_arg_val("pooled_last_tiles");
constexpr bool has_block_counts = get_named_compile_time_arg_val("has_block_counts") == 1;
constexpr uint32_t cb_block_counts = get_named_compile_time_arg_val("cb_block_counts");
constexpr uint32_t kMaskNb = get_named_compile_time_arg_val("mask_nb");
constexpr uint32_t kMaskNbPad = get_named_compile_time_arg_val("mask_nb_pad");
constexpr uint32_t kLogicalN = get_named_compile_time_arg_val("sparse_logical_n");
// The first extra slot's maxima CB pair (slot s > 0: slot_max_cb0 + 2 (s - 1), + 1); slot 0 uses the recipe's 10 / 11.
constexpr uint32_t slot_max_cb0 = get_named_compile_time_arg_val("slot_max_cb0");
// One region of the state banks per slot.
constexpr uint32_t kOutRegionTiles = Sq_chunk_t * vDHt * 2;  // CB 9: per Q row, O then a group's chunk PV
constexpr uint32_t kSumRegionTiles = Sq_chunk_t;             // CB 13
constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
constexpr uint32_t kt_entry_tiles = DHt * Sk_chunk_t;
constexpr uint32_t v_entry_tiles = Sk_chunk_t * vDHt;
constexpr uint32_t kRec = kBW + 3;
constexpr bool kPlanInL1 = plan_rec_base != 0;

static_assert(Sq_chunk_t == kBlockTiles, "one Q block per mask row");
static_assert(Sk_chunk_t % kBlockTiles == 0, "a K/V piece holds whole blocks");
static_assert(Sk_chunk_t <= kRecipeRingBlockTableTiles, "the block mask table covers a piece");
static_assert(recipe::ring && recipe::ring_mask == 3 && recipe::kv_hold && !recipe::ring_stream_state);
static_assert(!recipe::fp32, "the BF16-state recipes (STANDARD, FAST): one state per slot in the state banks");
static_assert(resident_slots >= 1 && resident_slots <= 32, "per-slot tables");

// One step of slot `slot`'s state over the piece at the K/V CBs' front.
__attribute__((noinline)) void sparse_ring_run_chunk(RecipeAccumulatorState& state) {
    // The BF16 recipes process two Q tile rows per group.
    constexpr uint32_t subblock_h = 2;
    sdpa_recipe_segment<
        Sq_chunk_t,
        Sk_chunk_t,
        DHt,
        vDHt,
        scale_fp32,
        subblock_h,
        recipe::qk_w,
        subblock_h,
        recipe::pv_w,
        q,
        k,
        v,
        qk,
        identity_scale,
        exp_max_diff,
        col_identity,
        recip_scratch,
        out,
        true>(state, 1, false, false);
}

// The state banks' L1 bases (each thread's own copy of the CB interfaces; set at kernel start).
uint32_t out_bank_base, sum_bank_base;

// This thread's view of `cb` resized to the window; returns the window's first address.
ALWI uint32_t set_window(uint32_t cb, uint32_t base, uint32_t tile0, uint32_t tiles) {
    auto& iface = get_local_cb_interface(cb);
    iface.fifo_size = tiles * iface.fifo_page_size;
    iface.fifo_limit = base + (tile0 + tiles) * iface.fifo_page_size;
    iface.fifo_num_pages = tiles;
    return base + tile0 * iface.fifo_page_size;
}

// UNPACK and PACK see `cb` as the window [base + tile0, base + tile0 + tiles) of its L1: the recipe step wraps the
// state banks' pointers at the CB's end (one push per row brings l back to the chunk's first row), so each slot's
// region must be a CB of its own. MATH keeps no CB pointers. The window holds no tiles between steps (each step
// balances its pushes and pops), so only the geometry and the pointers change.
ALWI void window_cb(uint32_t cb, uint32_t base, uint32_t tile0, uint32_t tiles) {
    UNPACK((get_local_cb_interface(cb).fifo_rd_ptr = set_window(cb, base, tile0, tiles)));
    PACK((get_local_cb_interface(cb).fifo_wr_ptr = set_window(cb, base, tile0, tiles)));
}

ALWI uint32_t cb_base(uint32_t cb) {
    const auto& iface = get_local_cb_interface(cb);
    return iface.fifo_limit - iface.fifo_size;
}

// The state banks and Q of slot `slot`.
ALWI void point_at_slot(uint32_t slot) {
    window_cb(out_b, out_bank_base, slot * kOutRegionTiles, kOutRegionTiles);
    window_cb(sum_b, sum_bank_base, slot * kSumRegionTiles, kSumRegionTiles);
    UNPACK({
        auto& iface = get_local_cb_interface(q);
        iface.fifo_rd_ptr = (iface.fifo_limit - iface.fifo_size) + slot * q_chunk_tiles * iface.fifo_page_size;
    })
}

// A fresh state for slot `slot`: the canonical banks, with the slot's own maxima pair.
ALWI void reset_slot_state(RecipeAccumulatorState& state, uint32_t slot) {
    recipe_ring_reset_state(state);
    if (slot > 0) {
        state.prev.max = slot_max_cb0 + 2 * (slot - 1);
        state.cur.max = slot_max_cb0 + 2 * (slot - 1) + 1;
    }
}

// Normalize slot `slot`'s state (O and l in place in its regions) into the output CB, then release its maxima. A slot
// that kept no block anywhere has no state: zeros.
__attribute__((noinline)) void normalize_slot(RecipeAccumulatorState& state, uint32_t slot) {
    constexpr uint32_t out_tiles = Sq_chunk_t * vDHt;
    if (state.processed_chunks == 0) {
        reconfig_data_format(identity_scale, identity_scale);
        pack_reconfig_data_format(out);
        configure_single_tile_pack(out);
        zero_block(identity_scale, out, out_tiles);
        pack_reconfig_data_format(recip_scratch);
        return;
    }
    point_at_slot(slot);
    const uint32_t n_out = state.cur.out, n_sum = state.cur.sum;
    // Publish the slot's O and l (at its regions' bases) to the normalization.
    CircularBuffer(n_out).reserve_back(kOutRegionTiles);
    CircularBuffer(n_sum).reserve_back(kSumRegionTiles);
    CircularBuffer(n_out).push_back(kOutRegionTiles);
    CircularBuffer(n_sum).push_back(kSumRegionTiles);
    pack_reconfig_data_format(recip_scratch);
    if constexpr (recipe::fused_active) {
        recipe_cold_step::normalize_row_streaming<
            vDHt,
            compute_kernel_lib::DEST_AUTO_LIMIT,
            col_identity,
            recip_scratch,
            out,
            scale_fp32>(n_sum, n_out, Sq_chunk_t);
    } else {
        recipe_hot_step::normalize_row_streaming<
            vDHt,
            compute_kernel_lib::DEST_AUTO_LIMIT,
            col_identity,
            recip_scratch,
            out,
            scale_fp32>(n_sum, n_out, Sq_chunk_t);
    }
    sdpa_cb_pop_front_out_of_line(state.prev.max, Sq_chunk_t);
}

ALWI bool last_active_ring_iter(uint32_t mask, uint32_t ring_iter) {
    return ring_iter + 1 >= 32 || (mask >> (ring_iter + 1)) == 0;
}

// Per-slot states (their words: a static of RecipeAccumulatorState would need dynamic initialization) and the plan
// records, static: the TRISC stacks hold only the step's frame.
static_assert(sizeof(RecipeAccumulatorState) % sizeof(uint32_t) == 0);
constexpr uint32_t kStateWords = sizeof(RecipeAccumulatorState) / sizeof(uint32_t);
uint32_t slot_words[resident_slots][kStateWords];
uint32_t plan_copy[kPlanInL1 ? 1 : resident_slots * kRec];
// The working state of the slot being stepped (one object, updated in place by the recipe step).
RecipeAccumulatorState* slot_state_ptr(uint32_t slot) {
    return reinterpret_cast<RecipeAccumulatorState*>(slot_words[slot]);
}
}  // namespace

void kernel_main() {
    uint32_t argidx = 0;
    const uint32_t global_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t ring_size_runtime = get_arg_val<uint32_t>(argidx++);
    const uint32_t ring_index_runtime = get_arg_val<uint32_t>(argidx++);
    const uint32_t forward_writes_expected = get_arg_val<uint32_t>(argidx++);
    const uint32_t backward_writes_expected = get_arg_val<uint32_t>(argidx++);
    const uint32_t logical_nt = get_arg_val<uint32_t>(argidx++);
    argidx += 4;  // KV-pad rotation words (unused: no KV cache)
    const uint32_t active_ring_iter_mask = get_arg_val<uint32_t>(argidx++);

    compute_kernel_hw_startup<SrcOrder::Reverse>(q, k, qk);
    if constexpr (recipe::lofi) {
        recipe_mm_init(q, k, true, 1, 1, 1);
    } else {
        matmul_init(q, k);
    }
    // The reduce scaler and the column identity stay fronted for the whole kernel.
    CircularBuffer(identity_scale).wait_front(1);
    CircularBuffer(col_identity).wait_front(1);
    init_sdpa_streaming_semaphores();
    out_bank_base = cb_base(out_b);
    sum_bank_base = cb_base(sum_b);
    [[maybe_unused]] const volatile uint8_t* block_counts = nullptr;
    if constexpr (has_block_counts) {
        CircularBuffer(cb_block_counts).wait_front(1);
        block_counts = reinterpret_cast<const volatile uint8_t*>(ckernel::get_tile_address(cb_block_counts, 0));
        invalidate_l1_cache();
    }
    // Real tokens of local block b of shard s, as the reader counts them.
    const auto blk_count = [&](uint32_t s, uint32_t b) -> uint32_t {
        if (b >= kMaskNb) {
            return 0;
        }
        if constexpr (has_block_counts) {
            return block_counts[s * kMaskNbPad + b];
        } else {
            return sparse_blocks::geometric_len<kBlockTiles * 32>(kv_local_padded_Nt * 32, kLogicalN, kMaskNb, s, b);
        }
    };

    const uint32_t q_per_core = global_q_end - global_q_start;
    for (uint32_t q_pass = 0; q_pass < num_q_passes; ++q_pass) {
        const uint32_t pass_q_start = global_q_start + (q_pass * q_per_core) / num_q_passes;
        const uint32_t pass_q_end = global_q_start + ((q_pass + 1) * q_per_core) / num_q_passes;
        const uint32_t n_res = pass_q_end - pass_q_start;
        const auto res_head = [&](uint32_t i) -> uint32_t { return ((pass_q_start + i) / num_q_chunks) % NH; };
        for (uint32_t i = 0; i < n_res; ++i) {
            reset_slot_state(*slot_state_ptr(i), i);
        }
        RingSDPAOpIndexer fused_op_indexer(
            ring_size_runtime, ring_index_runtime, forward_writes_expected, backward_writes_expected);
        bool pooled_pending = pooled_route;
        for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
            const uint32_t ring_id = fused_op_indexer.get_next_ring_id_and_sync();
            if (((active_ring_iter_mask >> ring_iter) & 1u) == 0) {
                continue;
            }
            // The iteration's plan: per slot [kept-block words][head-group piece union lo, hi][do_pooled].
            [[maybe_unused]] const volatile uint32_t* plan_l1 = nullptr;
            if constexpr (kPlanInL1) {
                CircularBuffer(cb_sparse_plan).wait_front(1);
                plan_l1 = reinterpret_cast<const volatile uint32_t*>(ckernel::get_tile_address(cb_sparse_plan, 0)) +
                          plan_rec_base;
                invalidate_l1_cache();
            } else {
                for (uint32_t i = 0; i < n_res; ++i) {
                    CircularBuffer(cb_sparse_plan).wait_front(1);
                    for (uint32_t w = 0; w < kRec; ++w) {
                        plan_copy[i * kRec + w] = ckernel::read_tile_value(cb_sparse_plan, 0, w);
                    }
                    CircularBuffer(cb_sparse_plan).pop_front(1);
                }
            }
            const auto plan_word = [&](uint32_t i, uint32_t w) -> uint32_t {
                if constexpr (kPlanInL1) {
                    return plan_l1[i * kRec + w];
                } else {
                    return plan_copy[i * kRec + w];
                }
            };
            const auto union_kept = [&](uint32_t i, uint32_t piece) -> bool {
                const uint32_t w = piece < 32 ? plan_word(i, kBW) : plan_word(i, kBW + 1);
                return ((w >> (piece & 31)) & 1u) != 0;
            };
            const auto block_kept = [&](uint32_t i, uint32_t b) -> bool {
                return b < kBW * 32 && ((plan_word(i, b >> 5) >> (b & 31)) & 1u) != 0;
            };
            const bool do_pooled = pooled_pending && n_res > 0 && plan_word(0, kBW + 2) != 0;
            // Steps slot i over the piece at the K/V CBs' front with the current mask.
            const auto step_slot = [&](uint32_t i) {
                point_at_slot(i);
                sparse_ring_run_chunk(*slot_state_ptr(i));
            };
            const auto pop_kv = [&]() {
                CircularBuffer(k).wait_front(kt_entry_tiles);
                sdpa_cb_pop_front_out_of_line(k, kt_entry_tiles);
                CircularBuffer(v).wait_front(v_entry_tiles);
                sdpa_cb_pop_front_out_of_line(v, v_entry_tiles);
            };
            // The shard's pieces, in the reader's order: per piece, the heads of the slots in order, a head's piece
            // streamed iff its group's union keeps it.
            recipe_edge_chunk = false;
            for (uint32_t piece = 0; piece < num_local_k_chunks; ++piece) {
                if (!kv_chunk_starts_before_logical_end<
                        false,
                        false,
                        kv_local_padded_Nt,
                        chunk_size_t,
                        kv_local_padded_Nt>(ring_id, piece * Sk_chunk_t, logical_nt)) {
                    continue;
                }
                for (uint32_t i0 = 0; i0 < n_res;) {
                    uint32_t i1 = i0;
                    while (i1 < n_res && res_head(i1) == res_head(i0)) {
                        ++i1;
                    }
                    if (union_kept(i0, piece)) {
                        for (uint32_t i = i0; i < i1; ++i) {
                            // The slot's block mask of this piece: a kept block's tiles cover its real tokens.
                            bool any = false, masked = false;
                            for (uint32_t j = 0; j < kBlocksPerPiece; ++j) {
                                const uint32_t b = piece * kBlocksPerPiece + j;
                                const uint32_t count = block_kept(i, b) ? blk_count(ring_id, b) : 0;
                                any = any || count > 0;
                                for (uint32_t t = 0; t < kBlockTiles; ++t) {
                                    const uint32_t lo = t * 32;
                                    const uint32_t valid = count <= lo ? 0 : (count - lo < 32 ? count - lo : 32);
                                    recipe_ring_tile_valid[j * kBlockTiles + t] = static_cast<uint8_t>(valid);
                                    masked = masked || valid < 32;
                                }
                            }
                            if (any) {
                                recipe_ring_mask.masked = masked;
                                step_slot(i);
                            }
                        }
                        pop_kv();
                    }
                    i0 = i1;
                }
            }
            // Pooled entries (the pass's first active iteration): per head, each pooled chunk once, every slot of the
            // head stepping it with its bias tiles (the reader pushes them in slot order after the chunk's K / V).
            if constexpr (pooled_route) {
                if (do_pooled) {
                    for (uint32_t i0 = 0; i0 < n_res;) {
                        uint32_t i1 = i0;
                        while (i1 < n_res && res_head(i1) == res_head(i0)) {
                            ++i1;
                        }
                        for (uint32_t pc = 0; pc < pooled_chunks; ++pc) {
                            const uint32_t tiles = pc + 1 == pooled_chunks ? pooled_last_tiles : Sk_chunk_t;
                            for (uint32_t t = 0; t < Sk_chunk_t; ++t) {
                                recipe_ring_tile_valid[t] = t < tiles ? 32 : 0;
                            }
                            recipe_ring_mask.masked = tiles < Sk_chunk_t;
                            recipe_edge_chunk = true;
                            for (uint32_t i = i0; i < i1; ++i) {
                                step_slot(i);
                            }
                            recipe_edge_chunk = false;
                            pop_kv();
                        }
                        i0 = i1;
                    }
                }
            }
            pooled_pending = false;
            if constexpr (kPlanInL1) {
                CircularBuffer(cb_sparse_plan).pop_front(1);
            }
            if (last_active_ring_iter(active_ring_iter_mask, ring_iter)) {
                for (uint32_t i = 0; i < n_res; ++i) {
                    normalize_slot(*slot_state_ptr(i), i);
                }
                // With several passes the reader fills the Q CB to capacity every pass: pop it whole from its base,
                // so the next pass's Q starts there too.
                point_at_slot(0);
                constexpr uint32_t q_pass_tiles = num_q_passes > 1 ? resident_slots * q_chunk_tiles : 0;
                const uint32_t q_tiles_pushed = num_q_passes > 1 ? q_pass_tiles : n_res * q_chunk_tiles;
                CircularBuffer(q).wait_front(q_tiles_pushed);
                sdpa_cb_pop_front_out_of_line(q, q_tiles_pushed);
            }
        }
    }
    if constexpr (has_block_counts) {
        CircularBuffer(cb_block_counts).pop_front(1);
    }
}
