// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sparse ring SDPA compute on the precision recipes (host: sparse_ring_sdpa_program_factory.cpp). The recipe core loop
// (recipe/recipe_streaming.hpp) is dense SDPA's; this kernel is the sparse ring's outer loop. A core keeps the Q blocks
// of its current pass ("slots", one block each) resident with one recurrent state per slot, and applies each K/V piece
// the reader streams (once per ring iteration, heads interleaved) to the slots of the piece's head. The slots of one
// head are stacked into one Q chunk of stack_slots blocks per recipe step (a window of consecutive slots), with the
// block mask per Q tile row (recipe_ring_row_blocks): a row sees a K tile's valid columns (its block's real tokens)
// only if its slot keeps the block. A slot that keeps no block of the piece, or that belongs to another head, is
// skipped inside the step (its state untouched). With the pooled route the pass's first active iteration adds the
// pooled entries, one step per window with each slot's column-bias tiles as the additive mask (recipe_mask_flagged).
// The pass's last active iteration normalizes every slot into the output CB.
//
// Slot states: the recipe state banks (recipe_ring_cbs.hpp) hold one region per slot: O (CB 9), l (CB 13) and the
// maxima (CB 10). A step sees its window of them as whole CBs (window_cb): the recipe step wraps the banks' pointers at
// a CB's end. Every slot starts a pass with O = l = 0 and the masked maximum, and every step is a continuation (no
// first chunk): a slot's first kept block takes the fused chunk's redo (recipe_ring_rows_fresh), whose rescale by
// exp(scale (masked - m)) = 0 drops the initial state exactly. CB 11 is the steps' maxima scratch (a pooled step's
// reduce path leaves its new maxima there; they are copied back to CB 10). The BF16-state recipes only (STANDARD,
// FAST): the FP32 ones keep O and l pushed between chunks. K and V stay in their CBs over a piece's steps
// (recipe_kv_hold); the kernel pops them after the piece.

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
// Slots per recipe step (the stacked Q chunk) and its Q tile rows.
constexpr uint32_t stack_slots = get_named_compile_time_arg_val("stack_slots");
constexpr uint32_t kStepRows = stack_slots * Sq_chunk_t;
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
// One region of the state banks per slot.
constexpr uint32_t kOutRegionTiles = Sq_chunk_t * vDHt * 2;  // CB 9: per Q row, O then a group's chunk PV
constexpr uint32_t kSumRegionTiles = Sq_chunk_t;             // CB 13, and the maxima in CB 10
constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
constexpr uint32_t kt_entry_tiles = DHt * Sk_chunk_t;
constexpr uint32_t v_entry_tiles = Sk_chunk_t * vDHt;
constexpr uint32_t kRec = kBW + 3;
constexpr bool kPlanInL1 = plan_rec_base != 0;
constexpr uint32_t kAllBlocks = (1u << kBlocksPerPiece) - 1u;

static_assert(Sq_chunk_t == kBlockTiles, "one Q block per mask row");
static_assert(Sk_chunk_t % kBlockTiles == 0, "a K/V piece holds whole blocks");
static_assert(Sk_chunk_t <= kRecipeRingBlockTableTiles, "the block mask table covers a piece");
static_assert(kBlocksPerPiece < 32, "per-row kept blocks of a piece in one word");
static_assert(recipe::ring && recipe::ring_mask == 3 && recipe::kv_hold && !recipe::ring_stream_state);
static_assert(recipe::ring_block_tiles == kBlockTiles, "the recipe's block mask rows are Q blocks");
static_assert(!recipe::fp32, "the BF16-state recipes (STANDARD, FAST): one state per slot in the state banks");
static_assert(resident_slots >= 1 && resident_slots <= 32, "per-slot bit masks");
static_assert(stack_slots >= 1 && stack_slots <= resident_slots && kStepRows <= 32, "a step's rows fit the row masks");
static_assert(stack_slots == 1 || kBlockTiles % 2 == 0, "stacked blocks align with the two-row groups");

// The wrapper's own steps around the recipe chunks (windows, state setup, copies, normalization) build for size: the
// tightest sparse ring program had 928 B of the kernel config buffer left with them at the kernel's level.
#define SPARSE_RING_COLD __attribute__((noinline, optimize("Os")))

// A piece's step: the fused chunk over the stacked window (every step continues its states, and a piece is never
// masked additively). The reduce path is not instantiated for the stacked chunk (its code grows with the Q chunk);
// a fresh group's redo inside the fused chunk is its first chunk. Size-optimized on UNPACK and PACK (kernel config
// buffer). Then the step's O and l pushes are released (the bytes stay in place).
__attribute__((noinline)) void sparse_ring_fused_step(RecipeAccumulatorState& state) {
    recipe_arm_score_exp<scale_fp32>();
    if constexpr (recipe::unpack_thread || recipe::pack_thread) {
        recipe_cold_step::fused_os::sdpa_fused_chunk<
            kStepRows,
            Sk_chunk_t,
            DHt,
            vDHt,
            scale_fp32,
            recipe::qk_w,
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
            true>(state.prev, state.cur, false, false);
    } else {
        recipe_cold_step::fused_plain::sdpa_fused_chunk<
            kStepRows,
            Sk_chunk_t,
            DHt,
            vDHt,
            scale_fp32,
            recipe::qk_w,
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
            true>(state.prev, state.cur, false, false);
    }
    sdpa_cb_pop_front_out_of_line(state.cur.out, kStepRows * vDHt * sdpa_out_stride);
    sdpa_cb_pop_front_out_of_line(state.cur.sum, kStepRows);
}

// A pooled entry's step for one slot: the reduce path with the slot's column-bias tiles as the additive mask. Its
// maxima ping-pong leaves the new ones in CB 11; the caller copies them back.
__attribute__((noinline)) void sparse_ring_pooled_step(RecipeAccumulatorState& state) {
    constexpr uint32_t subblock_h = 2;
    recipe_cold_step::sdpa_inner_loop_step<
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
        true,
        false>(state.prev, state.cur, false, false, false);
    sdpa_cb_pop_front_out_of_line(state.cur.out, Sq_chunk_t * vDHt * sdpa_out_stride);
    sdpa_cb_pop_front_out_of_line(state.cur.sum, Sq_chunk_t);
    sdpa_cb_pop_front_out_of_line(state.prev.max, Sq_chunk_t);
}

// The state banks' and the step scratch CBs' L1 bases (each thread's own copy of the CB interfaces; set at kernel
// start).
uint32_t out_bank_base, sum_bank_base, max_bank_base, max_scratch_base, qk_base, chunk_sum_base, exp_diff_base;

// This thread's view of `cb` resized to the window; returns the window's first address.
ALWI uint32_t set_window(uint32_t cb, uint32_t base, uint32_t tile0, uint32_t tiles) {
    auto& iface = get_local_cb_interface(cb);
    iface.fifo_size = tiles * iface.fifo_page_size;
    iface.fifo_limit = base + (tile0 + tiles) * iface.fifo_page_size;
    iface.fifo_num_pages = tiles;
    return base + tile0 * iface.fifo_page_size;
}

// UNPACK and PACK see `cb` as the window [base + tile0, base + tile0 + tiles) of its L1: the recipe step wraps the
// state banks' pointers at the CB's end (one push per row brings l back to the chunk's first row), so a step's
// region must be a CB of its own. MATH keeps no CB pointers. Between steps the window's pushes and pops balance (the
// maxima stay pushed as a whole), so only the geometry and the pointers change.
ALWI void window_cb(uint32_t cb, uint32_t base, uint32_t tile0, uint32_t tiles) {
    UNPACK((get_local_cb_interface(cb).fifo_rd_ptr = set_window(cb, base, tile0, tiles)));
    PACK((get_local_cb_interface(cb).fifo_wr_ptr = set_window(cb, base, tile0, tiles)));
}

ALWI uint32_t cb_base(uint32_t cb) {
    const auto& iface = get_local_cb_interface(cb);
    return iface.fifo_limit - iface.fifo_size;
}

// The state banks and Q of the `n` slots from slot `s`, and the step scratch CBs sized for them (a step's pushes wrap
// at their ends).
SPARSE_RING_COLD void point_at_slots(uint32_t s, uint32_t n) {
    window_cb(out_b, out_bank_base, s * kOutRegionTiles, n * kOutRegionTiles);
    window_cb(sum_b, sum_bank_base, s * kSumRegionTiles, n * kSumRegionTiles);
    window_cb(max_a, max_bank_base, s * kSumRegionTiles, n * kSumRegionTiles);
    const uint32_t step_rows = (n < stack_slots ? n : stack_slots) * Sq_chunk_t;
    window_cb(max_b, max_scratch_base, 0, step_rows);
    window_cb(qk, qk_base, 0, step_rows * Sk_chunk_t);
    window_cb(sum_a, chunk_sum_base, 0, step_rows);
    window_cb(exp_max_diff, exp_diff_base, 0, step_rows);
    UNPACK({
        auto& iface = get_local_cb_interface(q);
        iface.fifo_rd_ptr = (iface.fifo_limit - iface.fifo_size) + s * q_chunk_tiles * iface.fifo_page_size;
    })
}

// `tiles` tiles of `value_bits` (0, or the masked score) packed into `cb` from its write pointer and published.
SPARSE_RING_COLD void fill_tiles(uint32_t cb, uint32_t tiles, uint32_t value_bits) {
    reconfig_data_format(identity_scale, identity_scale);
    pack_reconfig_data_format(cb);
    configure_single_tile_pack(cb);
    CircularBuffer(cb).reserve_back(tiles);
    sub_init(identity_scale, identity_scale);
    tile_regs_acquire();
    sub_tiles(identity_scale, identity_scale, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    if (value_bits != 0) {
        PACK({
            addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
            SFPU_UNARY_CALL(
                DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_fill_bitcast_, (APPROX, 8), 0, VectorMode::RC, value_bits);
            TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU);
        })
    }
    for (uint32_t i = 0; i < tiles; ++i) {
        pack_tile(0, cb);
    }
    tile_regs_release();
    CircularBuffer(cb).push_back(tiles);
}

// A pass's slot states: O and l zero (released again: the bytes stay), the maxima the masked score (kept pushed for
// the pass).
SPARSE_RING_COLD void init_slot_states() {
    point_at_slots(0, resident_slots);
    fill_tiles(out_b, resident_slots * kOutRegionTiles, 0);
    fill_tiles(sum_b, resident_slots * kSumRegionTiles, 0);
    fill_tiles(max_a, resident_slots * kSumRegionTiles, kRecipeRingMaskedBits);
    CircularBuffer(out_b).wait_front(resident_slots * kOutRegionTiles);
    CircularBuffer(out_b).pop_front(resident_slots * kOutRegionTiles);
    CircularBuffer(sum_b).wait_front(resident_slots * kSumRegionTiles);
    CircularBuffer(sum_b).pop_front(resident_slots * kSumRegionTiles);
    reconfig_data_format(qk, qk);
    pack_reconfig_data_format(qk);
}

// A pooled step's reduce path swapped the slot's maxima into the scratch (CB 11): copy them back to its CB 10 window.
SPARSE_RING_COLD void copy_back_maxima() {
    CircularBuffer(max_b).wait_front(Sq_chunk_t);
    // No reserve: CB 10 holds every slot's maxima pushed, so a reserve against the slot's window would count the other
    // slots' tiles (it never returns when they fill the window's page count). The step popped the slot's tiles, and
    // these packs follow the copies of CB 11 that follow the step on every thread.
    reconfig_data_format_srca(qk, max_b);
    pack_reconfig_data_format(max_a);
    configure_single_tile_pack(max_a);
    copy_init(max_b);
    for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
        tile_regs_acquire();
        copy_tile(max_b, r, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, max_a);
        tile_regs_release();
    }
    CircularBuffer(max_a).push_back(Sq_chunk_t);
    CircularBuffer(max_b).pop_front(Sq_chunk_t);
    reconfig_data_format_srca(max_b, qk);
    pack_reconfig_data_format(qk);
}

// Normalize slot `slot`'s state (O and l in place in its regions) into the output CB. A slot that kept no block
// anywhere: zeros.
SPARSE_RING_COLD void normalize_slot(uint32_t slot, bool touched) {
    constexpr uint32_t out_tiles = Sq_chunk_t * vDHt;
    if (!touched) {
        reconfig_data_format(identity_scale, identity_scale);
        pack_reconfig_data_format(out);
        configure_single_tile_pack(out);
        zero_block(identity_scale, out, out_tiles);
        pack_reconfig_data_format(recip_scratch);
        return;
    }
    window_cb(out_b, out_bank_base, slot * kOutRegionTiles, kOutRegionTiles);
    window_cb(sum_b, sum_bank_base, slot * kSumRegionTiles, kSumRegionTiles);
    // Publish the slot's O and l (at its regions' bases) to the normalization.
    CircularBuffer(out_b).reserve_back(kOutRegionTiles);
    CircularBuffer(sum_b).reserve_back(kSumRegionTiles);
    CircularBuffer(out_b).push_back(kOutRegionTiles);
    CircularBuffer(sum_b).push_back(kSumRegionTiles);
    pack_reconfig_data_format(recip_scratch);
    if constexpr (recipe::fused_active) {
        recipe_cold_step::normalize_row_streaming<
            vDHt,
            compute_kernel_lib::DEST_AUTO_LIMIT,
            col_identity,
            recip_scratch,
            out,
            scale_fp32>(sum_b, out_b, Sq_chunk_t);
    } else {
        recipe_hot_step::normalize_row_streaming<
            vDHt,
            compute_kernel_lib::DEST_AUTO_LIMIT,
            col_identity,
            recip_scratch,
            out,
            scale_fp32>(sum_b, out_b, Sq_chunk_t);
    }
}

ALWI bool last_active_ring_iter(uint32_t mask, uint32_t ring_iter) {
    return ring_iter + 1 >= 32 || (mask >> (ring_iter + 1)) == 0;
}

// The plan records, static: the TRISC stacks hold only the step's frame.
uint32_t plan_copy[kPlanInL1 ? 1 : resident_slots * kRec];
// Per slot, its kept blocks of the current chunk (the step's row masks), and the slots whose state has seen no key.
uint32_t slot_blocks[resident_slots];
uint32_t fresh_slots;

// Steps the window of slots [w0, w1) of one head (the stacked chunk starts at slot s <= w0) with their kept blocks
// (slot_blocks); slots outside the window keep no block. Slots that kept a block have seen a key afterwards.
SPARSE_RING_COLD void step_window(uint32_t w0, uint32_t w1, bool partial_tiles) {
    const uint32_t s = w0 < resident_slots - stack_slots ? w0 : resident_slots - stack_slots;
    uint32_t stepped = 0, fresh_rows = 0;
    bool masked = partial_tiles;
    for (uint32_t j = 0; j < stack_slots; ++j) {
        const uint32_t i = s + j;
        const uint32_t bits = i >= w0 && i < w1 ? slot_blocks[i] : 0u;
        masked = masked || bits != kAllBlocks;
        if (bits != 0) {
            stepped |= 1u << i;
            if ((fresh_slots >> i) & 1u) {
                fresh_rows |= ((1u << Sq_chunk_t) - 1u) << (j * Sq_chunk_t);
            }
        }
        for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
            recipe_ring_row_blocks[j * Sq_chunk_t + r] = bits;
        }
    }
    if (stepped == 0) {
        return;
    }
    recipe_ring_rows_fresh = fresh_rows;
    recipe_ring_mask.masked = masked;
    point_at_slots(s, stack_slots);
    RecipeAccumulatorState state;
    recipe_ring_reset_state(state);
    sparse_ring_fused_step(state);
    fresh_slots &= ~stepped;
}

// Pooled chunk step for slot i (all its rows keep every pooled column; the bias tiles mask them).
SPARSE_RING_COLD void pooled_slot_step(uint32_t i, bool partial_tiles) {
    for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
        recipe_ring_row_blocks[r] = kAllBlocks;
    }
    recipe_ring_rows_fresh = 0;
    recipe_ring_mask.masked = partial_tiles;
    point_at_slots(i, 1);
    RecipeAccumulatorState state;
    recipe_ring_reset_state(state);
    sparse_ring_pooled_step(state);
    copy_back_maxima();
    fresh_slots &= ~(1u << i);
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
    max_bank_base = cb_base(max_a);
    max_scratch_base = cb_base(max_b);
    qk_base = cb_base(qk);
    chunk_sum_base = cb_base(sum_a);
    exp_diff_base = cb_base(exp_max_diff);
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
        fresh_slots = 0xffffffffu;
        bool states_ready = false;
        RingSDPAOpIndexer fused_op_indexer(
            ring_size_runtime, ring_index_runtime, forward_writes_expected, backward_writes_expected);
        bool pooled_pending = pooled_route;
        for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
            const uint32_t ring_id = fused_op_indexer.get_next_ring_id_and_sync();
            if (((active_ring_iter_mask >> ring_iter) & 1u) == 0) {
                continue;
            }
            if (!states_ready) {
                init_slot_states();
                states_ready = true;
            }
            // The reader pushes the pass's Q (to the CB's capacity) on its first active iteration.
            CircularBuffer(q).wait_front(resident_slots * q_chunk_tiles);
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
                // The piece's tiles: each covers its block's real tokens (the same for every slot).
                uint32_t live_blocks = 0;
                bool partial_tiles = false;
                for (uint32_t j = 0; j < kBlocksPerPiece; ++j) {
                    const uint32_t count = blk_count(ring_id, piece * kBlocksPerPiece + j);
                    live_blocks |= (count > 0 ? 1u : 0u) << j;
                    for (uint32_t t = 0; t < kBlockTiles; ++t) {
                        const uint32_t lo = t * 32;
                        const uint32_t valid = count <= lo ? 0 : (count - lo < 32 ? count - lo : 32);
                        recipe_ring_tile_valid[j * kBlockTiles + t] = static_cast<uint8_t>(valid);
                        partial_tiles = partial_tiles || valid < 32;
                    }
                }
                for (uint32_t i0 = 0; i0 < n_res;) {
                    uint32_t i1 = i0;
                    while (i1 < n_res && res_head(i1) == res_head(i0)) {
                        ++i1;
                    }
                    if (union_kept(i0, piece)) {
                        for (uint32_t i = i0; i < i1; ++i) {
                            uint32_t bits = 0;
                            for (uint32_t j = 0; j < kBlocksPerPiece; ++j) {
                                bits |= (block_kept(i, piece * kBlocksPerPiece + j) ? 1u : 0u) << j;
                            }
                            slot_blocks[i] = bits & live_blocks;
                        }
                        for (uint32_t w0 = i0; w0 < i1; w0 += stack_slots) {
                            step_window(w0, w0 + stack_slots < i1 ? w0 + stack_slots : i1, partial_tiles);
                        }
                        pop_kv();
                    }
                    i0 = i1;
                }
            }
            // Pooled entries (the pass's first active iteration): per head, each pooled chunk once, every window of the
            // head stepping it with its slots' bias tiles (the reader pushes them in slot order after the chunk's K /
            // V).
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
                            recipe_edge_chunk = true;
                            for (uint32_t i = i0; i < i1; ++i) {
                                pooled_slot_step(i, tiles < Sk_chunk_t);
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
                    normalize_slot(i, ((fresh_slots >> i) & 1u) == 0);
                }
                // The maxima and Q stay pushed for the pass: pop them whole from their bases, so the next pass starts
                // there too.
                point_at_slots(0, resident_slots);
                CircularBuffer(max_a).wait_front(resident_slots * kSumRegionTiles);
                sdpa_cb_pop_front_out_of_line(max_a, resident_slots * kSumRegionTiles);
                sdpa_cb_pop_front_out_of_line(q, resident_slots * q_chunk_tiles);
            }
        }
    }
    if constexpr (has_block_counts) {
        CircularBuffer(cb_block_counts).pop_front(1);
    }
}
