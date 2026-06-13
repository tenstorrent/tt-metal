// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute for indexer_score. Per work unit (q_tiles_per_unit q-tile-rows x
// k_tiles_in_unit k-tiles):
//   acc[r,c] = sum_h relu(q[h,row,:] @ k[col,:]^T) * w[h,row]
//   tile on the causal diagonal: += -inf strict upper triangle; past it: full -inf
//   pack_untilize acc -> bf16 row-major out, (r, c) row-major order
// Heads stream in heads_per_group groups, heads_per_dest_pass rows per DEST
// subblock (bf16 DEST by default, half sync; sized by
// determine_largest_subblock_size on the host). q/w stay resident per group
// when all heads fit.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"

#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "indexer_score_common.hpp"

constexpr uint32_t cb_q = tt::CBIndex::c_0;
constexpr uint32_t cb_k = tt::CBIndex::c_1;
constexpr uint32_t cb_w = tt::CBIndex::c_2;
constexpr uint32_t cb_mask = tt::CBIndex::c_3;
constexpr uint32_t cb_qk = tt::CBIndex::c_24;
constexpr uint32_t cb_acc = tt::CBIndex::c_26;
constexpr uint32_t cb_out = tt::CBIndex::c_16;
constexpr uint32_t cb_acc_strip = tt::CBIndex::c_27;  // full-width strip accumulator (fast untilize input)
constexpr uint32_t cb_out_strip = tt::CBIndex::c_18;  // full-width strip output (fast untilize)

// qk subblock height (head rows per DEST pass); first per-kernel compile-time arg.
constexpr uint32_t heads_per_dest_pass = get_compile_time_arg_val(num_common_ct_args);
// heads buffered in cb_qk per matmul/mul phase chunk (multiple of HP).
constexpr uint32_t qk_batch_heads = get_compile_time_arg_val(num_common_ct_args + 1);
// k-columns batched per matmul<->mul mode switch in the full-strip path (1 = per-column, no batching).
constexpr uint32_t qk_col_batch = get_compile_time_arg_val(num_common_ct_args + 2);

// The per-(r,c) head reduction runs as two phases per head group: matmul_phase fills cb_qk
// with the group's relu(q.kT) tiles, then mul_accum_phase multiplies by the gates and packs
// each head onto the single accumulator tile via L1-accumulation. Splitting the phases lets
// the matmul<->eltwise reconfig + init happen once per group (set_matmul_mode / set_mul_mode)
// instead of once per head pass.

/** Configure srcA<-k, srcB<-q and matmul mode for the group's matmul phase; cb_qk packs overwrite.
 *  matmul maps in1->srcA, in0->srcB (matmul.h hw_configure(in1, in0)) -- swapping the reconfig
 *  misreads k when its format differs from q (invisible for bf16 k, corrupts bfp8 k). */
template <uint32_t q_cb, uint32_t k_cb, uint32_t qk_cb>
inline void set_matmul_mode() {
    // 4-arg guarded reconfig (prior mode = mul: srcA=qk, srcB=w): srcA qk->k changes format (fires),
    // srcB w->q is bf16->bf16 so the guard skips it -- only the operand select (in the matmul call)
    // differs, not the format. Saves one unpack reconfig stall per mode switch.
    reconfig_data_format(qk_cb, k_cb, cb_w, q_cb);
    // guarded: qk and the prior pack target (cb_out) share acc_fmt in the bf16 path, so this is a
    // no-op there; only the fp32_dest_acc fallback (qk=fp32, out=bf16) actually reconfigs.
    pack_reconfig_data_format(cb_out, qk_cb);
    pack_reconfig_l1_acc(0);  // cb_qk packs overwrite (mul phase turns L1-acc on for cb_acc)
    mm_block_init_short(
        q_cb, k_cb, 1 /*transpose k*/, 1 /*ct_dim*/, heads_per_dest_pass /*rt_dim*/, head_dim_tiles /*kt_dim*/);
    pack_relu_config(ReluConfig::zero());  // relu in the packer for the whole matmul phase
}

/** One DEST pass of the matmul phase: qk_cb += relu(q_cb[head..+HP] of row r @ k_cb col c^T).
 *  q blocks are [q_tiles_per_unit][heads_per_group][head_dim_tiles] so the subblock's head rows
 *  stride head_dim_tiles. Assumes set_matmul_mode() was called for the group. */
template <uint32_t q_cb, uint32_t k_cb, uint32_t qk_cb>
void matmul_relu_pass(uint32_t head_in_group, uint32_t r, uint32_t c) {
    tile_regs_acquire();
    const uint32_t q_base = (r * heads_per_group + head_in_group) * head_dim_tiles;
    for (uint32_t d = 0; d < head_dim_tiles; ++d) {
        matmul_block(
            q_cb,
            k_cb,
            q_base + d,
            c * head_dim_tiles + d,
            0,
            1 /*transpose k*/,
            1,
            heads_per_dest_pass,
            head_dim_tiles);
    }
    tile_regs_commit();
    cb_reserve_back(qk_cb, heads_per_dest_pass);
    tile_regs_wait();
    pack_tile_block(0, qk_cb, heads_per_dest_pass);  // one block pack of the pass's relu(q.kT) tiles
    tile_regs_release();
    cb_push_back(qk_cb, heads_per_dest_pass);
}

/** Configure srcA<-qk, srcB<-w and bcast-mul mode for the group's mul+accumulate phase. */
template <uint32_t qk_cb, uint32_t w_cb, uint32_t acc_cb>
inline void set_mul_mode() {
    pack_relu_config(ReluConfig::none());  // accumulator packs stay linear (negative gates)
    // 4-arg guarded reconfig (prior mode = matmul: srcA=k, srcB=q): srcA k->qk changes format (fires),
    // srcB q->w is bf16->bf16 so the guard skips it. Saves one unpack reconfig stall per mode switch.
    reconfig_data_format(cb_k, qk_cb, cb_q, w_cb);
    // guarded: qk and acc share acc_fmt, so this never actually reconfigs (no-op stall removed).
    pack_reconfig_data_format(qk_cb, acc_cb);
    mul_bcast_cols_init_short(qk_cb, w_cb);
    // Override the math init with acc_to_dest=1 so each mul_tiles_bcast_cols MACs onto the same
    // DEST tile (hardware accumulate) instead of writing its own. tile_regs_acquire zeroes the
    // tile, so head 0 seeds it; the whole chunk's head reduction then needs a single pack rather
    // than one packer-L1-acc round trip per head (the old serialized same-address RMW chain).
    MATH((llk_math_eltwise_binary_init<ckernel::EltwiseBinaryType::ELWMUL, ckernel::BroadcastType::COL, MATH_FIDELITY>(
        qk_cb, w_cb, 1 /*acc_to_dest*/)));
}

/** Mul+accumulate one chunk of `n` heads (all resident in qk_cb) onto the output tile via the
 *  hardware MAC: dst0 += sum_h relu(qk_cb[h]) * w[w_base+h], summed in DEST by acc_to_dest, then
 *  packed once onto acc_cb[acc_slot]. `first` is the first chunk of the output tile and overwrites
 *  the slot (l1_acc off); later chunks (streamed / over-cap head groups) L1-accumulate onto it.
 *  Caller reserves/pushes acc_cb and resets l1_acc afterwards. */
template <uint32_t qk_cb, uint32_t w_cb, uint32_t acc_cb>
void mul_accum_chunk(uint32_t w_base, uint32_t n, bool first, uint32_t acc_slot) {
    cb_wait_front(qk_cb, n);
    tile_regs_acquire();
    for (uint32_t h = 0; h < n; ++h) {
        mul_tiles_bcast_cols(qk_cb, w_cb, h, w_base + h, 0);  // dst0 += relu(qk[h]) * w[h]
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_l1_acc(first ? 0 : 1);   // first chunk seeds (overwrite), later chunks accumulate
    pack_tile<true>(0, acc_cb, acc_slot);  // one pack for the whole chunk's head sum
    tile_regs_release();
    cb_pop_front(qk_cb, n);
}

/** Head reduction for one output tile (r, c): acc front = sum_h relu(q[h,r].k[c]^T) * w[h,r],
 *  MAC-summed in DEST per chunk (mul_accum_chunk), with L1-acc only across chunks. Caller reserves
 *  cb_acc and pushes it. Factored out so the masked-suffix and unmasked-prefix paths share it. */
template <uint32_t acc_cb>
inline void accumulate_heads(uint32_t r, uint32_t c, uint32_t acc_slot) {
    bool first = true;
    for (uint32_t group_start = 0; group_start < num_heads; group_start += heads_per_group) {
        if constexpr (stream_heads) {
            cb_wait_front(cb_q, q_group_tiles);
        }
        // process the group in qk_batch_heads-sized chunks (cb_qk capacity): each chunk runs all
        // its matmuls, then MACs the whole chunk's head sum into DEST in one pass, so both the
        // matmul<->eltwise reinit and the accumulator pack happen once per chunk, not per head
        for (uint32_t chunk = 0; chunk < heads_per_group; chunk += qk_batch_heads) {
            const uint32_t chunk_end = chunk + qk_batch_heads;
            set_matmul_mode<cb_q, cb_k, cb_qk>();
            for (uint32_t head = chunk; head < chunk_end; head += heads_per_dest_pass) {
                matmul_relu_pass<cb_q, cb_k, cb_qk>(head, r, c);
            }
            set_mul_mode<cb_qk, cb_w, acc_cb>();
            // w is laid out [q_tiles_per_unit][num_heads] (see reader read_w_group)
            const uint32_t w_base = r * num_heads + group_start + chunk;
            mul_accum_chunk<cb_qk, cb_w, acc_cb>(w_base, qk_batch_heads, first, acc_slot);
            first = false;
        }
        if constexpr (stream_heads) {
            cb_pop_front(cb_q, q_group_tiles);
        }
    }
    pack_reconfig_l1_acc(0);  // done accumulating; downstream packs (mask, untilize) overwrite
}

/** acc front += mask_cb[idx] (0 = diag strict-upper -inf, 1 = full -inf), via a real eltwise
 *  add (preserves -inf; the packer L1-acc path does not). Pops + repushes the acc front, so it
 *  must run when the masked tile is the only one in cb_acc -- which is why masked tiles (always a
 *  contiguous suffix of a unit's c-range) are produced one at a time after the unmasked prefix
 *  has been drained by its batch untilize. */
template <uint32_t acc_cb, uint32_t mask_cb>
void add_mask(uint32_t idx) {
    reconfig_data_format(acc_cb, mask_cb);
    pack_reconfig_data_format(acc_cb);
    add_tiles_init(acc_cb, mask_cb);
    cb_wait_front(acc_cb, 1);
    tile_regs_acquire();
    add_tiles(acc_cb, mask_cb, 0, idx, 0);
    tile_regs_commit();
    cb_pop_front(acc_cb, 1);
    cb_reserve_back(acc_cb, 1);
    tile_regs_wait();
    pack_tile(0, acc_cb);
    tile_regs_release();
    cb_push_back(acc_cb, 1);
}

/** Stamp the causal mask onto a strip's masked suffix slots [valid, k_tiles_per_unit) via packer
 *  L1-accumulate (the SDPA apply_causal_mask_lightweight idiom): the head reduction already packed a
 *  finite score into every slot, so each masked tile is one copy_tile + one accumulating pack -- add
 *  the diagonal strict-upper -inf tile on the diagonal slot (its lower triangle keeps the score) and
 *  the full -inf tile on every slot past it. The whole row then stays on the fast pack_untilize strip
 *  instead of dropping to the per-tile W=1 untilize + eltwise add_mask path (which also forced the
 *  valid prefix slow and stacked on the grid-tail cores). Mask idx: 0 = diag strict-upper, 1 = full. */
template <uint32_t acc_cb, uint32_t mask_cb>
inline void stamp_strip_mask(uint32_t slot_base, uint32_t valid, uint32_t k_tile0, uint32_t diag_tile) {
    copy_tile_to_dst_init_short(mask_cb);
    pack_reconfig_l1_acc(1);  // add the mask tile onto the slot's computed score (preserves -inf)
    for (uint32_t c = valid; c < k_tiles_per_unit; ++c) {
        const uint32_t midx = (k_tile0 + c == diag_tile) ? 0u : 1u;
        tile_regs_acquire();
        copy_tile(mask_cb, midx, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, acc_cb, slot_base + c);
        tile_regs_release();
    }
    pack_reconfig_l1_acc(0);
}

/** Untilize a unit's n finished acc tiles into row-major out as one num_blocks=n call: the helper
 *  does the pack_untilize init/uninit bracket ONCE and loops n single-tile blocks internally (no
 *  per-tile reconfig), instead of paying the bracket per tile. Per-tile (W=1) path, used for the
 *  partial-prefix and masked-suffix rows; full-width rows take the fast strip path instead. */
template <uint32_t acc_cb, uint32_t out_cb>
inline void untilize_acc_strip(uint32_t n) {
    compute_kernel_lib::untilize<1, acc_cb, out_cb>(n);
}

/** Restore the half-sync math<->pack DEST contract that the BH fast pack_untilize leaves unrestored
 *  in this (half-sync) kernel: its uninit's math re-sync _llk_math_pack_sync_init_ is compiled out by
 *  `if constexpr (DST_SYNC_MODE != FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE)` (exactly true here,
 *  dst_full_sync_en=false). This re-seeds BOTH sides of the math<->pack semaphore -- the same pair the
 *  full mm_block_init used (llk_math_pack_sync_init + llk_pack_dest_init, matmul.h) -- but drops the
 *  unpack/math/pack hw_configure + matmul/pack init that the next unit's set_matmul_mode re-derives
 *  anyway. Both halves are required: seeding only the math side leaves the pack side's view stale, so
 *  the math thread stalls on the semaphore every resync (a -7pt hit at QC=1/KC=4 where resyncs are
 *  densest). With both, it is the minimal correct resync -- far cheaper than a full block init. */
inline void resync_fast_untilize_dest() {
    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    PACK((llk_pack_dest_init<DST_ACCUM_MODE, PackMode::Default>()));
}

/** Accumulate one full-width row's k_tiles_per_unit output tiles into cb_acc_strip slots
 *  [slot_base, slot_base + k_tiles_per_unit) and stamp its causal mask -- but do NOT untilize.
 *  The caller reserves the whole unit's q_tiles_per_unit*KC strip region up front and untilizes all
 *  rows in ONE fast pack_untilize bracket + one math-sync resync (see produce_full_unit). This
 *  amortizes that fixed per-strip cost over the unit's QC rows instead of paying it per row, so the
 *  compute ceiling tracks the unit area QC*KC rather than KC alone (Proposal 1). */
inline void accumulate_full_strip_row(
    uint32_t r, uint32_t slot_base, uint32_t valid, uint32_t k_tile0, uint32_t diag_tile) {
    if constexpr (qk_col_batch > 1) {
        // Batched mode-switch path: the gate w is column-independent and the unit's whole head
        // group is one resident chunk, so a batch of qk_col_batch k-columns runs all its matmuls
        // (filling cb_qk with cols*qk_batch_heads relu(q.kT) tiles), then all its MAC reductions,
        // paying set_matmul_mode + set_mul_mode once per batch instead of once per column.
        const uint32_t w_base = r * num_heads;  // single chunk (group_start = 0); gate per (head, row)
        for (uint32_t c0 = 0; c0 < k_tiles_per_unit; c0 += qk_col_batch) {
            const uint32_t cols = (c0 + qk_col_batch <= k_tiles_per_unit) ? qk_col_batch : (k_tiles_per_unit - c0);
            set_matmul_mode<cb_q, cb_k, cb_qk>();
            for (uint32_t cc = 0; cc < cols; ++cc) {
                for (uint32_t head = 0; head < num_heads; head += heads_per_dest_pass) {
                    matmul_relu_pass<cb_q, cb_k, cb_qk>(head, r, c0 + cc);
                }
            }
            set_mul_mode<cb_qk, cb_w, cb_acc_strip>();
            for (uint32_t cc = 0; cc < cols; ++cc) {
                // single chunk -> each column's head sum is one MAC pass that seeds its own slot
                mul_accum_chunk<cb_qk, cb_w, cb_acc_strip>(w_base, qk_batch_heads, /*first=*/true, slot_base + c0 + cc);
            }
        }
        pack_reconfig_l1_acc(0);  // done; the unit untilize below packs overwrite
    } else {
        for (uint32_t c = 0; c < k_tiles_per_unit; ++c) {
            accumulate_heads<cb_acc_strip>(r, c, slot_base + c);  // head reduction -> cb_acc_strip slot
        }
    }
    // Stamp the diagonal/-inf mask onto this row's masked suffix in-place, so the whole strip still
    // untilizes via the fast W=KC pack_untilize. valid == k_tiles_per_unit -> fully valid, nothing to mask.
    if (valid < k_tiles_per_unit) {
        stamp_strip_mask<cb_acc_strip, cb_mask>(slot_base, valid, k_tile0, diag_tile);
    }
}

/** Produce a whole FULL unit (q_tiles_per_unit rows x k_tiles_per_unit cols) as ONE batched fast
 *  untilize. Accumulate every row's strip into a contiguous q_tiles_per_unit*KC region of cb_acc_strip
 *  (uniform single push so the fast packer's reads never wrap the ring), then untilize all QC strips
 *  with num_blocks=q_tiles_per_unit -- the pack_untilize init/uninit bracket runs ONCE for the whole
 *  unit, and a single resync_fast_untilize_dest() restores the half-sync math<->pack contract
 *  afterwards (without it dest_offset_id parity drifts and the -inf mask bleeds into valid tiles).
 *  The writer mirrors this: it pops the QC strips in row order. */
inline void produce_full_unit(const WorkUnitSpan& span, uint32_t k_tiles_in_unit) {
    constexpr uint32_t unit_strip = q_tiles_per_unit * k_tiles_per_unit;
    cb_reserve_back(cb_acc_strip, unit_strip);
    for (uint32_t r = 0; r < q_tiles_per_unit; ++r) {
        const uint32_t diag_tile = chunk_start_tiles + span.q_tile_start() + r;
        const uint32_t k_tile0 = span.k_tile_start();
        const uint32_t valid = row_valid_prefix(span.q_tile_start() + r, k_tile0, k_tiles_in_unit);
        accumulate_full_strip_row(r, r * k_tiles_per_unit, valid, k_tile0, diag_tile);
    }
    cb_push_back(cb_acc_strip, unit_strip);
    compute_kernel_lib::untilize<k_tiles_per_unit, cb_acc_strip, cb_out_strip>(q_tiles_per_unit);
    resync_fast_untilize_dest();  // fill the exact math-sync gap the fast-untilize uninit leaves
}

void kernel_main() {
    const uint32_t flat_start = get_arg_val<uint32_t>(0);
    const uint32_t flat_count = get_arg_val<uint32_t>(1);
    if (flat_count == 0) {
        return;
    }

    mm_block_init(
        cb_q, cb_k, cb_qk, 1 /*transpose k*/, 1 /*ct_dim*/, heads_per_dest_pass /*rt_dim*/, head_dim_tiles /*kt_dim*/);
    cb_wait_front(cb_mask, 2);

    WorkUnitSpan span;
    span.start(flat_start);

    bool have_group = false;
    for (uint32_t i = 0; i < flat_count; ++i) {
        if (!have_group) {
            cb_wait_front(cb_w, w_group_tiles);
            if constexpr (!stream_heads) {
                cb_wait_front(cb_q, q_group_tiles);  // all heads form one resident block (heads_per_group == num_heads)
            }
            have_group = true;
        }
        const uint32_t k_tiles_in_unit = span.k_tiles();
        cb_wait_front(cb_k, k_chunk_tiles);  // full chunk pushed even on edge units (ring alignment)

        // A FULL unit (every row spans all KC k-tiles) goes through the batched fast-untilize:
        // accumulate all q_tiles_per_unit rows' strips into cb_acc_strip, then untilize the whole
        // unit under ONE pack_untilize bracket + one math-sync resync (the fixed per-strip cost
        // we amortize over QC*KC, not KC). Masked suffixes are stamped onto the strip slots in-place.
        // Only a partial edge unit (k_tiles_in_unit < KC; the dense schedule never splits Tt unevenly
        // at sp7) or KC==1 falls to the per-tile path below. The writer mirrors this same branch.
        bool produced_full_unit = false;
        if constexpr (use_fast_strip) {
            if (k_tiles_in_unit == k_tiles_per_unit) {
                produce_full_unit(span, k_tiles_in_unit);
                produced_full_unit = true;
            }
        }

        if (!produced_full_unit) {
            // k_tile rises with c, so the causal split is a prefix/suffix: tiles c in [0, m) are
            // fully valid (k_tile < diag_tile), tiles [m, k) are masked (diagonal then future). The
            // valid prefix needs no mask, so produce it then batch-untilize W=1; the masked suffix
            // is rare and uses the immediate add_mask path (add_mask reorders cb_acc one tile alone).
            for (uint32_t r = 0; r < q_tiles_per_unit; ++r) {
                const uint32_t diag_tile = chunk_start_tiles + span.q_tile_start() + r;
                const uint32_t k_tile0 = span.k_tile_start();
                const uint32_t valid = row_valid_prefix(span.q_tile_start() + r, k_tile0, k_tiles_in_unit);

                for (uint32_t c = 0; c < valid; ++c) {
                    cb_reserve_back(cb_acc, 1);
                    accumulate_heads<cb_acc>(r, c, 0);
                    cb_push_back(cb_acc, 1);
                }
                if (valid > 0) {
                    untilize_acc_strip<cb_acc, cb_out>(valid);
                }

                for (uint32_t c = valid; c < k_tiles_in_unit; ++c) {
                    const uint32_t k_tile = k_tile0 + c;
                    cb_reserve_back(cb_acc, 1);
                    accumulate_heads<cb_acc>(r, c, 0);
                    cb_push_back(cb_acc, 1);
                    add_mask<cb_acc, cb_mask>(k_tile == diag_tile ? 0 : 1);
                    compute_kernel_lib::untilize<1, cb_acc, cb_out>(1);
                }
            }
        }
        cb_pop_front(cb_k, k_chunk_tiles);

        if (span.advance()) {
            cb_pop_front(cb_w, w_group_tiles);
            if constexpr (!stream_heads) {
                cb_pop_front(cb_q, q_group_tiles);
            }
            have_group = false;
        }
    }
}
