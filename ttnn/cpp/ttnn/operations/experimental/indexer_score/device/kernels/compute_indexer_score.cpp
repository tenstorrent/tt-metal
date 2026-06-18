// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute for indexer_score. Per work unit (QC q-rows x KC k-cols):
//   acc[r,c] = sum_h relu(q[h,row] @ k[col]^T) * w[h,row], then causal -inf mask,
//   then pack_untilize -> bf16 row-major out.
// Heads stream in groups (heads_per_dest_pass rows per DEST pass, half-sync bf16 DEST);
// q/w stay resident when all heads fit.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"  // Device 2.0 CircularBuffer wrapper (cb ops)

#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "indexer_score_common.hpp"  // shared CB indices, compile-time dims, work-unit walk
#include "api/compute/experimental/indexer_mul_custom.h"

// qk subblock height (head rows per DEST pass); first per-kernel compile-time arg.
constexpr uint32_t heads_per_dest_pass = get_compile_time_arg_val(num_common_ct_args);
// heads buffered in cb_qk per matmul/mul phase chunk (multiple of HP).
constexpr uint32_t qk_batch_heads = get_compile_time_arg_val(num_common_ct_args + 1);
// k-columns batched per matmul<->mul mode switch in the full-strip path (1 = per-column, no batching).
constexpr uint32_t qk_col_batch = get_compile_time_arg_val(num_common_ct_args + 2);

// k-cols sharing ONE dest acquire in the blocked-custom mul; bounded by the dest budget. One unpack
// context per head loads w[h] + ct_dim qk cols, so unpack-context sync is paid 1/ct_dim as often as
// the per-tile bcast-mul (the mul-phase bottleneck).
constexpr uint32_t mul_ct_dim = (k_tiles_per_unit < heads_per_dest_pass) ? k_tiles_per_unit : heads_per_dest_pass;

// Head reduction splits into a matmul phase (fill cb_qk with relu(q.kT)) and a mul phase (gate +
// head-reduce into the accumulator) so the matmul<->eltwise reconfig happens once per batch, not per
// head pass. Two implementations of that split live below: the head-major path
// (matmul_phase + mul_phase, blocked-custom MUL, qk_col_batch > 1) and the per-column streaming
// FALLBACK (accumulate_heads, qk_col_batch == 1). The set_*/set_mul_* helpers are shared by both.

/** srcA<-k, srcB<-q, matmul mode for the matmul phase. Reconfig order matters: matmul maps
 *  in1->srcA, in0->srcB, so swapping the reconfig misreads bfp8 k (bf16 k is unaffected). */
template <uint32_t q_cb, uint32_t k_cb, uint32_t qk_cb>
inline void set_matmul_mode() {
    // guarded reconfig: srcA qk->k changes format (fires); srcB w->q is bf16->bf16 (skipped).
    reconfig_data_format(qk_cb, k_cb, cb_w, q_cb);
    // guarded: no-op in the bf16 path; only the fp32-dest fallback (qk=fp32, out=bf16) reconfigs.
    pack_reconfig_data_format(cb_out_strip, qk_cb);
    pack_reconfig_l1_acc(0);  // cb_qk packs overwrite (mul phase turns L1-acc on for cb_acc_strip)
    mm_block_init_short(
        q_cb, k_cb, 1 /*transpose k*/, 1 /*ct_dim*/, heads_per_dest_pass /*rt_dim*/, head_dim_tiles /*kt_dim*/);
    pack_relu_config(ReluConfig::zero());  // relu in the packer for the whole matmul phase
}

/** Matmul one DEST pass of relu(q.kT): HP head-rows of q row r vs k col c, left in DEST (caller packs).
 *  q blocks are [QC][HG][Dt] so head rows stride head_dim_tiles. Assumes set_matmul_mode() ran. */
template <uint32_t q_cb, uint32_t k_cb>
inline void emit_qk_matmul_block(uint32_t head_in_group, uint32_t r, uint32_t c) {
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
}

/** One matmul-phase DEST pass: relu(q row r @ k col c^T) block-packed into cb_qk front (tile-major). */
template <uint32_t q_cb, uint32_t k_cb, uint32_t qk_cb>
void matmul_relu_pass(uint32_t head_in_group, uint32_t r, uint32_t c) {
    CircularBuffer qk(qk_cb);
    emit_qk_matmul_block<q_cb, k_cb>(head_in_group, r, c);
    qk.reserve_back(heads_per_dest_pass);
    tile_regs_wait();
    pack_tile_block(0, qk_cb, heads_per_dest_pass);
    tile_regs_release();
    qk.push_back(heads_per_dest_pass);
}

/** srcA<-qk, srcB<-w + pack format for a mul+accumulate phase (shared by the standard and
 *  blocked-custom mul; only the following bcast-mul init differs). */
template <uint32_t qk_cb, uint32_t w_cb, uint32_t acc_cb>
inline void set_mul_reconfig() {
    pack_relu_config(ReluConfig::none());  // accumulator packs stay linear (negative gates)
    // guarded reconfig: srcA k->qk changes format (fires); srcB q->w is bf16->bf16 (skipped).
    reconfig_data_format(cb_k, qk_cb, cb_q, w_cb);
    pack_reconfig_data_format(qk_cb, acc_cb);  // guarded: qk and acc share acc_fmt -> no-op.
}

/** srcA<-qk, srcB<-w + bcast-mul mode for the mul+accumulate phase. */
template <uint32_t qk_cb, uint32_t w_cb, uint32_t acc_cb>
inline void set_mul_mode() {
    set_mul_reconfig<qk_cb, w_cb, acc_cb>();
    mul_bcast_cols_init_short(qk_cb, w_cb);
    // acc_to_dest=1: each mul MACs onto the same DEST tile (tile_regs_acquire zeroes it, head 0
    // seeds), so the chunk's head reduction needs one pack, not a per-head packer-L1-acc RMW.
    MATH((llk_math_eltwise_binary_init<ckernel::EltwiseBinaryType::ELWMUL, ckernel::BroadcastType::COL, MATH_FIDELITY>(
        qk_cb, w_cb, 1 /*acc_to_dest*/)));
}

/** Mul+accumulate `chunk_heads` resident heads onto the output tile via hw MAC:
 *  dst0 += sum_h relu(qk[h]) * w[w_base+h], packed once to acc_cb[acc_slot]. `first` overwrites the
 *  slot (l1_acc off); later chunks L1-accumulate. Caller owns acc_cb and the l1_acc reset. */
template <uint32_t qk_cb, uint32_t w_cb, uint32_t acc_cb>
void mul_accum_chunk(uint32_t w_base, uint32_t chunk_heads, bool first, uint32_t acc_slot) {
    CircularBuffer qk(qk_cb);
    qk.wait_front(chunk_heads);
    tile_regs_acquire();
    for (uint32_t h = 0; h < chunk_heads; ++h) {
        mul_tiles_bcast_cols(qk_cb, w_cb, h, w_base + h, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_l1_acc(first ? 0 : 1);  // first chunk seeds (overwrite), later chunks accumulate
    pack_tile<true>(0, acc_cb, acc_slot);
    tile_regs_release();
    qk.pop_front(chunk_heads);
}

/** Matmul DEST pass packed HEAD-MAJOR: head h's `cols` columns land contiguous in cb_qk (slot
 *  (head+d)*cols + col_in_batch), the layout the blocked mul streams as SrcA. Uses llk_matmul_pack
 *  out_of_order (generic pack misreads the matmul DEST layout). Caller reserves cols*num_heads once. */
template <uint32_t q_cb, uint32_t k_cb, uint32_t qk_cb>
void matmul_relu_pass_headmajor(uint32_t head_in_group, uint32_t r, uint32_t c, uint32_t col_in_batch, uint32_t cols) {
    emit_qk_matmul_block<q_cb, k_cb>(head_in_group, r, c);
    tile_regs_wait();
    for (uint32_t d = 0; d < heads_per_dest_pass; ++d) {
        PACK((llk_matmul_pack<DST_ACCUM_MODE, true /*out_of_order*/, PackMode::Default>(
            d, qk_cb, 1 /*ntiles*/, (head_in_group + d) * cols + col_in_batch)));  // relu(q.kT), head-major slot
    }
    tile_regs_release();
}

/** Like set_mul_mode but with the blocked-custom bcast-col MUL (one unpack context per head: w[h]
 *  loaded once + ct_dim qk cols, MAC-reduced in dest). Same reconfig. */
template <uint32_t qk_cb, uint32_t w_cb, uint32_t acc_cb>
inline void set_mul_mode_custom() {
    set_mul_reconfig<qk_cb, w_cb, acc_cb>();
    mul_bcast_cols_init_short_custom(qk_cb, w_cb);
}

/** Per-column (qk_col_batch==1) head reduction for output tile (r,c): acc_cb[acc_slot] =
 *  sum_h relu(q[h,r].k[c]^T)*w[h,r], MAC'd in DEST per chunk, L1-acc only across chunks. Caller
 *  owns acc_cb. */
template <uint32_t acc_cb>
inline void accumulate_heads(uint32_t r, uint32_t c, uint32_t acc_slot) {
    CircularBuffer q(cb_q);
    bool first = true;
    for (uint32_t group_start = 0; group_start < num_heads; group_start += heads_per_group) {
        if constexpr (stream_heads) {
            q.wait_front(q_group_tiles);
        }
        // per chunk (cb_qk capacity): run its matmuls, then MAC the chunk's head sum into DEST once,
        // so the matmul<->eltwise reinit and the acc pack happen per chunk, not per head
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
            q.pop_front(q_group_tiles);
        }
    }
    pack_reconfig_l1_acc(0);  // done accumulating; downstream packs (mask, untilize) overwrite
}

/** Stamp the causal mask for absolute column `k_tile` onto acc write slot `slot` (written into the
 *  already-reserved acc_cb slot before its push_back, no repush). Two cases:
 *   - diagonal (k_tile == diag_tile): L1-ACCUMULATE the strict-upper -inf tile, keeping the lower tri.
 *   - past diagonal (incl. pad cols >= Tt): OVERWRITE with -inf, so stale-k garbage in a pad column is
 *     discarded rather than turned to nan by `garbage + -inf`. */
template <uint32_t acc_cb, uint32_t mask_cb>
inline void stamp_mask_tile(uint32_t slot, uint32_t k_tile, uint32_t diag_tile) {
    const bool is_diag = (k_tile == diag_tile);
    const uint32_t midx = is_diag ? 0u : 1u;  // 0 = diag strict-upper -inf, 1 = full -inf
    copy_tile_to_dst_init_short(mask_cb);
    pack_reconfig_l1_acc(is_diag ? 1 : 0);  // diag accumulates (keeps score); full -inf overwrites
    tile_regs_acquire();
    copy_tile(mask_cb, midx, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(0, acc_cb, slot);
    tile_regs_release();
    pack_reconfig_l1_acc(0);
}

// ---------------------------------------------------------------------------------------------------
// The three per-unit phases, called in order from kernel_main:
//   PHASE 1 matmul_phase  -- fill cb_qk with relu(q.kT) for a batch of a row's k-columns
//   PHASE 2 mul_phase     -- gate-scale + head-reduce that batch into the unit accumulator
//   PHASE 3 (inline)      -- untilize the whole accumulated QC x KC unit in one pack_untilize bracket
// Phases 1/2 run per k-column batch: cb_qk holds one batch, so they alternate. When the batch is the
// whole row (qk_col_batch == KC, heads8/16) that's one matmul + one mul per row; heads64
// (qk_col_batch=2 < KC) interleaves per 2-column batch.
// ---------------------------------------------------------------------------------------------------

/** PHASE 1 -- fill cb_qk HEAD-MAJOR with relu(q[:,r].k[col_base..]^T) for one k-col batch of row r.
 *  One set_matmul_mode for the batch (reinit hoisted out of the col loop), one cb_qk reserve/push. */
inline void matmul_phase(uint32_t r, uint32_t col_base, uint32_t cols) {
    const uint32_t batch_tiles = cols * num_heads;
    CircularBuffer qk(cb_qk);
    set_matmul_mode<cb_q, cb_k, cb_qk>();
    qk.reserve_back(batch_tiles);
    for (uint32_t col_in_batch = 0; col_in_batch < cols; ++col_in_batch) {
        for (uint32_t head = 0; head < num_heads; head += heads_per_dest_pass) {
            matmul_relu_pass_headmajor<cb_q, cb_k, cb_qk>(head, r, col_base + col_in_batch, col_in_batch, cols);
        }
    }
    qk.push_back(batch_tiles);
}

/** PHASE 2 -- gate-multiply the batch's cb_qk by w and head-reduce into cb_acc_strip via the blocked
 *  bcast-col MUL. Per ct_dim-col sub-batch, one dest acquire holds the column accumulators; each head
 *  is one unpack context (w[h] once + ct_dim cols) that MACs onto dest[0..n_cols), so unpack-context
 *  sync is per head, not per (col, head). ELWMUL accumulates in dest -> one pack per column. cb_qk is
 *  head-major (head h's cols contiguous); whole batch shares one set_mul_mode (w is column-independent). */
inline void mul_phase(uint32_t r, uint32_t slot_base, uint32_t col_base, uint32_t cols) {
    const uint32_t batch_tiles = cols * num_heads;
    const uint32_t w_base = r * num_heads;  // single chunk (group_start = 0); gate per (head, row)
    // gates consumed only here: wait now (after this batch's matmuls) so the reader reads w behind the
    // latency-critical q/k. Cumulative wait -> no-op once the resident group is in.
    CircularBuffer qk(cb_qk);
    CircularBuffer(cb_w).wait_front(w_group_tiles);  // single use here; gates popped in kernel_main
    set_mul_mode_custom<cb_qk, cb_w, cb_acc_strip>();
    qk.wait_front(batch_tiles);
    for (uint32_t sub_base = 0; sub_base < cols; sub_base += mul_ct_dim) {
        const uint32_t n_cols = (sub_base + mul_ct_dim <= cols) ? mul_ct_dim : (cols - sub_base);
        tile_regs_acquire();
        for (uint32_t h = 0; h < num_heads; ++h) {
            mul_tiles_bcast_cols_custom(cb_qk, cb_w, h * cols + sub_base, w_base + h, 0, n_cols);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < n_cols; ++j) {
            pack_tile(j, cb_acc_strip, slot_base + col_base + sub_base + j);
        }
        tile_regs_release();
    }
    qk.pop_front(batch_tiles);
}

/** PHASE 1+2 fallback for head-streaming / KC==1 (qk_col_batch == 1): each k-col runs accumulate_heads
 *  (matmul + STANDARD bcast-mul interleaved per head group). It reads cb_w by index, so wait the gates
 *  here (k already waited in kernel_main). */
inline void accumulate_row_streaming(uint32_t r, uint32_t slot_base) {
    CircularBuffer(cb_w).wait_front(w_group_tiles);
    for (uint32_t c = 0; c < k_tiles_per_unit; ++c) {
        accumulate_heads<cb_acc_strip>(r, c, slot_base + c);  // head reduction -> cb_acc_strip slot
    }
}

/** Stamp the causal -inf mask onto row r's masked suffix [valid, KC) in place, so the strip still
 *  untilizes via the fast W=KC path (empty loop when the row is fully valid). */
inline void stamp_masked_suffix(const WorkUnitSpan& span, uint32_t r, uint32_t slot_base, uint32_t k_tiles_in_unit) {
    const uint32_t k_tile0 = span.k_tile_start();
    const uint32_t diag_tile = chunk_start_tiles + span.q_tile_start() + r;
    const uint32_t valid = row_valid_prefix(span.q_tile_start() + r, k_tile0, k_tiles_in_unit);
    for (uint32_t c = valid; c < k_tiles_per_unit; ++c) {
        stamp_mask_tile<cb_acc_strip, cb_mask>(slot_base + c, k_tile0 + c, diag_tile);
    }
}

void kernel_main() {
    const uint32_t flat_start = get_arg_val<uint32_t>(0);
    const uint32_t flat_count = get_arg_val<uint32_t>(1);
    if (flat_count == 0) {
        return;
    }

    mm_block_init(
        cb_q, cb_k, cb_qk, 1 /*transpose k*/, 1 /*ct_dim*/, heads_per_dest_pass /*rt_dim*/, head_dim_tiles /*kt_dim*/);
    CircularBuffer(cb_mask).wait_front(num_mask_tiles);  // single use; the mask CB is never popped

    // CBs touched twice below (wait/pop, reserve/push) get one instance each.
    CircularBuffer k(cb_k);
    CircularBuffer acc(cb_acc_strip);
    CircularBuffer q(cb_q);

    WorkUnitSpan span;
    span.start(flat_start);

    constexpr uint32_t unit_strip = q_tiles_per_unit * k_tiles_per_unit;  // QC x KC accumulator slots
    constexpr uint32_t q_row_tiles = q_group_tiles / q_tiles_per_unit;    // heads_per_group * head_dim_tiles

    // One QC x KC unit per iteration. Every unit spans KC k-tiles; the dense schedule may leave a
    // partial last unit (valid < KC), masked in stamp_masked_suffix.
    //
    // No whole-block q/w wait here: resident q is waited PER ROW below (reader pushes a row at a time, so
    // row 0 runs while row 1 drains); w is waited in the mul phase. Only k is waited up front.
    for (uint32_t i = 0; i < flat_count; ++i) {
        k.wait_front(k_chunk_tiles);
        const uint32_t k_tiles_in_unit = span.k_tiles();

        acc.reserve_back(unit_strip);
        for (uint32_t r = 0; r < q_tiles_per_unit; ++r) {
            // wait q rows 0..r only (reader pushes per row): row r reads only its row, so row 0 runs
            // while row 1 arrives. Cumulative + non-consuming -> immediate once q is resident.
            if constexpr (!stream_heads) {
                q.wait_front((r + 1) * q_row_tiles);
            }
            const uint32_t slot_base = r * k_tiles_per_unit;

            // PHASE 1 (matmul) + PHASE 2 (mul) per k-col batch. cb_qk holds one batch, so they
            // alternate; GLM5/DSv32 (heads8/16) = one of each per row.
            if constexpr (qk_col_batch > 1) {
                for (uint32_t col_base = 0; col_base < k_tiles_per_unit; col_base += qk_col_batch) {
                    const uint32_t cols =
                        (col_base + qk_col_batch <= k_tiles_per_unit) ? qk_col_batch : (k_tiles_per_unit - col_base);
                    matmul_phase(r, col_base, cols);          // PHASE 1: relu(q.kT) -> cb_qk (head-major)
                    mul_phase(r, slot_base, col_base, cols);  // PHASE 2: gate-mul + head-reduce -> cb_acc_strip
                }
            } else {
                accumulate_row_streaming(r, slot_base);  // PHASE 1+2 head-streaming / KC==1 fallback
            }

            stamp_masked_suffix(span, r, slot_base, k_tiles_in_unit);  // causal -inf on the row's masked suffix
        }
        acc.push_back(unit_strip);

        // PHASE 3 -- untilize all QC strips in ONE pack_untilize bracket (cost amortizes over QC*KC).
        compute_kernel_lib::untilize<k_tiles_per_unit, cb_acc_strip, cb_out_strip>(q_tiles_per_unit);

        k.pop_front(k_chunk_tiles);
        if (span.advance()) {
            CircularBuffer(cb_w).pop_front(w_group_tiles);  // single use; gates waited in the mul phase
            if constexpr (!stream_heads) {
                q.pop_front(q_group_tiles);
            }
        }
    }
}
