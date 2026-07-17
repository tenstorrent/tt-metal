// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute for indexer_score. Per work unit (QC q-rows x KC k-cols), per output plane:
//   acc = sum_{h in group} act(q[h]@k^T)*w[h], causal -inf mask, then untilize (or block-max-pool).
//   act = relu when apply_relu, else identity (raw dot). num_out_groups==1 sums all heads -> 1 plane;
//   >1 keeps the groups separate. Heads stream in DEST passes (half-sync bf16); q/w resident when they fit.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reduce.h"            // block-max-pool: PoolType / ReduceDim enums
#include "api/compute/reduce_custom.h"     // block-max-pool: batched reduce_block_max_row (scaler-resident)
#include "api/dataflow/circular_buffer.h"  // Device 2.0 CircularBuffer wrapper (cb ops)

#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"  // block-max-pool: compute_kernel_lib::reduce
#include "indexer_score_common.hpp"  // shared CB indices, compile-time dims, work-unit walk
#include "api/compute/experimental/indexer_mul_custom.h"

// qk subblock height (head rows per DEST pass).
constexpr uint32_t heads_per_dest_pass = get_compile_time_arg_val(num_common_ct_args);
// heads buffered in cb_qk per matmul/mul phase chunk (multiple of HP).
constexpr uint32_t qk_batch_heads = get_compile_time_arg_val(num_common_ct_args + 1);
// k-cols batched per matmul<->mul mode switch in the full-strip path (1 = per-column).
constexpr uint32_t qk_col_batch = get_compile_time_arg_val(num_common_ct_args + 2);
// 1 = relu(q.kT) before the gate-mul; 0 = raw q.kT (no relu).
constexpr bool apply_relu = get_compile_time_arg_val(num_common_ct_args + 3) != 0;

// Fused single-head path (one head/plane, no relu, bf16 q; host-decided). Gate w folded onto q up front
// ((w*q).k == w*(q.k), valid without relu) so the matmul writes the gated score straight to the
// accumulator -- no cb_qk staging, no gate-mul phase. reduce_heads==1 keeps the per-row gate exact.
constexpr bool fuse_single = get_compile_time_arg_val(num_common_ct_args + 4) != 0;
// Fused + no mcast: stream k (waited incrementally in the matmul). Fused + mcast: k is one block, waited whole.
constexpr bool fused_stream_k = get_compile_time_arg_val(num_common_ct_args + 5) != 0;

// k-cols sharing ONE dest acquire in the blocked-custom mul (dest-bounded). One unpack context per head
// (w[h] + ct_dim qk cols), so unpack-context sync is paid 1/ct_dim of the per-tile bcast-mul rate.
constexpr uint32_t mul_ct_dim = (k_tiles_per_unit < heads_per_dest_pass) ? k_tiles_per_unit : heads_per_dest_pass;

// Head reduction splits into a matmul phase (fill cb_qk) and a mul phase (gate + head-reduce) so the
// matmul<->eltwise reconfig is per batch, not per head. Two impls: head-major (matmul_phase + mul_phase,
// blocked-custom MUL, qk_col_batch > 1) and per-column streaming FALLBACK (accumulate_heads,
// qk_col_batch == 1). The set_*/set_mul_* helpers are shared.

/** srcA<-k, srcB<-q matmul mode. Reconfig order matters: matmul maps in1->srcA, in0->srcB, so swapping
 *  it misreads bfp8 k (bf16 k unaffected). */
template <uint32_t q_cb, uint32_t k_cb, uint32_t qk_cb>
inline void set_matmul_mode() {
    // guarded: srcA qk->k fires; srcB w->q is bf16->bf16 (skipped).
    reconfig_data_format(qk_cb, k_cb, cb_w, q_cb);
    // guarded: no-op in bf16; only the fp32-dest fallback reconfigs.
    pack_reconfig_data_format(cb_out_strip, qk_cb);
    pack_reconfig_l1_acc(0);  // cb_qk packs overwrite
    matmul_block_init(
        q_cb, k_cb, 1 /*transpose k*/, 1 /*ct_dim*/, heads_per_dest_pass /*rt_dim*/, head_dim_tiles /*kt_dim*/);
    // relu(q.kT) in the packer when apply_relu; else linear, so the raw dot flows to the gate-mul.
    if constexpr (apply_relu) {
        pack_relu_config(ReluConfig::zero());
    } else {
        pack_relu_config(ReluConfig::none());
    }
}

/** One DEST pass of q.kT (relu'd in the packer when apply_relu): HP head-rows of q_row vs k_col, left in
 *  DEST (caller packs). q blocks are [QC][HG][Dt] so head rows stride head_dim_tiles. Assumes set_matmul_mode(). */
template <uint32_t q_cb, uint32_t k_cb>
inline void emit_qk_matmul_block(uint32_t head_in_group, uint32_t q_row, uint32_t k_col) {
    tile_regs_acquire();
    const uint32_t q_base = (q_row * heads_per_group + head_in_group) * head_dim_tiles;
    for (uint32_t dim_tile = 0; dim_tile < head_dim_tiles; ++dim_tile) {
        matmul_block(
            q_cb,
            k_cb,
            q_base + dim_tile,
            k_col * head_dim_tiles + dim_tile,
            0,
            1 /*transpose k*/,
            1,
            heads_per_dest_pass,
            head_dim_tiles);
    }
    tile_regs_commit();
}

/** One matmul-phase DEST pass: act(q-row q_row @ k-col k_col^T) block-packed into cb_qk front. */
template <uint32_t q_cb, uint32_t k_cb, uint32_t qk_cb>
void matmul_relu_pass(uint32_t head_in_group, uint32_t q_row, uint32_t k_col) {
    CircularBuffer qk(qk_cb);
    emit_qk_matmul_block<q_cb, k_cb>(head_in_group, q_row, k_col);
    qk.reserve_back(heads_per_dest_pass);
    tile_regs_wait();
    pack_tile_block(0, qk_cb, heads_per_dest_pass);
    tile_regs_release();
    qk.push_back(heads_per_dest_pass);
}

/** srcA<-qk, srcB<-w + pack format for a mul+accumulate phase (shared; only the bcast-mul init differs). */
template <uint32_t qk_cb, uint32_t w_cb, uint32_t acc_cb>
inline void set_mul_reconfig() {
    pack_relu_config(ReluConfig::none());  // accumulator packs stay linear (negative gates)
    // guarded: srcA k->qk fires; srcB q->w is bf16->bf16 (skipped).
    reconfig_data_format(cb_k, qk_cb, cb_q, w_cb);
    pack_reconfig_data_format(qk_cb, acc_cb);  // guarded: qk/acc share acc_fmt -> no-op.
}

/** srcA<-qk, srcB<-w + bcast-mul mode for the mul+accumulate phase. */
template <uint32_t qk_cb, uint32_t w_cb, uint32_t acc_cb>
inline void set_mul_mode() {
    set_mul_reconfig<qk_cb, w_cb, acc_cb>();
    mul_bcast_cols_init_short(qk_cb, w_cb);
    // acc_to_dest=1: each mul MACs onto the same DEST tile (head 0 seeds the acquire-zeroed reg), so the
    // chunk's head reduction needs one pack, not a per-head packer-L1-acc RMW.
    MATH((llk_math_eltwise_binary_init<ckernel::EltwiseBinaryType::ELWMUL, ckernel::BroadcastType::COL, MATH_FIDELITY>(
        qk_cb, w_cb, 1 /*acc_to_dest*/)));
}

/** Mul+accumulate `chunk_heads` resident heads via hw MAC: dst0 += sum_h qk[h]*w[w_base+h], packed once
 *  to acc_cb[acc_slot]. `first` overwrites the slot (l1_acc off); later chunks L1-accumulate. */
template <uint32_t qk_cb, uint32_t w_cb, uint32_t acc_cb>
void mul_accum_chunk(uint32_t w_base, uint32_t chunk_heads, bool first, uint32_t acc_slot) {
    CircularBuffer qk(qk_cb);
    qk.wait_front(chunk_heads);
    tile_regs_acquire();
    for (uint32_t head = 0; head < chunk_heads; ++head) {
        mul_tiles_bcast_cols(qk_cb, w_cb, head, w_base + head, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_l1_acc(first ? 0 : 1);  // first chunk overwrites, later chunks accumulate
    pack_tile<true>(0, acc_cb, acc_slot);
    tile_regs_release();
    qk.pop_front(chunk_heads);
}

/** act(q.kT) DEST pass packed HEAD-MAJOR: head's `cols` cols land contiguous in cb_qk at slot
 *  (pack_head+d)*cols + col_in_batch (the layout the blocked mul streams as SrcA). q_head = global head
 *  for the matmul; pack_head = its index in the output group. Needs llk_matmul_pack out_of_order (generic
 *  pack misreads the matmul DEST layout). Caller reserves cols*reduce_heads once. */
template <uint32_t q_cb, uint32_t k_cb, uint32_t qk_cb>
void matmul_relu_pass_headmajor(
    uint32_t q_head, uint32_t pack_head, uint32_t r, uint32_t c, uint32_t col_in_batch, uint32_t cols) {
    emit_qk_matmul_block<q_cb, k_cb>(q_head, r, c);
    tile_regs_wait();
    for (uint32_t d = 0; d < heads_per_dest_pass; ++d) {
        PACK((llk_matmul_pack<DST_ACCUM_MODE, true /*out_of_order*/, PackMode::Default>(
            d,
            qk_cb,
            1 /*ntiles*/,
            (pack_head + d) * cols + col_in_batch)));  // act(q.kT), head-major slot
    }
    tile_regs_release();
}

/** Like set_mul_mode but with the blocked-custom bcast-col MUL (one unpack context per head: w[h] once +
 *  ct_dim qk cols, MAC-reduced in dest). Same reconfig. */
template <uint32_t qk_cb, uint32_t w_cb, uint32_t acc_cb>
inline void set_mul_mode_custom() {
    set_mul_reconfig<qk_cb, w_cb, acc_cb>();
    mul_bcast_cols_init_short_custom(qk_cb, w_cb);
}

/** Per-column (qk_col_batch==1) head reduction for tile (q_row, k_col): acc_cb[acc_slot] =
 *  sum_h act(q[h,q_row].k[k_col]^T)*w[h,q_row], MAC'd in DEST per chunk, L1-acc only across chunks. */
template <uint32_t acc_cb>
inline void accumulate_heads(uint32_t q_row, uint32_t k_col, uint32_t acc_slot) {
    CircularBuffer q(cb_q);
    bool first = true;
    for (uint32_t group_start = 0; group_start < num_heads; group_start += heads_per_group) {
        if constexpr (stream_heads) {
            q.wait_front(q_group_tiles);
        }
        // per chunk (cb_qk capacity): matmul it, then MAC its head sum into DEST once -- so the
        // matmul<->eltwise reinit and the acc pack are per chunk, not per head
        for (uint32_t chunk = 0; chunk < heads_per_group; chunk += qk_batch_heads) {
            const uint32_t chunk_end = chunk + qk_batch_heads;
            set_matmul_mode<cb_q, cb_k, cb_qk>();
            for (uint32_t head = chunk; head < chunk_end; head += heads_per_dest_pass) {
                matmul_relu_pass<cb_q, cb_k, cb_qk>(head, q_row, k_col);
            }
            set_mul_mode<cb_qk, cb_w, acc_cb>();
            // w is laid out [q_tiles_per_unit][num_heads] (see reader read_w_group)
            const uint32_t w_base = q_row * num_heads + group_start + chunk;
            mul_accum_chunk<cb_qk, cb_w, acc_cb>(w_base, qk_batch_heads, first, acc_slot);
            first = false;
        }
        if constexpr (stream_heads) {
            q.pop_front(q_group_tiles);
        }
    }
    pack_reconfig_l1_acc(0);  // done accumulating; downstream packs (mask, untilize) overwrite
}

/** Stamp the causal mask for absolute column `k_tile` onto acc slot `slot` (in place, no repush):
 *   - diagonal (k_tile == diag_tile): L1-ACCUMULATE the strict-upper -inf tile, keeping the lower tri.
 *   - past diagonal (incl. pad cols >= Tt): OVERWRITE with -inf, so stale-k garbage is discarded rather
 *     than turned to nan by `garbage + -inf`. */
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
// The three per-unit phases (in order from kernel_main):
//   PHASE 1 matmul_phase -- fill cb_qk with act(q.kT) for a batch of a row's k-cols
//   PHASE 2 mul_phase    -- gate-scale + head-reduce that batch into the accumulator
//   PHASE 3 (inline)     -- untilize the whole QC x KC unit in one pack_untilize bracket
// Phases 1/2 alternate per k-col batch (cb_qk holds one batch). A whole-row batch (qk_col_batch==KC) does
// one of each per row; a smaller batch (qk_col_batch < KC) interleaves per batch.
// ---------------------------------------------------------------------------------------------------

/** PHASE 1 -- fill cb_qk HEAD-MAJOR with act(q[:,q_row].k[col_base..]^T) for one k-col batch of q_row.
 *  One set_matmul_mode for the batch, one cb_qk reserve/push. */
inline void matmul_phase(uint32_t r, uint32_t col_base, uint32_t cols, uint32_t head_base) {
    const uint32_t batch_tiles = cols * reduce_heads;
    CircularBuffer qk(cb_qk);
    set_matmul_mode<cb_q, cb_k, cb_qk>();
    qk.reserve_back(batch_tiles);
    for (uint32_t col_in_batch = 0; col_in_batch < cols; ++col_in_batch) {
        // hl walks this group's heads [head_base, +reduce_heads), packed head-major at hl.
        for (uint32_t hl = 0; hl < reduce_heads; hl += heads_per_dest_pass) {
            matmul_relu_pass_headmajor<cb_q, cb_k, cb_qk>(
                head_base + hl, hl, r, col_base + col_in_batch, col_in_batch, cols);
        }
    }
    qk.push_back(batch_tiles);
}

/** PHASE 2 -- gate-multiply the batch's cb_qk by w and head-reduce into cb_acc_strip via the blocked
 *  bcast-col MUL. Per ct_dim-col sub-batch, one dest acquire holds the col accumulators; each head is one
 *  unpack context (w[h] once + ct_dim cols) MAC'ing onto dest[0..n_cols), so unpack-context sync is per
 *  head, not per (col, head). cb_qk is head-major; whole batch shares one set_mul_mode (w col-independent). */
inline void mul_phase(uint32_t r, uint32_t slot_base, uint32_t col_base, uint32_t cols, uint32_t head_base) {
    const uint32_t batch_tiles = cols * reduce_heads;
    // gate per (head, row); this group's heads are [head_base, +reduce_heads) of row r.
    const uint32_t w_base = r * num_heads + head_base;
    // gates used only here: wait now (after the matmuls) so the reader reads w behind the latency-critical
    // q/k. Cumulative -> no-op once the resident group is in.
    CircularBuffer qk(cb_qk);
    CircularBuffer(cb_w).wait_front(w_group_tiles);  // gates popped in kernel_main
    set_mul_mode_custom<cb_qk, cb_w, cb_acc_strip>();
    qk.wait_front(batch_tiles);
    for (uint32_t sub_base = 0; sub_base < cols; sub_base += mul_ct_dim) {
        const uint32_t n_cols = (sub_base + mul_ct_dim <= cols) ? mul_ct_dim : (cols - sub_base);
        tile_regs_acquire();
        for (uint32_t hl = 0; hl < reduce_heads; ++hl) {
            mul_tiles_bcast_cols_custom(cb_qk, cb_w, hl * cols + sub_base, w_base + hl, 0, n_cols);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t out_col = 0; out_col < n_cols; ++out_col) {
            pack_tile(out_col, cb_acc_strip, slot_base + col_base + sub_base + out_col);
        }
        tile_regs_release();
    }
    qk.pop_front(batch_tiles);
}

/** PHASE 1+2 fallback for head-streaming / KC==1 (qk_col_batch == 1): each k-col runs accumulate_heads
 *  (matmul + STANDARD bcast-mul per head group). Reads cb_w by index, so wait the gates here. */
inline void accumulate_row_streaming(uint32_t q_row, uint32_t slot_base) {
    CircularBuffer(cb_w).wait_front(w_group_tiles);
    for (uint32_t k_col = 0; k_col < k_tiles_per_unit; ++k_col) {
        accumulate_heads<cb_acc_strip>(q_row, k_col, slot_base + k_col);
    }
}

/** Streaming pad: a phantom band carries no k/output -- it only keeps the row's q-mcast in lockstep (the
 *  reader re-issues the band-independent q reads on short columns). Drain exactly the q blocks the reader
 *  pushed for one band -- QC*KC tiles x (Hi/HB) head groups -- without compute. Mirrors the reader's loop. */
inline void drain_phantom_band_q() {
    CircularBuffer q(cb_q);
    for (uint32_t tile_idx = 0; tile_idx < q_tiles_per_unit * k_tiles_per_unit; ++tile_idx) {
        for (uint32_t group_start = 0; group_start < num_heads; group_start += heads_per_group) {
            q.wait_front(q_group_tiles);
            q.pop_front(q_group_tiles);
        }
    }
}

/** Fused single-head: scale resident q in place by gate w (per-query, bcast over head-dim).
 *  q tile (head_base, row r, dim d) *= w[r, head_base]. One mul per q-tile instead of the per-output
 *  gate-mul over KC columns. */
inline void scale_q_by_w_inplace(uint32_t head_base) {
    CircularBuffer q(cb_q);
    CircularBuffer w(cb_w);
    q.wait_front(q_group_tiles);
    w.wait_front(w_group_tiles);
    pack_relu_config(ReluConfig::none());
    reconfig_data_format(cb_k, cb_q, cb_acc_strip, cb_w);  // srcA->q, srcB->w (guarded)
    pack_reconfig_data_format(cb_out_strip, cb_q);         // pack->cb_q (bf16, in place)
    mul_bcast_cols_init_short(cb_q, cb_w);
    for (uint32_t r = 0; r < q_tiles_per_unit; ++r) {
        const uint32_t w_idx = r * num_heads + head_base;
        for (uint32_t d = 0; d < head_dim_tiles; ++d) {
            const uint32_t q_idx = (r * heads_per_group + head_base) * head_dim_tiles + d;
            tile_regs_acquire();
            mul_tiles_bcast_cols(cb_q, cb_w, q_idx, w_idx, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, cb_q, q_idx);  // gated q back in place
            tile_regs_release();
        }
    }
}

/** Fused single-head: srcA<-k, srcB<-q matmul mode packing straight to cb_acc_strip, no relu (raw dot;
 *  q pre-gated by scale_q_by_w_inplace). */
template <uint32_t q_cb, uint32_t k_cb, uint32_t acc_cb>
inline void set_matmul_to_acc_mode() {
    reconfig_data_format(cb_q, k_cb, cb_w, q_cb);  // srcA(q)->k, srcB(w)->q [guarded]
    pack_reconfig_data_format(cb_q, acc_cb);       // pack->acc
    pack_reconfig_l1_acc(0);
    matmul_block_init(q_cb, k_cb, 1 /*transpose k*/, 1 /*ct_dim*/, heads_per_dest_pass, head_dim_tiles);
    pack_relu_config(ReluConfig::none());
}

/** Fused single-head matmul of n_cols tiles (row r, cols [c_base, +n_cols)) in ONE tile_regs acquire --
 *  one DEST reg per col (n_cols <= DEST cap), each accumulating the head-dim matmul -- then pack together.
 *  Amortizes the per-output acquire/release + pack overhead. Assumes set_matmul_to_acc_mode ran. */
template <uint32_t q_cb, uint32_t k_cb, uint32_t acc_cb>
inline void matmul_cols_to_acc(uint32_t head, uint32_t r, uint32_t c_base, uint32_t n_cols, uint32_t acc_slot_base) {
    const uint32_t q_base = (r * heads_per_group + head) * head_dim_tiles;
    tile_regs_acquire();
    for (uint32_t col = 0; col < n_cols; ++col) {
        for (uint32_t d = 0; d < head_dim_tiles; ++d) {
            matmul_block(
                q_cb,
                k_cb,
                q_base + d,
                (c_base + col) * head_dim_tiles + d,
                col,
                1 /*transpose k*/,
                1,
                heads_per_dest_pass,
                head_dim_tiles);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t col = 0; col < n_cols; ++col) {
        pack_tile<true>(col, acc_cb, acc_slot_base + col);
    }
    tile_regs_release();
}

/** Custom batched block-max-pool (vs compute_kernel_lib::reduce<MAX,REDUCE_ROW> for the pooled path). bf16
 *  MAX+REDUCE_ROW is the FPU/GMPOOL path, so per-output cost beyond the reduce is just acquire/release +
 *  pack. The library reduces ONE block per acquire; this batches ALL blocks_per_unit blocks of a q-row
 *  into one acquire (one DEST reg per block), cutting acquire/release ~blocks_per_unit x. reduce_init's
 *  packer mask puts each block's per-query max in col 0 (writer reads col 0). Requires
 *  blocks_per_unit <= DEST cap (8 half-sync); fall back to the library reduce when it exceeds that. */
template <uint32_t acc_cb, uint32_t scaler_cb, uint32_t out_cb>
inline void block_max_pool_batched(uint32_t unit_strip) {
    CircularBuffer acc(acc_cb);
    CircularBuffer out(out_cb);
    acc.wait_front(unit_strip);
    // One reduce_block_max_row per block folds block_tiles k-tiles into DEST[b] in one call, the 1.0
    // scaler resident in srcB across the block_tiles GMPOOL-MAX ops -- killing the per-reduce_tile
    // math-thread dispatch overhead that bound the pool. block_tiles is compile-time -> template arg.
    reduce_block_max_row_init<block_tiles>(out_cb);
    for (uint32_t r = 0; r < q_tiles_per_unit; ++r) {
        out.reserve_back(blocks_per_unit);
        tile_regs_acquire();
        for (uint32_t b = 0; b < blocks_per_unit; ++b) {
            const uint32_t base = r * k_tiles_per_unit + b * block_tiles;
            reduce_block_max_row<block_tiles>(acc_cb, scaler_cb, base, b);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t b = 0; b < blocks_per_unit; ++b) {
            pack_tile<true>(b, out_cb, b);  // DEST[b] (block b col-0 maxes) -> slot b
        }
        tile_regs_release();
        out.push_back(blocks_per_unit);
    }
    reduce_block_max_row_uninit(acc_cb);
    acc.pop_front(unit_strip);
}

/** Stamp the causal -inf mask onto row r's masked suffix [valid, KC) in place (empty when fully valid).
 *  straddle_* shift the diagonal on the mid-slab boundary chip (0 elsewhere); see causal_diag_tile. */
inline void stamp_masked_suffix(
    const WorkUnitSpan& span,
    uint32_t q_row,
    uint32_t slot_base,
    uint32_t k_tiles_in_unit,
    uint32_t chunk_start_tiles,
    uint32_t straddle_q_tile,
    uint32_t straddle_jump_tiles) {
    const uint32_t k_tile0 = span.k_tile_start();
    const uint32_t q_row_abs = span.q_tile_start() + q_row;
    const uint32_t diag_tile =
        iscore::causal_diag_tile(q_row_abs, chunk_start_tiles, straddle_q_tile, straddle_jump_tiles);
    const uint32_t valid =
        row_valid_prefix(q_row_abs, k_tile0, k_tiles_in_unit, chunk_start_tiles, straddle_q_tile, straddle_jump_tiles);
    for (uint32_t k_col = valid; k_col < k_tiles_per_unit; ++k_col) {
        stamp_mask_tile<cb_acc_strip, cb_mask>(slot_base + k_col, k_tile0 + k_col, diag_tile);
    }
}

void kernel_main() {
    // Banded schedule: this core owns a (group-phase x band) rectangle. groups stream in num_groups phases
    // (group = row_group0 + p*group_stride); each walks num_bands k-bands (band = band0 + j). One cell ==
    // one QC x KC work unit.
    const uint32_t row_group0 = get_arg_val<uint32_t>(0);
    const uint32_t group_stride = get_arg_val<uint32_t>(1);
    const uint32_t num_groups = get_arg_val<uint32_t>(2);
    const uint32_t band0 = get_arg_val<uint32_t>(3);
    const uint32_t num_bands = get_arg_val<uint32_t>(4);
    const uint32_t max_bands = get_arg_val<uint32_t>(5);  // row's widest column; streaming drains q to this
    // Valid KV length in tiles: caps each cell's valid cols (mask suffix grows over the tail). Full when
    // unset (dense path unchanged). Hash-excluded.
    const uint32_t kv_len_tiles = get_arg_val<uint32_t>(6);
    // Per-device chunk-start offset (tiles); runtime so distinct values reuse one program.
    const uint32_t chunk_start_tiles = get_arg_val<uint32_t>(7);
    // Mid-slab boundary-chip diagonal straddle (tiles): q-rows >= straddle_q_tile jump by straddle_jump_tiles.
    // Both 0 on every non-boundary device and in the chunk-aligned case, leaving the diagonal linear.
    const uint32_t straddle_q_tile = get_arg_val<uint32_t>(8);
    const uint32_t straddle_jump_tiles = get_arg_val<uint32_t>(9);
    if (num_groups == 0 || num_bands == 0) {
        return;
    }

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q, cb_k, cb_qk);
    matmul_block_init(
        cb_q, cb_k, 1 /*transpose k*/, 1 /*ct_dim*/, heads_per_dest_pass /*rt_dim*/, head_dim_tiles /*kt_dim*/);
    CircularBuffer(cb_mask).wait_front(num_mask_tiles);  // never popped
    if constexpr (block_pool) {
        CircularBuffer(cb_scaler).wait_front(1);  // 1.0 reduce-MAX scaler, reused, never popped
    }

    CircularBuffer k(cb_k);
    CircularBuffer acc(cb_acc_strip);
    CircularBuffer q(cb_q);

    WorkUnitSpan span;
    span.set_valid_k_len_tiles(kv_len_tiles);

    constexpr uint32_t unit_strip = q_tiles_per_unit * k_tiles_per_unit;  // QC x KC accumulator slots
    constexpr uint32_t q_row_tiles = q_group_tiles / q_tiles_per_unit;    // heads_per_group * head_dim_tiles

    // group-OUTER, band-INNER: q/w resident for a group's whole band run; each band is one QC x KC unit.
    // A partial last band (valid < KC) is masked in stamp_masked_suffix. Streaming pads the loop to
    // max_bands for q-mcast lockstep: a phantom band [num_bands, max_bands) only drains the re-issued q
    // (no k/compute/output). Resident never pads.
    const uint32_t band_iters = stream_heads ? max_bands : num_bands;
    for (uint32_t phase = 0; phase < num_groups; ++phase) {
        const uint32_t group = row_group0 + phase * group_stride;
        for (uint32_t band = 0; band < band_iters; ++band) {
            if constexpr (stream_heads) {
                if (band >= num_bands) {
                    drain_phantom_band_q();  // q-mcast rendezvous only; no compute/output
                    continue;
                }
            }
            span.set(group, band0 + band);
            // Fused + streamed k waits incrementally inside the matmul (overlap the DRAM read); all other
            // paths wait the whole chunk here.
            if constexpr (!fuse_single || !fused_stream_k) {
                k.wait_front(k_chunk_tiles);
            }
            const uint32_t k_tiles_in_unit = span.k_tiles();

            // Fused single-head: gate resident q by w IN PLACE once per group load (band 0; q is reused
            // across bands, so per-band scaling would compound w). One pass per output plane's head.
            if constexpr (fuse_single) {
                if (band == 0) {
                    for (uint32_t g = 0; g < num_out_groups; ++g) {
                        scale_q_by_w_inplace(g * reduce_heads);
                    }
                }
            }

            // One output plane per group (g-major): group g sums only its reduce_heads heads into the
            // accumulator, then untilizes/pools its own plane. num_out_groups==1 = head-summed (one plane);
            // >1 = per-group planes. k/q/w resident across all groups.
            for (uint32_t g = 0; g < num_out_groups; ++g) {
                const uint32_t head_base = g * reduce_heads;
                if constexpr (fuse_single) {
                    // Matmul the whole unit straight to the accumulator (q pre-gated). Matmul-all then
                    // mask-all keeps ONE matmul-mode set and ONE srcA reconfig.
                    acc.reserve_back(unit_strip);
                    set_matmul_to_acc_mode<cb_q, cb_k, cb_acc_strip>();
                    {
                        // mm_col_batch: DEST column batch (half-sync bf16 capacity), shared with the reader's
                        // k-stream chunk so producer/consumer granularities match (indexer_score_common.hpp).
                        // Column-outer: wait only this sub-chunk's k tiles, matmul for ALL rows, then next --
                        // consuming k as the reader streams it. k indexed absolutely into cb_k.
                        for (uint32_t c = 0; c < k_tiles_per_unit; c += mm_col_batch) {
                            const uint32_t c_end =
                                (c + mm_col_batch <= k_tiles_per_unit) ? (c + mm_col_batch) : k_tiles_per_unit;
                            const uint32_t n = c_end - c;
                            if constexpr (fused_stream_k) {
                                k.wait_front(c_end * head_dim_tiles);  // streamed: wait k cols [0, c_end)
                            }
                            for (uint32_t r = 0; r < q_tiles_per_unit; ++r) {
                                matmul_cols_to_acc<cb_q, cb_k, cb_acc_strip>(head_base, r, c, n, r * k_tiles_per_unit + c);
                            }
                        }
                    }
                    // Matmul left srcA in k's format; the mask copies a bf16 -inf tile -> reconfig srcA to bf16.
                    reconfig_data_format_srca(cb_k, cb_mask);
                    for (uint32_t r = 0; r < q_tiles_per_unit; ++r) {
                        stamp_masked_suffix(
                            span,
                            r,
                            r * k_tiles_per_unit,
                            k_tiles_in_unit,
                            chunk_start_tiles,
                            straddle_q_tile,
                            straddle_jump_tiles);
                    }
                    acc.push_back(unit_strip);
                } else {
                    acc.reserve_back(unit_strip);
                    for (uint32_t q_row = 0; q_row < q_tiles_per_unit; ++q_row) {
                        // wait q rows 0..q_row only (reader pushes per row), so row 0 runs while row 1
                        // arrives. Cumulative + non-consuming -> immediate once resident; no-op for g>0.
                        if constexpr (!stream_heads) {
                            q.wait_front((q_row + 1) * q_row_tiles);
                        }
                        const uint32_t slot_base = q_row * k_tiles_per_unit;

                        // PHASE 1 (matmul) + PHASE 2 (mul) per k-col batch (cb_qk holds one batch, so they
                        // alternate); a whole-row batch = one of each per row.
                        if constexpr (qk_col_batch > 1) {
                            for (uint32_t col_base = 0; col_base < k_tiles_per_unit; col_base += qk_col_batch) {
                                const uint32_t cols = (col_base + qk_col_batch <= k_tiles_per_unit)
                                                          ? qk_col_batch
                                                          : (k_tiles_per_unit - col_base);
                                matmul_phase(q_row, col_base, cols, head_base);          // act(q.kT)->cb_qk
                                mul_phase(q_row, slot_base, col_base, cols, head_base);  // gate-mul + reduce
                            }
                        } else {
                            // head-streaming / KC==1 fallback (all heads -> one plane; G==1 only).
                            accumulate_row_streaming(q_row, slot_base);
                        }

                        stamp_masked_suffix(
                            span,
                            q_row,
                            slot_base,
                            k_tiles_in_unit,
                            chunk_start_tiles,
                            straddle_q_tile,
                            straddle_jump_tiles);
                    }
                    acc.push_back(unit_strip);
                }

                // PHASE 3 -- emit this plane's output. block_size==0: untilize the QC strips in one
                // pack_untilize bracket. block-pool: reduce<MAX,REDUCE_ROW> over the masked unit, each block
                // folded to one col-0 tile (writer reads col 0). Future keys are -inf so straddling/future
                // blocks pool correctly.
                if constexpr (block_pool) {
                    // Batch all blocks of a q-row into one DEST acquire when they fit (<= 8 half-sync);
                    // else the library reduce (one acquire per block).
                    if constexpr (blocks_per_unit <= 8) {
                        block_max_pool_batched<cb_acc_strip, cb_scaler, cb_out_strip>(unit_strip);
                    } else {
                        compute_kernel_lib::reduce<
                            PoolType::MAX,
                            ReduceDim::REDUCE_ROW,
                            cb_acc_strip,
                            cb_scaler,
                            cb_out_strip,
                            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
                            compute_kernel_lib::ReduceInputBlockShape::of(
                                /*rows = blocks/row */ blocks_per_unit,
                                /*cols = tiles/block */ block_tiles,
                                /*batches = q-rows */ q_tiles_per_unit));
                    }
                } else {
                    compute_kernel_lib::untilize<k_tiles_per_unit, cb_acc_strip, cb_out_strip>(q_tiles_per_unit);
                }
            }

            k.pop_front(k_chunk_tiles);
        }
        // group's bands done: release its resident q/w (one block per group).
        CircularBuffer(cb_w).pop_front(w_group_tiles);  // gates waited in the mul/scale phase
        if constexpr (!stream_heads) {
            q.pop_front(q_group_tiles);
        }
    }
}
