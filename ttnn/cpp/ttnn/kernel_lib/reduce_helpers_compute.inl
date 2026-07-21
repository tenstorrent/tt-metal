// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Implementation file for reduce_helpers_compute.hpp
// Do not include directly - include reduce_helpers_compute.hpp instead

#include "api/compute/add_int_sfpu.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/debug/assert.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/dfb_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_common.hpp"

namespace compute_kernel_lib {

namespace detail {

// SFPU MAX fold (also used by reduce_{h,w}_neg for -MAX(-x) MIN).
template <DataFormat format>
ALWI void sfpu_reduce_max_fold_init() {
    static_assert(format == DataFormat::Int32, "SFPU reduce MAX fold: Int32 only");
    binary_max_int32_tile_init();
}

template <DataFormat format>
ALWI void sfpu_reduce_max_fold_tile(uint32_t a, uint32_t b, uint32_t out) {
    static_assert(format == DataFormat::Int32, "SFPU reduce MAX fold: Int32 only");
    binary_max_int32_tile(a, b, out);
}

// SFPU SUM fold (Int32 cross-tile add). add_int_tile is exact 2's-complement and wraps on
// overflow, matching torch's Int32 sum semantics.
template <DataFormat format>
ALWI void sfpu_reduce_sum_fold_init() {
    static_assert(format == DataFormat::Int32, "SFPU reduce SUM fold: Int32 only");
    add_int_tile_init();
}

template <DataFormat format>
ALWI void sfpu_reduce_sum_fold_tile(uint32_t a, uint32_t b, uint32_t out) {
    static_assert(format == DataFormat::Int32, "SFPU reduce SUM fold: Int32 only");
    add_int_tile<format>(a, b, out);
}

// Pool-type dispatched cross-tile fold init (MAX -> binary_max, SUM -> add_int).
// Used only by compute_kernel_lib::reduce(); the _neg kernels call the MAX fold directly.
template <PoolType pool_type, DataFormat format>
ALWI void sfpu_reduce_fold_init() {
    if constexpr (pool_type == PoolType::SUM) {
        sfpu_reduce_sum_fold_init<format>();
    } else {
        sfpu_reduce_max_fold_init<format>();
    }
}

// Copy one input tile into DST and fold into the running accumulator (first tile seeds dst_idx
// directly). Fold op is selected by pool_type: MAX -> running max, SUM -> running sum.
template <PoolType pool_type, DataFormat format>
ALWI void sfpu_copy_and_fold(
    uint32_t input_cb_id, uint32_t tile_idx, uint32_t dst_idx, uint32_t work_dst, bool is_first_tile) {
    if (is_first_tile) {
        copy_tile(input_cb_id, tile_idx, dst_idx);
    } else {
        copy_tile(input_cb_id, tile_idx, work_dst);
        if constexpr (pool_type == PoolType::SUM) {
            sfpu_reduce_sum_fold_tile<format>(dst_idx, work_dst, dst_idx);
        } else {
            sfpu_reduce_max_fold_tile<format>(dst_idx, work_dst, dst_idx);
        }
    }
}

// Matches sfpu_copy_and_fold_max is_first_tile: copy on axis 0 unless Accumulate already reloaded DST.
template <typename AccumulateT>
ALWI bool sfpu_is_first_tile(uint32_t axis_index, const AccumulateT& accumulate) {
    if constexpr (is_accumulate_v<AccumulateT>) {
        return axis_index == 0 && accumulate.is_first();
    }
    return axis_index == 0;
}

// Post-reduce scalar multiply. mul_unary_tile is fp32-only, so Int32 is bracketed with typecasts
// (truncates toward zero on the way back); all other formats use plain mul_unary_tile.
template <DataFormat reduce_format>
ALWI void reduce_post_mul_tile(uint32_t dst, uint32_t scaler_bits) {
    if constexpr (reduce_format == DataFormat::Int32) {
        typecast_tile_init<(uint32_t)DataFormat::Int32, (uint32_t)DataFormat::Float32>();
        typecast_tile<(uint32_t)DataFormat::Int32, (uint32_t)DataFormat::Float32>(dst);
        binop_with_scalar_tile_init();
        mul_unary_tile(dst, scaler_bits);
        typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Int32>();
        typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Int32>(dst);
    } else {
        binop_with_scalar_tile_init();
        mul_unary_tile(dst, scaler_bits);
    }
}

// Does `dfb_id` unpack straight into DEST (bypassing SrcA/SrcB)? True only for a 32-bit CB tagged
// UnpackToDestFp32, where the JIT keeps unpack_dst_format == unpack_src_format (Default downgrades to Tf32, bf16
// is never 32-bit). FoldViaAdd reads the accumulator via SrcA/SrcB, so it is invalid for such a CB. UNPACK/MATH
// only (PACK cannot see unpack_dst_format); mirrors tilize_helpers' has_unpack_to_dest_fp32.
ALWI bool dfb_unpacks_to_dest(uint32_t dfb_id) {
#if defined(UCK_CHLKC_PACK)
    (void)dfb_id;
    return false;
#else
    const uint32_t src = unpack_src_format[dfb_id];
    const bool src_is_32bit = (src == (uint32_t)DataFormat::Float32) || (src == (uint32_t)DataFormat::Int32);
    return src_is_32bit && (src == unpack_dst_format[dfb_id]);
#endif
}

// -----------------------------------------------------------------------------
// AccumulateViaAdd datapath (ReduceAlgorithm::AccumulateViaAdd).
//
// Each output tile is produced independently: sum its reduce-dim tiles into DST[0] with pairwise
// add_tiles(acc_to_dest) (parity resolved at the seed — copy_tile one tile when the count is odd, add
// the first pair when even, no phantom zero tile), finalize within the tile on the SFPU (sfpu_reduce
// SUM, which reads DST in place), and for AVG multiply by 1/N once. One DST register per output tile, so
// an arbitrary (Ht, Wt, NC) block is handled without the REDUCE_COL DST/chunk limit.
//
// Restrictions (enforced by reduce()): float SUM, or standalone AVG (1/N derived from tile geometry).
// ALL FOUR ReduceInputPolicy values are supported — BulkWaitBulkPop / WaitUpfrontNoPop / NoWaitNoPop index a
// resident block; WaitAndPopPerTile streams the reduce dim through DST. should_pop policies (Bulk / WaitAndPop)
// pop the input and pack per output; no-pop policies (WaitUpfront / NoWait) leave the input resident and
// bulk-reserve the outputs upfront, packing output o -> its OWN page o. The one-time SFPU-macro load
// (sfpu_reduce_init) is hoisted OUT of the per-output loop; only the light MOP inits (add_tiles/copy) run per
// output.
//
// PARTIAL (non-tile-aligned) reduce dims — ROW/COL only, signalled by partial_scaler.valid_reduce_dim_elements
// (= P valid elements in the LAST reduce-dim tile): the last tile is folded in with a DEST-ACCUMULATING
// masked broadcast-mul (0/1 mask tile at scaler_dfb_id[last_tile_scaler_idx]; row-0 mask for ROW,
// col-0 for COL) via fold_partial_last(), so the padding contributes 0. The bulk stays pure add_tiles
// (fidelity-flat, 2 tiles/op); only the one partial tile is a (fidelity-affected) mul, and a mean divides by
// the true count (full_cnt*32 + P). The bcast shorthands overwrite DEST (clear_fp32_dst_acc=true), so the
// accumulating variant is the LLK directly with acc_to_dest=1 at init and clear_fp32_dst_acc=false at the op.
// Partial is supported standalone (any should_pop / no-pop policy), under ROW streaming, and folded into
// cross-call Accumulate (ROW/COL). SCALAR partial is rejected (a 2-D corner mask a single row/col tile can't
// encode).
//
// CROSS-CALL ACCUMULATE (AccumulateT == Accumulate, BulkWaitBulkPop) — the accumulator CB holds the RAW
// partial-sum tile per output (NOT a reduced tile). On the first chunk (is_first) we sum this chunk's tiles
// (+ the masked partial last, if any). On every later chunk we fold the accumulator into the SAME pairwise add
// NATIVELY — no binary_dest_reuse — with the parity of full_cnt deciding how:
//   even -> ONE dest reload: copy the accumulator into DST as the seed, then add the new tiles in pairs.
//   odd  -> no reload: seed with the first pair of new tiles and make the LAST add's second operand the
//           accumulator (the large running sum lands once, at the end — better numerics than seeding with it).
// The within-tile finalize (sfpu_reduce [+ 1/N for AVG] + post_reduce_op) runs only when accumulate.is_last();
// non-last chunks pack the raw partial sum back to the output CB (which the caller points at the accumulator).
template <
    PoolType reduce_type,
    ReduceDim reduce_dim,
    uint32_t input_dfb_id,
    uint32_t scaler_dfb_id,
    uint32_t output_dfb_id,
    ReduceInputPolicy input_policy,
    ReduceDataFormatReconfigMode reconfig_mode,
    typename AccumulateT,
    typename PostReduceOp>
ALWI void reduce_accumulate_via_add(
    ReduceInputBlockShape shape,
    ReduceInputMemoryLayout input_memory_layout,
    ReducePartialScaler partial_scaler,
    AccumulateT accumulate,
    PostReduceOp post_reduce_op) {
    const uint32_t Ht = shape.rows, Wt = shape.cols, NC = shape.batches;
    // row_pitch = tile distance between consecutive rows of the resident block (>= Wt). row_stride > Wt lets
    // the reduce run over the first Wt columns of a WIDER resident tensor — the padding tiles [Wt, row_pitch)
    // are simply never indexed. 0 => contiguous (row_pitch = Wt). Honored for ROW/COL under BulkWaitBulkPop;
    // SCALAR / streaming / cross-call accumulate require contiguous (asserted in reduce()).
    const uint32_t row_pitch = (input_memory_layout.row_stride > 0u) ? input_memory_layout.row_stride : Wt;
    const uint32_t in_tiles = Ht * row_pitch * NC;

    // DST accumulation format drives the SFPU finalize (fp32 DST when fp32_dest_acc_en is on).
    constexpr DataFormat dst_fmt = DST_ACCUM_MODE ? DataFormat::Float32 : DataFormat::Float16_b;

    constexpr bool is_row = (reduce_dim == ReduceDim::REDUCE_ROW);
    constexpr bool is_col = (reduce_dim == ReduceDim::REDUCE_COL);
    constexpr auto MASK_BCAST = is_col ? ckernel::BroadcastType::COL : ckernel::BroadcastType::ROW;
    // WaitAndPopPerTile STREAMS the reduce-dim tiles through DST — DST *is* the accumulator (add_tiles
    // acc_to_dest), so an arbitrarily large reduce needs only two input tiles resident at a time; there is
    // no CB accumulator and no reload. Streaming is contiguous per output (row/scalar); col is strided, and
    // the partial masked last tile needs indexed access — both use BulkWaitBulkPop over a resident block.
    constexpr bool streaming = (input_policy == ReduceInputPolicy::WaitAndPopPerTile);
    constexpr bool has_accum = is_accumulate_v<AccumulateT>;  // cross-call CB accumulator (raw partial sum)

    // CB-policy predicates (match the standard path). should_pop_p: the output is popped per output tile
    // (Bulk / WaitAndPop) vs bulk-reserved upfront + bulk-pushed at the end (WaitUpfrontNoPop / NoWaitNoPop).
    // helper_waits_block: the whole resident block is waited once (Bulk / WaitUpfront) — NoWaitNoPop trusts the
    // caller to have it resident, WaitAndPop streams per pair. helper_pops_block: only BulkWaitBulkPop pops it.
    constexpr bool should_pop_p =
        (input_policy == ReduceInputPolicy::WaitAndPopPerTile || input_policy == ReduceInputPolicy::BulkWaitBulkPop);
    constexpr bool no_wait_p = (input_policy == ReduceInputPolicy::NoWaitNoPop);
    constexpr bool helper_waits_block = (!streaming && !no_wait_p);
    constexpr bool helper_pops_block = (!streaming && should_pop_p);

    // tiles that collapse into one output, and their stride in the row-major (batch-major) input block.
    const uint32_t cnt = is_row ? Wt : (is_col ? Ht : (Ht * Wt));
    const uint32_t stride = is_col ? row_pitch : 1u;  // COL steps down a column by the row pitch
    const uint32_t n_out = is_row ? (Ht * NC) : (is_col ? (Wt * NC) : NC);

    // This datapath produces a SUM (per output tile). The mean is NOT computed here — normalization is a
    // caller-owned quantity (the true reduced-element count is a property of the whole logical reduction,
    // not of a single call's tile geometry, and it cannot be derived across cross-call accumulate chunks or
    // for uneven shards). Callers get a mean via compute_kernel_lib::reduce_mean, which runs this SUM and
    // applies 1/N with a caller-supplied N in a finalize post_reduce_op.
    const bool has_partial = (partial_scaler.valid_reduce_dim_elements > 0);
    const uint32_t mask_idx = partial_scaler.last_tile_scaler_idx;
    const uint32_t full_cnt = has_partial ? (cnt - 1u) : cnt;  // tiles summed via pure add_tiles

    DataflowBuffer input_dfb(input_dfb_id), scaler_dfb(scaler_dfb_id), output_dfb(output_dfb_id);
    DataflowBuffer accum_dfb([&]() -> uint32_t {
        if constexpr (has_accum) {
            return accumulate.config.cb_accumulator;
        } else {
            return 0;
        }
    }());

    // Accumulate: reload the running accumulator on every chunk except the first; finalize only on the last.
    // (do_finalize is always true when there is no cross-call accumulation.)
    bool do_finalize = true;
    if constexpr (has_accum) {
        do_finalize = accumulate.is_last();
    }

    // Per-reduce()-call setup — LIGHT only. The heavy hw_configure (unpack/math/pack HW setup + pack_dest_init)
    // is the once-per-kernel boot (compute_kernel_hw_startup, same as every reduce) and must NEVER run per
    // reduce() call, so it is not done here. Per call we do only the light format reconfig (gated by
    // reconfig_mode, to adapt SrcA/SrcB/packer formats when this reduce chains after a different-format op —
    // the AccumulateViaAdd analogue of the standard path's reconfig_data_format) plus the light SFPU-macro
    // (re)load; the per-output add_tiles_init / copy_tile_init below re-arm the MOP. This mirrors how ReduceTile
    // relies on boot hw_configure + light reduce_init.
    constexpr bool reconfig_in =
        (reconfig_mode == ReduceDataFormatReconfigMode::INPUT ||
         reconfig_mode == ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT);
    constexpr bool reconfig_out =
        (reconfig_mode == ReduceDataFormatReconfigMode::OUTPUT ||
         reconfig_mode == ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT);
    if constexpr (reconfig_in) {
        reconfig_data_format(input_dfb_id, input_dfb_id);  // both add operands = the input CB
    }
    if constexpr (reconfig_out) {
        pack_reconfig_data_format(output_dfb_id);
    }
    sfpu_reduce_init<PoolType::SUM, dst_fmt>();  // light: (re)load the SFPU reduce macro (persists across adds)
    // Basic validity the reduce() dispatch skips on this path (its compile-time restrictions are asserted
    // there). Capacity self-asserts in each wait_front/reserve_back, except NoWaitNoPop which does neither.
    ASSERT(input_dfb_id != output_dfb_id && Ht > 0 && Wt > 0 && NC > 0);
#ifndef ARCH_QUASAR  // is_valid_dfb_tile_page_size is WH/BH only
    UNPACK(ASSERT(is_valid_dfb_tile_page_size(input_dfb_id, (DataFormat)unpack_src_format[input_dfb_id])));
    PACK(ASSERT(is_valid_dfb_tile_page_size(output_dfb_id, (DataFormat)pack_dst_format[output_dfb_id])));
#endif
    if constexpr (no_wait_p) {  // no wait/reserve to self-assert capacity: caller must have the block resident
        ASSERT(get_dfb_num_pages(input_dfb_id) >= in_tiles);
    }

    // Scaler is consumed by the partial 0/1 mask or the CopySeedZeroPair zero tile; never popped.
    bool wait_scaler = has_partial;
    if constexpr (has_accum) {
        // AccumulateReloadMode contracts. acc_cb (running RAW partial sum) is maybe_unused: only ASSERTs read it.
        [[maybe_unused]] const uint32_t acc_cb = accumulate.config.cb_accumulator;
        ASSERT(input_dfb_id != acc_cb);
        // FoldViaAdd reads acc_cb via SrcA/SrcB — invalid for an UnpackToDestFp32 CB (see dfb_unpacks_to_dest).
        UNPACK(ASSERT(accumulate.reload != AccumulateReloadMode::FoldViaAdd || !dfb_unpacks_to_dest(acc_cb)));
        // CopySeedZeroPair takes scaler_dfb for its zero tile, so it can't also carry a partial mask.
        ASSERT(accumulate.reload != AccumulateReloadMode::CopySeedZeroPair || !has_partial);
#ifdef ARCH_QUASAR
        ASSERT(accumulate.reload != AccumulateReloadMode::CopySeedSfpuAdd);  // needs add_binary_tile (WH/BH only)
#endif
        wait_scaler = wait_scaler || (accumulate.reload == AccumulateReloadMode::CopySeedZeroPair);
    }
    if (wait_scaler) {
        ASSERT(input_dfb_id != scaler_dfb_id && output_dfb_id != scaler_dfb_id);
        scaler_dfb.wait_front(mask_idx + 1);  // partial: 0/1 mask (row-0/col-0); CopySeedZeroPair: zero tile
    }
    if constexpr (helper_waits_block) {
        input_dfb.wait_front(in_tiles);  // Bulk / WaitUpfront: whole resident block, indexed per output
    }
    if constexpr (!should_pop_p) {
        output_dfb.reserve_back(n_out);  // no-pop: reserve every output page upfront (pack o -> page o below)
    }

    // Fold the masked partial LAST reduce-dim tile into DST, ACCUMULATING (acc_to_dest=1 at init,
    // clear_fp32_dst_acc=false at the op — the bcast shorthands would overwrite). Shared by the standalone,
    // cross-call-accumulate, and streaming paths so the partial fold lives in one place. `last_idx` is the
    // input-CB index of that tile (absolute into the resident block, or front-relative 0 for streaming).
    // Referenced from a runtime `if (has_partial)` in every instantiation, so it is never truly unused.
    [[maybe_unused]] auto fold_partial_last = [&](uint32_t last_idx) {
        MATH((llk_math_eltwise_binary_init<ckernel::EltwiseBinaryType::ELWMUL, MASK_BCAST, MATH_FIDELITY>(
            input_dfb_id, scaler_dfb_id, 1)));
        UNPACK((llk_unpack_AB_init<MASK_BCAST>(input_dfb_id, scaler_dfb_id)));
        UNPACK((llk_unpack_AB<MASK_BCAST>(input_dfb_id, scaler_dfb_id, last_idx, mask_idx)));
        MATH((llk_math_eltwise_binary<ckernel::EltwiseBinaryType::ELWMUL, MASK_BCAST, DST_ACCUM_MODE, MATH_FIDELITY>(
            0, false)));
    };

    for (uint32_t o = 0; o < n_out; ++o) {
        tile_regs_acquire();

        if constexpr (streaming) {
            // Stream this output's reduce-dim tiles through DST in pairs, waiting/popping as they arrive
            // (front-relative indices 0/1). Contiguous per output (row/scalar), so tiles arrive in reduce
            // order; DST holds the running sum across the whole stream. acc_to_dest=true throughout: a
            // freshly-acquired DST reads 0 on its first write, so the first add is the plain sum — no separate
            // overwrite-seed init. Odd count: seed DST with a unary copy. The pure-add part covers full_cnt
            // tiles; a partial (ROW only) folds the masked last tile after (== cnt when aligned).
            uint32_t consumed = 0;
            if (full_cnt & 1u) {
                input_dfb.wait_front(1);
                copy_tile_init(input_dfb_id);
                copy_tile(input_dfb_id, 0, 0);
                input_dfb.pop_front(1);
                consumed = 1;
            }
            add_tiles_init(input_dfb_id, input_dfb_id, true);
            for (; consumed < full_cnt; consumed += 2) {
                input_dfb.wait_front(2);
                add_tiles(input_dfb_id, input_dfb_id, 0, 1, 0);
                input_dfb.pop_front(2);
            }
            if (has_partial) {  // ROW partial: the LAST reduce-dim tile is now at the CB front; fold it masked
                input_dfb.wait_front(1);
                fold_partial_last(0);
                input_dfb.pop_front(1);
            }
        } else {
            // Indexed access into the resident block; `start` is output o's first reduce-dim tile. row_pitch
            // is the per-row tile pitch (== Wt when contiguous), so padded rows are skipped automatically.
            uint32_t start;
            if constexpr (is_row) {
                start = o * row_pitch;
            } else if constexpr (is_col) {
                start = (o / Wt) * (Ht * row_pitch) + (o % Wt);
            } else {
                start = o * (Ht * row_pitch);  // scalar: row_pitch == Wt (contiguous, asserted in reduce())
            }

            if constexpr (has_accum) {
                if (accumulate.is_first()) {
                    // First chunk: no accumulator yet — sum this chunk's full_cnt tiles (aligned; accumulate
                    // rejects partial). acc_to_dest=true throughout: a freshly-acquired DST reads 0 on its
                    // first write, so the first add is the plain sum. Odd count: seed DST with a unary copy.
                    uint32_t k = 0;
                    if (full_cnt & 1u) {
                        copy_tile_init(input_dfb_id);
                        copy_tile(input_dfb_id, start, 0);
                        k = 1;
                    }
                    add_tiles_init(input_dfb_id, input_dfb_id, true);
                    for (; k < full_cnt; k += 2) {
                        add_tiles(input_dfb_id, input_dfb_id, start + k * stride, start + (k + 1) * stride, 0);
                    }
                    if (has_partial) {  // ROW/COL partial: fold the masked last tile into this chunk's sum
                        fold_partial_last(start + full_cnt * stride);
                    }
                } else {
                    // Later chunk: fold output o's running accumulator (raw partial sum, front of accum CB)
                    // with this chunk's new tiles. Strategy = accumulate.reload (see AccumulateReloadMode):
                    // FoldViaAdd reads the accumulator via SrcB (fast, Default-acc only); the CopySeed* modes
                    // reload it into DST via copy_tile (the only access a UnpackToDestFp32 acc_cb allows).
                    const uint32_t acc_cb = accumulate.config.cb_accumulator;
                    accum_dfb.wait_front(1);
                    if (accumulate.reload == AccumulateReloadMode::FoldViaAdd) {
                        // Fold the accumulator as an add_tiles SRCB operand — no dest reload. Reads acc via
                        // SrcB, so ONLY valid when acc_cb is UnpackToDestMode::Default. Parity of full_cnt
                        // decides; add_tiles_init does NOT reconfig format, so reconfig SRCB around the acc-add
                        // (acc may be fp32 while the input is bf16) and restore it after.
                        if (full_cnt & 1u) {
                            if (full_cnt == 1u) {
                                reconfig_data_format_srcb(input_dfb_id, acc_cb);
                                add_tiles_init(input_dfb_id, acc_cb, true);  // fresh DST reads 0 -> new[0] + acc
                                add_tiles(input_dfb_id, acc_cb, start, 0, 0);
                                reconfig_data_format_srcb(acc_cb, input_dfb_id);
                            } else {
                                add_tiles_init(input_dfb_id, input_dfb_id, true);                 // fresh DST reads 0
                                add_tiles(input_dfb_id, input_dfb_id, start, start + stride, 0);  // seed new pair
                                add_tiles_init(input_dfb_id, input_dfb_id, true);
                                for (uint32_t k = 2; k + 1 < full_cnt; k += 2) {
                                    add_tiles(
                                        input_dfb_id, input_dfb_id, start + k * stride, start + (k + 1) * stride, 0);
                                }
                                reconfig_data_format_srcb(input_dfb_id, acc_cb);
                                add_tiles_init(input_dfb_id, acc_cb, true);  // last new tile + accumulator
                                add_tiles(input_dfb_id, acc_cb, start + (full_cnt - 1u) * stride, 0, 0);
                                reconfig_data_format_srcb(acc_cb, input_dfb_id);
                            }
                        } else {
                            reconfig_data_format_srca(input_dfb_id, acc_cb);
                            copy_tile_init(acc_cb);
                            copy_tile(acc_cb, 0, 0);  // DST = accumulator (even count reloads as the seed)
                            reconfig_data_format_srca(acc_cb, input_dfb_id);
                            add_tiles_init(input_dfb_id, input_dfb_id, true);
                            for (uint32_t k = 0; k < full_cnt; k += 2) {
                                add_tiles(input_dfb_id, input_dfb_id, start + k * stride, start + (k + 1) * stride, 0);
                            }
                        }
                    } else if (accumulate.reload == AccumulateReloadMode::CopySeedSfpuAdd) {
                        // Sum this chunk's new tiles into DST[0] with pure pairwise add_tiles (fresh DST reads
                        // 0 -> full fp32 accumulation, no DEST-reuse TF32 truncation), reload the accumulator
                        // into DST[1] via copy_tile (U2D-safe, lossless), then SFPU-add DST[0] += DST[1] (the
                        // SFPU operates on DST in fp32). The accumulator is never an FPU SrcA/B operand. WH/BH
                        // only (add_binary_tile is not on Quasar).
                        {
                            uint32_t k = 0;
                            if (full_cnt & 1u) {
                                copy_tile_init(input_dfb_id);
                                copy_tile(input_dfb_id, start, 0);  // DST[0] = new[0] (odd seed, fresh)
                                k = 1;
                            }
                            add_tiles_init(input_dfb_id, input_dfb_id, true);
                            for (; k < full_cnt; k += 2) {
                                add_tiles(input_dfb_id, input_dfb_id, start + k * stride, start + (k + 1) * stride, 0);
                            }
                        }
                        reconfig_data_format_srca(input_dfb_id, acc_cb);
                        copy_tile_init(acc_cb);
                        copy_tile(acc_cb, 0, 1);  // DST[1] = accumulator (adjacent slot)
                        reconfig_data_format_srca(acc_cb, input_dfb_id);
#ifndef ARCH_QUASAR
                        add_binary_tile_init();
                        add_binary_tile(0, 1, 0);                    // DST[0] = DST[0] + DST[1] (fp32 SFPU add)
                        sfpu_reduce_init<PoolType::SUM, dst_fmt>();  // restore the reduce macro for the finalize
#else
                        ASSERT(false);  // CopySeedSfpuAdd needs add_binary_tile (WH/BH only)
#endif
                    } else {
                        // CopySeed*: reload the accumulator into DST via copy_tile — the ONLY access a
                        // UnpackToDestFp32 acc_cb allows (the accumulator is never an FPU operand). copy_tile
                        // uses SrcA (or unpack-direct-to-dest when tagged), so reconfig SRCA around it; SrcB is
                        // left untouched (== input from the per-call reconfig), which the partial fold needs.
                        reconfig_data_format_srca(input_dfb_id, acc_cb);
                        copy_tile_init(acc_cb);
                        copy_tile(acc_cb, 0, 0);  // DST = accumulator
                        reconfig_data_format_srca(acc_cb, input_dfb_id);
                        if (accumulate.reload == AccumulateReloadMode::CopySeedUniform) {
                            // Add every new tile via a DEST-reuse add (new tile -> SrcA, running sum reused
                            // from DST). 1 tile/op; acc stays resident in DST, never an FPU CB operand.
                            binary_dest_reuse_tiles_init<
                                ckernel::EltwiseBinaryType::ELWADD,
                                ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCB>(input_dfb_id);
                            for (uint32_t k = 0; k < full_cnt; ++k) {
                                binary_dest_reuse_tiles<
                                    ckernel::EltwiseBinaryType::ELWADD,
                                    ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                                    input_dfb_id, start + k * stride, 0);
                            }
                        } else if (accumulate.reload == AccumulateReloadMode::CopySeedZeroPair) {
                            // Odd leftover pairs with a ZERO tile (scaler_dfb[0]) via an acc_to_dest add_tiles:
                            // DST += input[leftover] + 0, keeping the running sum in fp32 DST (no DEST-reuse
                            // truncation, no SFPU). Bulk in pairs. Aligned only (scaler_dfb is the zero tile).
                            uint32_t k = 0;
                            if (full_cnt & 1u) {
                                add_tiles_init(input_dfb_id, scaler_dfb_id, true);
                                add_tiles(input_dfb_id, scaler_dfb_id, start, 0, 0);  // DST += input[start] + 0
                                k = 1;
                            }
                            add_tiles_init(input_dfb_id, input_dfb_id, true);
                            for (; k < full_cnt; k += 2) {
                                add_tiles(input_dfb_id, input_dfb_id, start + k * stride, start + (k + 1) * stride, 0);
                            }
                        } else {
                            // CopySeedPairs: odd leftover first via one DEST-reuse add, then the bulk in pairs
                            // (2 tiles/op). Ending on add_tiles(input,input) leaves SrcB=input for the fold.
                            uint32_t k = 0;
                            if (full_cnt & 1u) {
                                binary_dest_reuse_tiles_init<
                                    ckernel::EltwiseBinaryType::ELWADD,
                                    ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCB>(input_dfb_id);
                                binary_dest_reuse_tiles<
                                    ckernel::EltwiseBinaryType::ELWADD,
                                    ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCB>(input_dfb_id, start, 0);
                                k = 1;
                            }
                            add_tiles_init(input_dfb_id, input_dfb_id, true);
                            for (; k < full_cnt; k += 2) {
                                add_tiles(input_dfb_id, input_dfb_id, start + k * stride, start + (k + 1) * stride, 0);
                            }
                        }
                    }
                    accum_dfb.pop_front(1);
                    if (has_partial) {  // ROW/COL partial: fold the masked last tile in after the accumulator
                        fold_partial_last(start + full_cnt * stride);
                    }
                }
            } else {
                // acc_to_dest=true throughout: a freshly-acquired DST reads 0 on its first write, so the first
                // add is the plain sum — no separate overwrite-seed init. Odd count: seed DST with a unary copy.
                uint32_t k = 0;
                if (full_cnt & 1u) {
                    copy_tile_init(input_dfb_id);
                    copy_tile(input_dfb_id, start, 0);
                    k = 1;
                }
                add_tiles_init(input_dfb_id, input_dfb_id, true);
                for (; k < full_cnt; k += 2) {
                    add_tiles(input_dfb_id, input_dfb_id, start + k * stride, start + (k + 1) * stride, 0);
                }
                // partial: fold the LAST reduce-dim tile in, masked, ACCUMULATING into DST.
                if (has_partial) {
                    fold_partial_last(start + full_cnt * stride);
                }
            }
        }

        // Finalize within the tile only on the last chunk (always, when there is no cross-call accumulate);
        // non-last accumulate chunks leave the RAW partial sum in DST to write back to the accumulator CB.
        if (do_finalize) {
            if constexpr (is_row) {
                sfpu_reduce<PoolType::SUM, dst_fmt, ReduceDim::REDUCE_ROW>(0, 1, 1);
            } else if constexpr (is_col) {
                sfpu_reduce<PoolType::SUM, dst_fmt, ReduceDim::REDUCE_COL>(0, 1, 1);
            } else {
                sfpu_reduce<PoolType::SUM, dst_fmt, ReduceDim::REDUCE_ROW>(0, 1, 1);
                sfpu_reduce<PoolType::SUM, dst_fmt, ReduceDim::REDUCE_COL>(0, 1, 1);
            }
            // Standalone AVG: divide by the element count from tile geometry (aligned ROW/COL = cnt*32,
            // SCALAR = Ht*Wt*1024, partial ROW/COL = full_cnt*32 + P). Emits the same sfpu_reduce ->
            // mul_unary_tile sequence as reduce_mean, so reduce<AVG> and reduce_mean are bit-identical for the
            // standalone case. Cross-chunk / sharded / uneven mean uses reduce_mean (caller N); AVG + Accumulate
            // is rejected in reduce() (the geometry N cannot span chunks).
            if constexpr (reduce_type == PoolType::AVG) {
                const uint32_t n_geom =
                    (is_row || is_col) ? (full_cnt * 32u + partial_scaler.valid_reduce_dim_elements) : (cnt * 1024u);
                float inv_f = 1.0f / static_cast<float>(n_geom);
                uint32_t inv_bits = 0;
                __builtin_memcpy(&inv_bits, &inv_f, sizeof(inv_bits));
                mul_unary_tile(0, inv_bits);  // no binop_with_scalar init needed after sfpu_reduce
            }
            // DST now holds the reduced value (raw SUM, or the mean for AVG). A caller post_reduce_op (e.g.
            // reduce_mean's caller 1/N, or recip for softmax) then sees the finalized value.
            post_reduce_op(0);
        }

        tile_regs_commit();
        tile_regs_wait();
        if constexpr (should_pop_p) {  // Bulk / WaitAndPop: reserve + pack + push per output tile
            output_dfb.reserve_back(1);
            pack_tile(0, output_dfb_id);
            output_dfb.push_back(1);
        } else {  // no-pop: bulk-reserved upfront; write output o to its OWN page o. (The standard no-pop body
                  // packs every output to the default page 0 — correct only for a single output; the fast path
                  // passes o explicitly so multi-output no-pop is correct.)
            pack_tile(0, output_dfb_id, o);
        }
        tile_regs_release();
    }
    if constexpr (!should_pop_p) {
        output_dfb.push_back(n_out);  // no-pop: bulk-push all outputs at the end
    }
    if constexpr (helper_pops_block) {
        input_dfb.pop_front(in_tiles);  // only BulkWaitBulkPop pops the resident block
    }
}

}  // namespace detail

// =============================================================================
// ReduceDataFormatReconfigMode Helper Functions
// =============================================================================

constexpr bool reconfig_input(ReduceDataFormatReconfigMode mode) {
    return mode == ReduceDataFormatReconfigMode::INPUT || mode == ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT;
}

constexpr bool reconfig_output(ReduceDataFormatReconfigMode mode) {
    return mode == ReduceDataFormatReconfigMode::OUTPUT || mode == ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT;
}

// =============================================================================
// ReduceInputPolicy Helper Functions
// =============================================================================

constexpr bool waits_per_tile(ReduceInputPolicy p) { return p == ReduceInputPolicy::WaitAndPopPerTile; }
constexpr bool waits_bulk(ReduceInputPolicy p) { return p == ReduceInputPolicy::BulkWaitBulkPop; }
constexpr bool waits_upfront(ReduceInputPolicy p) { return p == ReduceInputPolicy::WaitUpfrontNoPop; }
constexpr bool no_wait(ReduceInputPolicy p) { return p == ReduceInputPolicy::NoWaitNoPop; }
constexpr bool should_pop(ReduceInputPolicy p) {
    return p == ReduceInputPolicy::WaitAndPopPerTile || p == ReduceInputPolicy::BulkWaitBulkPop;
}
constexpr bool manages_cb(ReduceInputPolicy p) {
    // Returns true if the reduce function manages CB wait/reserve/push (not preloaded)
    return p != ReduceInputPolicy::NoWaitNoPop;
}

// =============================================================================
// Helper Function Implementations
// =============================================================================

template <PoolType reduce_type, ReduceDim reduce_dim>
ALWI void reduce_init_short_with_dt(uint32_t old_dfb_id, uint32_t input_dfb_id, uint32_t scaler_dfb_id) {
    constexpr bool swap_operands = reduce_swaps_operands<reduce_type, reduce_dim, false>();
    const uint32_t srca_dfb_id = swap_operands ? scaler_dfb_id : input_dfb_id;

    // Reconfigure SRCA data format from old_dfb_id to the correct SrcA format
    UNPACK(
        (llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE, p_dim_stride_target::IGNORE>(old_dfb_id, srca_dfb_id)));
    MATH((llk_math_reconfig_data_format_srca<DST_ACCUM_MODE>(old_dfb_id, srca_dfb_id)));

    // Reconfigure unpacker for reduce operation (SRCA and SRCB)
    UNPACK((llk_unpack_AB_reduce_init<reduce_type, reduce_dim>(input_dfb_id, scaler_dfb_id)));

    // Reconfigure math for reduce operation
    MATH((llk_math_reduce_init<reduce_type, reduce_dim, DST_ACCUM_MODE, MATH_FIDELITY>(input_dfb_id, scaler_dfb_id)));

    // Skip packer reconfiguration - it remains valid from initial reduce_init call
}

template <typename AccumulateT>
ALWI constexpr uint32_t get_dst_index(const AccumulateT& accumulate) {
    if constexpr (is_accumulate_v<AccumulateT>) {
        return accumulate.config.dst_index;
    } else {
        return 0;
    }
}

template <
    PoolType reduce_type,
    ReduceDim reduce_dim,
    DataFormat reduce_format,
    typename AccumulateT,
    bool is_sfpu = false>
ALWI void reload_accumulator_if_needed(
    DataflowBuffer& accum_dfb, uint32_t input_dfb_id, uint32_t scaler_dfb_id, const AccumulateT& accumulate) {
    if constexpr (is_accumulate_v<AccumulateT>) {
        if (!accumulate.is_first()) {  // Reload on all iterations except first
            constexpr uint32_t onetile = 1;
            accum_dfb.wait_front(onetile);
            constexpr bool swap_operands = reduce_swaps_operands<reduce_type, reduce_dim, is_sfpu>();
            const uint32_t prev_srca_cb = swap_operands ? scaler_dfb_id : input_dfb_id;

            // For MAX + REDUCE_ROW, GMPOOL's running accumulator lives at row 0 of face 0
            // (max for rows 0-15) and row 0 of face 2 (max for rows 16-31); faces 1 and 3
            // are never read. The LLK's reduce_row_perform_transpose then rotates those
            // row-0 accumulators into col 0 of face 0 and col 0 of face 2 for packing.
            // A vanilla copy_tile reload would leave the running max at col 0, but the
            // next GMPOOL iteration only reads row 0 — so it would be silently dropped.
            // Within-face-16x16-transpose on reload puts col 0 of each face back at row 0
            // of that face, restoring the exact layout GMPOOL expects.
            constexpr bool reload_within_face_transpose =
                (reduce_type == PoolType::MAX && reduce_dim == ReduceDim::REDUCE_ROW);

            reconfig_data_format_srca(prev_srca_cb, accumulate.config.cb_accumulator);
            copy_tile_to_dst_init_short(
                accumulate.config.cb_accumulator,
                /*transpose_of_faces=*/0,
                /*transpose_within_16x16_face=*/reload_within_face_transpose ? 1u : 0u);
            copy_tile(accumulate.config.cb_accumulator, 0, accumulate.config.dst_index);
            accum_dfb.pop_front(onetile);

            // CRITICAL: Re-init after copy_tile corrupts SRCA config
            // Use short version since packer config is still valid from initial init
            // Pass accumulator DFB as old_dfb_id to reconfigure data format from accumulator to input DFB
            if constexpr (is_sfpu) {
                detail::sfpu_reduce_fold_init<reduce_type, reduce_format>();
            } else {
                reduce_init_short_with_dt<reduce_type, reduce_dim>(
                    accumulate.config.cb_accumulator, input_dfb_id, scaler_dfb_id);
            }
        }
    }
}

template <ReduceInputPolicy input_policy>
ALWI void assert_input_dfb_size(uint32_t input_dfb_id, uint32_t tiles_per_bulk, uint32_t total_tiles) {
    if constexpr (waits_per_tile(input_policy)) {
        ASSERT(get_dfb_num_pages(input_dfb_id) >= 1);
    } else if constexpr (waits_bulk(input_policy)) {
        ASSERT(get_dfb_num_pages(input_dfb_id) >= tiles_per_bulk);
        ASSERT(get_dfb_num_pages(input_dfb_id) % tiles_per_bulk == 0);
    } else {  // waits_upfront or no_wait
        ASSERT(get_dfb_num_pages(input_dfb_id) >= total_tiles);
    }
}

template <ReduceInputPolicy input_policy>
ALWI void assert_output_dfb_size(uint32_t output_dfb_id, uint32_t total_outputs) {
    if constexpr (should_pop(input_policy)) {
        // Per-tile reserve/push: only needs 1 page
        ASSERT(get_dfb_num_pages(output_dfb_id) >= 1);
    } else {
        // Bulk reserve upfront: needs all outputs
        ASSERT(get_dfb_num_pages(output_dfb_id) >= total_outputs);
    }
}

// =============================================================================
// Main Reduce Function Implementation
// =============================================================================

template <
    PoolType reduce_type,
    ReduceDim reduce_dim,
    uint32_t input_dfb_id,
    uint32_t scaler_dfb_id,
    uint32_t output_dfb_id,
    ReduceInputPolicy input_policy,
    ReduceDataFormatReconfigMode reconfig_mode,
    ReduceAlgorithm algorithm,
    typename AccumulateT,
    typename PostReduceOp>
ALWI void reduce(
    ReduceInputBlockShape input_block_shape,
    ReduceInputMemoryLayout input_memory_layout,
    AccumulateT accumulate,
    PostReduceOp post_reduce_op,
    ReducePartialScaler partial_scaler) {
    // Int32 MAX is routed to the SFPU path via is_sfpu_reduce_path<>(); all other formats use FPU/GMPOOL.
    constexpr DataFormat reduce_format = static_cast<DataFormat>(unpack_src_format[input_dfb_id]);
    // =============================================================================
    // Static Assertions (compile-time validation)
    // =============================================================================
    static_assert(
        (reduce_type != PoolType::MAX && reduce_type != PoolType::SUM) || reduce_dim != ReduceDim::REDUCE_SCALAR ||
            reduce_format != DataFormat::Int32,
        "Int32 MAX/SUM REDUCE_SCALAR is not supported (host decomposes Int32 HW reduce into W-then-H)");
    static_assert(
        reduce_type != PoolType::AVG || reduce_format != DataFormat::Int32, "Int32 AVG (mean) is not supported");
    static_assert(
        is_accumulation_type_v<AccumulateT>,
        "AccumulateT must be a valid accumulation type (NoAccumulation or Accumulate)");
    static_assert(is_post_reduce_op_v<PostReduceOp>, "PostReduceOp must be callable with a uint32_t argument");
    static_assert(
        !is_accumulate_v<AccumulateT> || !(reduce_type == PoolType::MAX && reduce_dim == ReduceDim::REDUCE_SCALAR),
        "Accumulate with PoolType::MAX + REDUCE_SCALAR is not supported: the pack edge mask "
        "keeps only DST(0,0), but GMPOOL needs that running max broadcast across face-0 row 4 "
        "on the reload pass, which the current copy_tile reload cannot reproduce.");
#ifdef ARCH_QUASAR
    // The MAX + REDUCE_ROW accumulator reload relies on a within-16x16-face transpose during
    // copy_tile_to_dst_init_short (see reload_accumulator_if_needed). That transpose is rejected
    // by copy_tile_to_dst_init_short on Quasar ("Transpose within face not supported on Quasar"),
    // and there is no Quasar-compatible reload that restores the layout GMPOOL expects.
    static_assert(
        !is_accumulate_v<AccumulateT> || !(reduce_type == PoolType::MAX && reduce_dim == ReduceDim::REDUCE_ROW),
        "Accumulate with PoolType::MAX + REDUCE_ROW is not supported on Quasar: the accumulator "
        "reload requires a within-16x16-face transpose, which copy_tile_to_dst_init_short asserts "
        "against on Quasar.");
#endif

    // =============================================================================
    // Algorithm selection. Auto resolves to a concrete datapath (for now always ReduceTile; a cost
    // heuristic will choose later). AccumulateViaAdd is a restricted, faster datapath for wide float
    // SUM reduces; anything it cannot express is rejected here (compile-time where possible) and must use
    // ReduceTile. ReduceTile (and Auto) fall through to the standard body below.
    // =============================================================================
    constexpr ReduceAlgorithm resolved_algorithm =
        (algorithm == ReduceAlgorithm::Auto) ? ReduceAlgorithm::ReduceTile : algorithm;
    if constexpr (resolved_algorithm == ReduceAlgorithm::AccumulateViaAdd) {
        static_assert(
            reduce_type == PoolType::SUM || reduce_type == PoolType::AVG,
            "AccumulateViaAdd computes SUM, or standalone AVG (1/N derived from tile geometry: aligned ROW/COL "
            "= cnt*32, SCALAR = Ht*Wt*1024, partial ROW/COL = full_cnt*32 + P). A cross-chunk / sharded / uneven "
            "mean must use compute_kernel_lib::reduce_mean (caller-supplied N — not derivable from one call's "
            "geometry). MAX/MIN are not expressible via additive accumulate; use ReduceTile.");
        static_assert(
            reduce_format != DataFormat::Int32,
            "AccumulateViaAdd: float only (add_tiles + sfpu_reduce). Int32 must use ReduceTile.");
        // All four ReduceInputPolicy values are supported: BulkWaitBulkPop / WaitUpfrontNoPop / NoWaitNoPop
        // index a resident block (should_pop vs bulk-reserve output); WaitAndPopPerTile streams the reduce dim.
        // Cross-call Accumulate (CB accumulator holding the RAW partial sum, folded into the pairwise add):
        // SUM only (the internal AVG 1/N is per-call and cannot span chunks — use SUM + a 1/N post_reduce_op
        // on the last chunk), BulkWaitBulkPop only, and tile-aligned only (asserted below).
        static_assert(
            !is_accumulate_v<AccumulateT> || reduce_type == PoolType::SUM,
            "AccumulateViaAdd + Accumulate: SUM only. For a cross-chunk mean, use SUM and apply 1/N in a "
            "post_reduce_op on the last chunk (see reduce_rm).");
        static_assert(
            !is_accumulate_v<AccumulateT> || input_policy == ReduceInputPolicy::BulkWaitBulkPop,
            "AccumulateViaAdd + Accumulate: BulkWaitBulkPop only (the accumulator fold indexes a resident "
            "block).");
        // WaitAndPopPerTile streams the reduce-dim tiles through DST (the accumulator), so it needs
        // contiguous, tile-aligned reduce order: row/scalar only (col is strided), and no partial (the
        // masked last tile needs indexed access). Those cases use BulkWaitBulkPop.
        static_assert(
            input_policy != ReduceInputPolicy::WaitAndPopPerTile || reduce_dim != ReduceDim::REDUCE_COL,
            "AccumulateViaAdd streaming (WaitAndPopPerTile) is contiguous-only (row/scalar); REDUCE_COL is "
            "strided — use BulkWaitBulkPop.");
        // Streaming (WaitAndPopPerTile) + partial is supported for ROW (the masked last tile folds in as the
        // final streamed op). COL streaming is rejected above; SCALAR partial is rejected below.
        // Partial (non-tile-aligned) reduce dims are supported for ROW/COL under BulkWaitBulkPop (the last
        // reduce-dim tile is folded in with a masked accumulating broadcast-mul; valid_reduce_dim_elements =
        // P + a 0/1 mask tile in scaler_dfb). REDUCE_SCALAR can be partial in BOTH axes at once (a single
        // row/col mask can't express the corner), so it is rejected — use ReduceTile.
        if constexpr (reduce_dim == ReduceDim::REDUCE_SCALAR) {
            ASSERT(partial_scaler.valid_reduce_dim_elements == 0);
        }
        // Cross-call Accumulate + partial is supported for ROW/COL (the masked last tile folds into each
        // chunk's sum via fold_partial_last). SCALAR partial is rejected above (622-ish) regardless of accumulate.
        // row_stride (a WIDER resident block, padded rows) is honored for ROW/COL indexed reduces — the
        // per-output indexing steps by the row pitch and skips the padding tiles — under BulkWaitBulkPop,
        // WaitUpfrontNoPop, NoWaitNoPop, AND cross-call Accumulate (the fold uses the same start/stride/pitch).
        // SCALAR walks a 2-D block (a single linear reduce-dim stride cannot skip per-row padding) and streaming
        // is a pure contiguous stream (no indexing) — both require a contiguous layout (row_stride 0 or == Wt).
        if (input_memory_layout.row_stride != 0) {
            ASSERT(input_memory_layout.row_stride >= input_block_shape.cols);
            if constexpr (reduce_dim == ReduceDim::REDUCE_SCALAR) {
                ASSERT(input_memory_layout.row_stride == input_block_shape.cols);
            }
            if constexpr (input_policy == ReduceInputPolicy::WaitAndPopPerTile) {
                ASSERT(input_memory_layout.row_stride == input_block_shape.cols);
            }
        }
        detail::reduce_accumulate_via_add<
            reduce_type,
            reduce_dim,
            input_dfb_id,
            scaler_dfb_id,
            output_dfb_id,
            input_policy,
            reconfig_mode,
            AccumulateT,
            PostReduceOp>(input_block_shape, input_memory_layout, partial_scaler, accumulate, post_reduce_op);
        return;
    }

    // =============================================================================
    // Runtime Assertions (parameter validation)
    // =============================================================================
    ASSERT(input_dfb_id != output_dfb_id);
    ASSERT(input_dfb_id != scaler_dfb_id);
    ASSERT(output_dfb_id != scaler_dfb_id);
    UNPACK(ASSERT(is_valid_dfb_tile_page_size(input_dfb_id, (DataFormat)unpack_src_format[input_dfb_id])));
    UNPACK(ASSERT(is_valid_dfb_tile_page_size(scaler_dfb_id, (DataFormat)unpack_src_format[scaler_dfb_id])));
    PACK(ASSERT(is_valid_dfb_tile_page_size(output_dfb_id, (DataFormat)pack_dst_format[output_dfb_id])));
    ASSERT(input_block_shape.rows > 0);
    ASSERT(input_block_shape.cols > 0);
    ASSERT(input_block_shape.batches > 0);
    if (input_memory_layout.row_stride != 0) {
        ASSERT(input_memory_layout.row_stride >= input_block_shape.cols);
    }

    // Compile-time flag: true when Accumulate type is passed, false otherwise
    constexpr bool enable_accumulation = is_accumulate_v<AccumulateT>;
    // Extract block shape components
    const uint32_t Ht = input_block_shape.rows;
    const uint32_t Wt = input_block_shape.cols;
    const uint32_t num_batches = input_block_shape.batches;

    constexpr bool is_sfpu = is_sfpu_reduce_path<reduce_type, reduce_dim, reduce_format>();

    DataflowBuffer input_dfb(input_dfb_id);
    DataflowBuffer scaler_dfb(scaler_dfb_id);
    DataflowBuffer output_dfb(output_dfb_id);
    DataflowBuffer accum_dfb([&]() -> uint32_t {
        if constexpr (enable_accumulation) {
            return accumulate.config.cb_accumulator;
        } else {
            return 0;
        }
    }());

    // Apply reconfig based on mode
    constexpr bool swap_operands = reduce_swaps_operands<reduce_type, reduce_dim, is_sfpu>();
    if constexpr (reconfig_input(reconfig_mode)) {
        if constexpr (swap_operands) {
            reconfig_data_format(scaler_dfb_id, input_dfb_id);
        } else {
            reconfig_data_format(input_dfb_id, scaler_dfb_id);
        }
    }
    if constexpr (reconfig_output(reconfig_mode)) {
        pack_reconfig_data_format(output_dfb_id);
    }
    // Initialization
    if constexpr (is_sfpu) {
        init_sfpu(input_dfb_id, output_dfb_id);
        copy_tile_to_dst_init_short(input_dfb_id);
    } else {
        reduce_init<reduce_type, reduce_dim>(input_dfb_id, scaler_dfb_id, output_dfb_id);
    }
    // Partial scaler: REDUCE_SCALAR can't use it (applies the scaler twice).
    // Other reduce dims may add a partial-fill tile at index >0; wait for both.
    if constexpr (reduce_dim == ReduceDim::REDUCE_SCALAR) {
        ASSERT(partial_scaler.last_tile_scaler_idx == 0);
    }
    scaler_dfb.wait_front(partial_scaler.last_tile_scaler_idx + 1);
    if constexpr (is_sfpu) {
        PACK((llk_pack_reduce_mask_config<reduce_dim, PackMode::Default>(output_dfb_id)));
    }

    constexpr uint32_t onetile = 1;

    // Pattern dispatch based on reduce_dim
    if constexpr (reduce_dim == ReduceDim::REDUCE_SCALAR) {
        // =================================================================
        // REDUCE_SCALAR: HW reduction - all tiles -> 1 output tile per batch
        // =================================================================
        const uint32_t stride = (input_memory_layout.row_stride > 0) ? input_memory_layout.row_stride : Wt;
        const uint32_t tiles_per_bulk = Ht * stride;
        const uint32_t total_input_tiles = tiles_per_bulk * num_batches;
        const uint32_t total_output_tiles = num_batches;
        UNPACK((assert_input_dfb_size<input_policy>(input_dfb_id, tiles_per_bulk, total_input_tiles)));
        PACK((assert_output_dfb_size<input_policy>(output_dfb_id, total_output_tiles)));

        // No-pop modes: bulk reserve output upfront
        if constexpr (!should_pop(input_policy)) {
            output_dfb.reserve_back(total_output_tiles);
        }

        // PersistentPolicy: wait for all tiles upfront
        if constexpr (waits_upfront(input_policy)) {
            input_dfb.wait_front(total_input_tiles);
        }

        uint32_t batch_offset = 0;
        for (uint32_t nc = 0; nc < num_batches; ++nc) {
            // BulkWaitBulkPop: wait for all Ht×Wt tiles in bulk
            if constexpr (waits_bulk(input_policy)) {
                input_dfb.wait_front(tiles_per_bulk);
            }

            tile_regs_acquire();

            // Reload accumulator if needed (zero overhead when AccumulateT is NoAccumulation)
            reload_accumulator_if_needed<reduce_type, reduce_dim, reduce_format, AccumulateT, is_sfpu>(
                accum_dfb, input_dfb_id, scaler_dfb_id, accumulate);

            const uint32_t dst_idx = get_dst_index(accumulate);
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                for (uint32_t wt = 0; wt < Wt; ++wt) {
                    if constexpr (waits_per_tile(input_policy)) {
                        // One-at-a-time: wait/pop per tile
                        input_dfb.wait_front(onetile);
                        reduce_tile<reduce_type, reduce_dim>(input_dfb_id, scaler_dfb_id, 0, 0, dst_idx);
                        input_dfb.pop_front(onetile);
                    } else if constexpr (waits_bulk(input_policy)) {
                        // BulkWaitBulkPop: use indexed access
                        uint32_t tile_idx = ht * stride + wt;
                        reduce_tile<reduce_type, reduce_dim>(input_dfb_id, scaler_dfb_id, tile_idx, 0, dst_idx);
                    } else {  // PreloadedPolicy or PersistentPolicy: indexed access
                        uint32_t tile_idx = batch_offset + ht * stride + wt;
                        reduce_tile<reduce_type, reduce_dim>(input_dfb_id, scaler_dfb_id, tile_idx, 0, dst_idx);
                    }
                }
            }

            // Call post-reduce operation on the single accumulated DST register.
            // No-op when PostReduceOp is the default NoOp.
            post_reduce_op(dst_idx);

            // Pop modes: reserve per-batch
            if constexpr (should_pop(input_policy)) {
                output_dfb.reserve_back(onetile);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(get_dst_index(accumulate), output_dfb_id);
            tile_regs_release();
            if constexpr (should_pop(input_policy)) {
                output_dfb.push_back(onetile);
            }

            // BulkWaitBulkPop: pop all tiles after processing
            if constexpr (waits_bulk(input_policy)) {
                input_dfb.pop_front(tiles_per_bulk);
            }

            // PreloadedPolicy or PersistentPolicy: update batch offset
            if constexpr (!should_pop(input_policy)) {
                batch_offset += tiles_per_bulk;
            }
        }

        // No-pop modes: bulk push output at end
        if constexpr (!should_pop(input_policy)) {
            output_dfb.push_back(total_output_tiles);
        }
    } else if constexpr (reduce_dim == ReduceDim::REDUCE_ROW) {
        // =================================================================
        // REDUCE_ROW: W reduction - each row -> 1 output tile (Ht outputs per batch)
        // =================================================================
        const uint32_t stride = (input_memory_layout.row_stride > 0) ? input_memory_layout.row_stride : Wt;
        const uint32_t total_output_tiles = Ht * num_batches;
        const uint32_t total_input_tiles = Ht * stride * num_batches;
        UNPACK((assert_input_dfb_size<input_policy>(input_dfb_id, Wt, total_input_tiles)));
        PACK((assert_output_dfb_size<input_policy>(output_dfb_id, total_output_tiles)));

        // No-pop modes: bulk reserve output upfront
        if constexpr (!should_pop(input_policy)) {
            output_dfb.reserve_back(total_output_tiles);
        }

        // PersistentPolicy: wait for all tiles upfront
        if constexpr (waits_upfront(input_policy)) {
            input_dfb.wait_front(total_input_tiles);
        }

        uint32_t index_offset = 0;
        for (uint32_t nc = 0; nc < num_batches; ++nc) {
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                // BulkWaitBulkPop: wait for entire row upfront
                if constexpr (waits_bulk(input_policy)) {
                    input_dfb.wait_front(Wt);
                }

                tile_regs_acquire();

                // Reload accumulator if needed (zero overhead when AccumulateT is NoAccumulation)
                reload_accumulator_if_needed<reduce_type, reduce_dim, reduce_format, AccumulateT, is_sfpu>(
                    accum_dfb, input_dfb_id, scaler_dfb_id, accumulate);
                if constexpr (is_sfpu) {
                    if (Wt > 1) {
                        detail::sfpu_reduce_fold_init<reduce_type, reduce_format>();
                    }
                }

                const uint32_t dst_idx = get_dst_index(accumulate);
                for (uint32_t wt = 0; wt < Wt; ++wt) {
                    if constexpr (is_sfpu) {
                        constexpr uint32_t sfpu_work_dst = 1;
                        const bool is_first_tile = detail::sfpu_is_first_tile(wt, accumulate);
                        if constexpr (waits_per_tile(input_policy)) {
                            input_dfb.wait_front(onetile);
                            detail::sfpu_copy_and_fold<reduce_type, reduce_format>(
                                input_dfb_id, 0, dst_idx, sfpu_work_dst, is_first_tile);
                            input_dfb.pop_front(onetile);
                        } else if constexpr (waits_bulk(input_policy)) {
                            detail::sfpu_copy_and_fold<reduce_type, reduce_format>(
                                input_dfb_id, wt, dst_idx, sfpu_work_dst, is_first_tile);
                        } else {
                            detail::sfpu_copy_and_fold<reduce_type, reduce_format>(
                                input_dfb_id, wt + index_offset, dst_idx, sfpu_work_dst, is_first_tile);
                        }
                    } else {
                        // Last W-tile picks up the partial scaler when one was prepared by the reader.
                        const uint32_t scaler_idx = (wt == Wt - 1) ? partial_scaler.last_tile_scaler_idx : 0;
                        if constexpr (waits_per_tile(input_policy)) {
                            // One-at-a-time: wait/pop per tile
                            input_dfb.wait_front(onetile);
                            reduce_tile<reduce_type, reduce_dim>(input_dfb_id, scaler_dfb_id, 0, scaler_idx, dst_idx);
                            input_dfb.pop_front(onetile);
                        } else if constexpr (waits_bulk(input_policy)) {
                            // BulkWaitBulkPop: use indexed access
                            reduce_tile<reduce_type, reduce_dim>(input_dfb_id, scaler_dfb_id, wt, scaler_idx, dst_idx);
                        } else {  // PreloadedPolicy or PersistentPolicy: indexed access
                            reduce_tile<reduce_type, reduce_dim>(
                                input_dfb_id, scaler_dfb_id, wt + index_offset, scaler_idx, dst_idx);
                        }
                    }
                }

                // SFPU intra-tile finalize
                if constexpr (is_sfpu) {
                    sfpu_reduce_init<reduce_type, reduce_format>();
                    sfpu_reduce<reduce_type, reduce_format, reduce_dim>(dst_idx, /*ct_dim=*/1, /*rt_dim=*/1);
                }

                // Call post-reduce operation (e.g., recip_tile for softmax)
                // User's lambda can include reduce_uninit() if needed before custom ops
                post_reduce_op(dst_idx);

                // Pop modes: reserve per-row to avoid deadlock
                if constexpr (should_pop(input_policy)) {
                    output_dfb.reserve_back(onetile);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(dst_idx, output_dfb_id);
                tile_regs_release();
                if constexpr (should_pop(input_policy)) {
                    output_dfb.push_back(onetile);
                }

                // BulkWaitBulkPop: pop all tiles after processing
                if constexpr (waits_bulk(input_policy)) {
                    input_dfb.pop_front(Wt);
                }

                // PreloadedPolicy or PersistentPolicy: update index offset
                if constexpr (!should_pop(input_policy)) {
                    index_offset += stride;
                }
            }
        }

        // No-pop modes: bulk push output at end
        if constexpr (!should_pop(input_policy)) {
            output_dfb.push_back(total_output_tiles);
        }
    } else {
        // =================================================================
        // REDUCE_COL: H reduction - each column -> 1 output tile (Wt outputs per batch)
        // Need chunking due to DEST register limits
        // StreamingPolicy: Tiles arrive in N C W_skip H W_chunk order (chunked by chunk_size)
        // PreloadedPolicy: Tiles in row-major order, indexed as batch_offset + ht*stride + wt
        // =================================================================

        // Auto-detect chunk size from DEST register capacity
        // Both reader (dataflow) and compute kernels compute this identically via DEST_AUTO_LIMIT
        constexpr uint32_t chunk_size = is_sfpu ? (DEST_AUTO_LIMIT - 1) : DEST_AUTO_LIMIT;
        const uint32_t stride = (input_memory_layout.row_stride > 0) ? input_memory_layout.row_stride : Wt;
        const uint32_t tiles_per_bulk = Ht * stride;
        const uint32_t total_output_tiles = Wt * num_batches;
        const uint32_t total_input_tiles = tiles_per_bulk * num_batches;
        UNPACK((assert_input_dfb_size<input_policy>(input_dfb_id, Ht * chunk_size, total_input_tiles)));
        PACK((assert_output_dfb_size<input_policy>(output_dfb_id, total_output_tiles)));

        // No-pop modes: bulk reserve output upfront
        if constexpr (!should_pop(input_policy)) {
            output_dfb.reserve_back(total_output_tiles);
        }

        // PersistentPolicy: wait for all tiles upfront
        if constexpr (waits_upfront(input_policy)) {
            input_dfb.wait_front(total_input_tiles);
        }

        uint32_t batch_offset = 0;
        for (uint32_t nc = 0; nc < num_batches; ++nc) {
            for (uint32_t wt = 0; wt < Wt; wt += chunk_size) {
                uint32_t chunk_end = (wt + chunk_size < Wt) ? (wt + chunk_size) : Wt;
                uint32_t current_chunk = chunk_end - wt;
                uint32_t tiles_in_chunk = Ht * current_chunk;

                // BulkWaitBulkPop: wait for entire chunk upfront
                if constexpr (waits_bulk(input_policy)) {
                    input_dfb.wait_front(tiles_in_chunk);
                }

                tile_regs_acquire();

                // Reload accumulator if needed (zero overhead when AccumulateT is NoAccumulation)
                reload_accumulator_if_needed<reduce_type, reduce_dim, reduce_format, AccumulateT, is_sfpu>(
                    accum_dfb, input_dfb_id, scaler_dfb_id, accumulate);
                if constexpr (is_sfpu) {
                    if (Ht > 1) {
                        detail::sfpu_reduce_fold_init<reduce_type, reduce_format>();
                    }
                }

                for (uint32_t ht = 0; ht < Ht; ++ht) {
                    // Base dst_index: from accumulation config or 0 for multi-column output
                    uint32_t dst_idx = get_dst_index(accumulate);
                    // Last H-tile picks up the partial scaler when one was prepared by the reader.
                    const uint32_t scaler_idx = (ht == Ht - 1) ? partial_scaler.last_tile_scaler_idx : 0;
                    for (uint32_t i = wt; i < chunk_end; ++i) {
                        if constexpr (is_sfpu) {
                            const bool is_first_tile = detail::sfpu_is_first_tile(ht, accumulate);
                            constexpr uint32_t sfpu_work_dst = chunk_size;
                            if constexpr (waits_per_tile(input_policy)) {
                                input_dfb.wait_front(onetile);
                                detail::sfpu_copy_and_fold<reduce_type, reduce_format>(
                                    input_dfb_id, 0, dst_idx, sfpu_work_dst, is_first_tile);
                                input_dfb.pop_front(onetile);
                            } else if constexpr (waits_bulk(input_policy)) {
                                const uint32_t tile_idx = ht * current_chunk + (i - wt);
                                detail::sfpu_copy_and_fold<reduce_type, reduce_format>(
                                    input_dfb_id, tile_idx, dst_idx, sfpu_work_dst, is_first_tile);
                            } else {
                                const uint32_t tile_idx = batch_offset + ht * stride + i;
                                detail::sfpu_copy_and_fold<reduce_type, reduce_format>(
                                    input_dfb_id, tile_idx, dst_idx, sfpu_work_dst, is_first_tile);
                            }
                        } else if constexpr (waits_per_tile(input_policy)) {
                            // One-at-a-time: wait/pop per tile
                            input_dfb.wait_front(onetile);
                            reduce_tile<reduce_type, reduce_dim>(input_dfb_id, scaler_dfb_id, 0, scaler_idx, dst_idx);
                            input_dfb.pop_front(onetile);
                        } else if constexpr (waits_bulk(input_policy)) {
                            // BulkWaitBulkPop: use indexed access
                            uint32_t tile_idx = ht * current_chunk + (i - wt);
                            reduce_tile<reduce_type, reduce_dim>(
                                input_dfb_id, scaler_dfb_id, tile_idx, scaler_idx, dst_idx);
                        } else {  // PreloadedPolicy or PersistentPolicy: indexed access
                            uint32_t tile_idx = batch_offset + ht * stride + i;
                            reduce_tile<reduce_type, reduce_dim>(
                                input_dfb_id, scaler_dfb_id, tile_idx, scaler_idx, dst_idx);
                        }
                        ++dst_idx;
                    }
                }

                // SFPU intra-tile finalize per output slot
                if constexpr (is_sfpu) {
                    const uint32_t sfpu_base_dst = get_dst_index(accumulate);
                    sfpu_reduce_init<reduce_type, reduce_format>();
                    for (uint32_t k = 0; k < current_chunk; ++k) {
                        sfpu_reduce<reduce_type, reduce_format, reduce_dim>(
                            sfpu_base_dst + k, /*ct_dim=*/1, /*rt_dim=*/1);
                    }
                }

                // Post-reduce operation for each output tile in chunk
                const uint32_t base_dst = get_dst_index(accumulate);
                for (uint32_t i = 0; i < current_chunk; ++i) {
                    post_reduce_op(base_dst + i);
                }

                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < current_chunk; ++i) {
                    // Pop modes: reserve/push per output tile
                    if constexpr (should_pop(input_policy)) {
                        output_dfb.reserve_back(onetile);
                    }
                    pack_tile(base_dst + i, output_dfb_id);
                    if constexpr (should_pop(input_policy)) {
                        output_dfb.push_back(onetile);
                    }
                }
                tile_regs_release();

                // BulkWaitBulkPop: pop all tiles after processing
                if constexpr (waits_bulk(input_policy)) {
                    input_dfb.pop_front(tiles_in_chunk);
                }
            }
            // Update batch_offset for indexed modes (PreloadedPolicy and PersistentPolicy)
            if constexpr (!should_pop(input_policy)) {
                batch_offset += tiles_per_bulk;
            }
        }

        // No-pop modes: bulk push output at end
        if constexpr (!should_pop(input_policy)) {
            output_dfb.push_back(total_output_tiles);
        }
    }

    // Cleanup
    if constexpr (is_sfpu) {
        PACK((llk_pack_reduce_mask_clear()));
    } else {
        reduce_uninit();
    }
}

// =============================================================================
// Mean = reduce<SUM> + an explicit caller-supplied 1/N normalization (see reduce_mean docs in the header).
// =============================================================================
template <
    ReduceDim reduce_dim,
    uint32_t input_dfb_id,
    uint32_t scaler_dfb_id,
    uint32_t output_dfb_id,
    ReduceInputPolicy input_policy,
    ReduceDataFormatReconfigMode reconfig_mode,
    ReduceAlgorithm algorithm,
    typename AccumulateT>
ALWI void reduce_mean(
    ReduceInputBlockShape input_block_shape,
    uint32_t n_reduced,
    ReduceInputMemoryLayout input_memory_layout,
    AccumulateT accumulate,
    ReducePartialScaler partial_scaler) {
    ASSERT(n_reduced > 0);
    // 1/N as float bits for mul_unary_tile. N is the caller's true reduced-element count; the kernel never
    // derives it from tile geometry. The multiply lands in a finalize post_reduce_op, which reduce() runs
    // only on the finalizing chunk (so for cross-call Accumulate the 1/N applies once, to the grand total).
    float inv_f = 1.0f / static_cast<float>(n_reduced);
    uint32_t inv_bits = 0;
    __builtin_memcpy(&inv_bits, &inv_f, sizeof(inv_bits));
    reduce<
        PoolType::SUM,
        reduce_dim,
        input_dfb_id,
        scaler_dfb_id,
        output_dfb_id,
        input_policy,
        reconfig_mode,
        algorithm>(
        input_block_shape,
        input_memory_layout,
        accumulate,
        [inv_bits](uint32_t dst) { mul_unary_tile(dst, inv_bits); },
        partial_scaler);
}

}  // namespace compute_kernel_lib
