// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Implementation file for reduce_helpers_compute.hpp
// Do not include directly - include reduce_helpers_compute.hpp instead

#include "api/compute/add_int_sfpu.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
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

// SFPU cross-tile add. Int32 uses add_int_tile; Float32 uses add_binary_tile for
// accurate fp32 accumulation. add_binary_tile is unavailable on Quasar, so guard
// it with ARCH_QUASAR to avoid template lookup failures.
template <DataFormat format>
ALWI void sfpu_reduce_sum_fold_init() {
    if constexpr (format == DataFormat::Int32) {
        add_int_tile_init();
    } else {
#ifndef ARCH_QUASAR
        add_binary_tile_init();
#else
        static_assert(format == DataFormat::Int32, "Accurate fp32 SFPU mean is not supported on Quasar");
#endif
    }
}

template <DataFormat format>
ALWI void sfpu_reduce_sum_fold_tile(uint32_t a, uint32_t b, uint32_t out) {
    if constexpr (format == DataFormat::Int32) {
        add_int_tile<format>(a, b, out);
    } else {
#ifndef ARCH_QUASAR
        add_binary_tile(a, b, out);
#else
        static_assert(format == DataFormat::Int32, "Accurate fp32 SFPU mean is not supported on Quasar");
#endif
    }
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

// -----------------------------------------------------------------------------
// AccumulateViaAdd datapath (ReduceAlgorithm::AccumulateViaAdd).
//
// Each output tile is produced independently: sum its reduce-dim tiles into DST[0] with pairwise
// add_tiles(acc_to_dest) (parity resolved at the seed — copy_tile one tile when the count is odd, add
// the first pair when even, no phantom zero tile), finalize within the tile on the SFPU (sfpu_reduce
// SUM, which reads DST in place), and for AVG multiply by 1/N once. One DST register per output tile, so
// an arbitrary (Ht, Wt, NC) block is handled without the REDUCE_COL DST/chunk limit.
//
// Restrictions (enforced by reduce()): float SUM/AVG, NoAccumulation, BulkWaitBulkPop. The whole block is
// waited/popped in bulk. One-time init (binary_op_init_common, sfpu_reduce_init) is hoisted OUT of the
// per-output loop; only the light MOP inits (add_tiles/copy) run per output.
//
// PARTIAL (non-tile-aligned) reduce dims — ROW/COL only, signalled by partial_scaler.valid_reduce_dim_elements
// (= P valid elements in the LAST reduce-dim tile): the last tile is folded in with a DEST-ACCUMULATING
// masked broadcast-mul (0/1 mask tile at scaler_dfb_id[last_tile_scaler_idx]; row-0 mask for ROW /
// mul_tiles_bcast_rows, col-0 for COL / mul_tiles_bcast_cols), so the padding contributes 0. The bulk stays
// pure add_tiles (fidelity-flat, 2 tiles/op); only the one partial tile is a (fidelity-affected) mul, and
// the mean divides by the true count (full_cnt*32 + P). The bcast shorthands overwrite DEST
// (clear_fp32_dst_acc=true), so the accumulating variant is the LLK directly with acc_to_dest=1 at init and
// clear_fp32_dst_acc=false at the op.
//
// CROSS-CALL ACCUMULATE (AccumulateT == Accumulate, BulkWaitBulkPop) — the accumulator CB holds the RAW
// partial-sum tile per output (NOT a reduced tile). On the first chunk (is_first) we just sum this chunk's
// tiles. On every later chunk we fold the accumulator into the SAME pairwise add NATIVELY — no
// binary_dest_reuse — with the parity of full_cnt (== cnt; accumulate rejects partial) deciding how:
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
    typename AccumulateT,
    typename PostReduceOp>
ALWI void reduce_accumulate_via_add(
    ReduceInputBlockShape shape,
    ReducePartialScaler partial_scaler,
    AccumulateT accumulate,
    PostReduceOp post_reduce_op) {
    const uint32_t Ht = shape.rows, Wt = shape.cols, NC = shape.batches;
    const uint32_t in_tiles = Ht * Wt * NC;

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

    // tiles that collapse into one output, and their stride in the row-major (batch-major) input block.
    const uint32_t cnt = is_row ? Wt : (is_col ? Ht : (Ht * Wt));
    const uint32_t stride = is_col ? Wt : 1u;
    const uint32_t n_out = is_row ? (Ht * NC) : (is_col ? (Wt * NC) : NC);

    const bool has_partial = (partial_scaler.valid_reduce_dim_elements > 0);
    const uint32_t P = partial_scaler.valid_reduce_dim_elements;
    const uint32_t mask_idx = partial_scaler.last_tile_scaler_idx;
    const uint32_t full_cnt = has_partial ? (cnt - 1u) : cnt;  // tiles summed via pure add_tiles
    // reduced element count for the AVG mean (partial -> true count; scalar is always tile-aligned).
    const uint32_t n_elems = (reduce_dim == ReduceDim::REDUCE_SCALAR)
                                 ? (cnt * 1024u)
                                 : (has_partial ? (full_cnt * 32u + P) : (cnt * 32u));
    float inv_f = 1.0f / static_cast<float>(n_elems);
    uint32_t inv_bits = 0;
    __builtin_memcpy(&inv_bits, &inv_f, sizeof(inv_bits));  // 1/N as float bits for mul_unary_tile

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

    // one-time setup (per reduce() call) — NEVER inside the per-output loop.
    binary_op_init_common(input_dfb_id, input_dfb_id, output_dfb_id);
    sfpu_reduce_init<PoolType::SUM, dst_fmt>();  // SFPU reduce macro persists across the FPU adds (replay)
    if (has_partial) {
        scaler_dfb.wait_front(mask_idx + 1);  // 0/1 mask tile (row-0 for ROW, col-0 for COL)
    }
    if constexpr (!streaming) {
        input_dfb.wait_front(in_tiles);  // resident block, indexed per output
    }

    for (uint32_t o = 0; o < n_out; ++o) {
        tile_regs_acquire();

        if constexpr (streaming) {
            // Stream this output's `cnt` reduce-dim tiles through DST in pairs, waiting/popping as they
            // arrive (front-relative indices 0/1). Contiguous per output (row/scalar), so tiles arrive in
            // reduce order; DST holds the running sum across the whole stream.
            uint32_t consumed;
            if (cnt & 1u) {
                input_dfb.wait_front(1);
                copy_tile_init(input_dfb_id);
                copy_tile(input_dfb_id, 0, 0);
                input_dfb.pop_front(1);
                consumed = 1;
            } else {
                input_dfb.wait_front(2);
                add_tiles_init(input_dfb_id, input_dfb_id, false);
                add_tiles(input_dfb_id, input_dfb_id, 0, 1, 0);
                input_dfb.pop_front(2);
                consumed = 2;
            }
            add_tiles_init(input_dfb_id, input_dfb_id, true);
            for (; consumed < cnt; consumed += 2) {
                input_dfb.wait_front(2);
                add_tiles(input_dfb_id, input_dfb_id, 0, 1, 0);
                input_dfb.pop_front(2);
            }
        } else {
            // Indexed access into the resident block; `start` is output o's first reduce-dim tile.
            uint32_t start;
            if constexpr (is_row) {
                start = o * Wt;
            } else if constexpr (is_col) {
                start = (o / Wt) * (Ht * Wt) + (o % Wt);
            } else {
                start = o * (Ht * Wt);
            }

            if constexpr (has_accum) {
                if (accumulate.is_first()) {
                    // First chunk: no accumulator yet — sum this chunk's full_cnt tiles (aligned; accumulate
                    // rejects partial), parity resolved at the seed.
                    uint32_t k;
                    if (full_cnt & 1u) {
                        copy_tile_init(input_dfb_id);
                        copy_tile(input_dfb_id, start, 0);
                        k = 1;
                    } else {
                        add_tiles_init(input_dfb_id, input_dfb_id, false);
                        add_tiles(input_dfb_id, input_dfb_id, start, start + stride, 0);
                        k = 2;
                    }
                    add_tiles_init(input_dfb_id, input_dfb_id, true);
                    for (; k < full_cnt; k += 2) {
                        add_tiles(input_dfb_id, input_dfb_id, start + k * stride, start + (k + 1) * stride, 0);
                    }
                } else {
                    // Later chunk: fold output o's running accumulator (raw partial sum, front of accum CB)
                    // into the SAME pairwise add — natively, no binary_dest_reuse. Parity of full_cnt decides.
                    const uint32_t acc_cb = accumulate.config.cb_accumulator;
                    accum_dfb.wait_front(1);
                    if (full_cnt & 1u) {
                        // odd new-tile count: accumulator is the LAST add's SECOND (SRCB) operand — no dest
                        // reload. add_tiles_init does NOT reconfigure the data format, so reconfigure SRCB
                        // around the acc-add (accumulator CB may be fp32 while the input is bf16) and restore
                        // it to the input after — mirrors the standard reload_accumulator_if_needed's SRCA
                        // reconfig, on SRCB because the accumulator is operand B here.
                        if (full_cnt == 1u) {
                            reconfig_data_format_srcb(input_dfb_id, acc_cb);
                            add_tiles_init(input_dfb_id, acc_cb, false);
                            add_tiles(input_dfb_id, acc_cb, start, 0, 0);  // DST = new[0] + accumulator
                            reconfig_data_format_srcb(acc_cb, input_dfb_id);
                        } else {
                            add_tiles_init(input_dfb_id, input_dfb_id, false);
                            add_tiles(input_dfb_id, input_dfb_id, start, start + stride, 0);  // seed new pair
                            add_tiles_init(input_dfb_id, input_dfb_id, true);
                            for (uint32_t k = 2; k + 1 < full_cnt; k += 2) {
                                add_tiles(input_dfb_id, input_dfb_id, start + k * stride,
                                          start + (k + 1) * stride, 0);
                            }
                            reconfig_data_format_srcb(input_dfb_id, acc_cb);
                            add_tiles_init(input_dfb_id, acc_cb, true);  // last new tile + accumulator
                            add_tiles(input_dfb_id, acc_cb, start + (full_cnt - 1u) * stride, 0, 0);
                            reconfig_data_format_srcb(acc_cb, input_dfb_id);
                        }
                    } else {
                        // even new-tile count: ONE dest reload (copy accumulator as the seed), then the new
                        // tiles in pairs. copy_tile_init does NOT reconfigure the data format, so reconfigure
                        // SRCA around the copy (accumulator CB may be fp32 while the input is bf16) and restore
                        // it to the input before the adds — mirrors the standard reload_accumulator_if_needed.
                        reconfig_data_format_srca(input_dfb_id, acc_cb);
                        copy_tile_init(acc_cb);
                        copy_tile(acc_cb, 0, 0);  // DST = accumulator
                        reconfig_data_format_srca(acc_cb, input_dfb_id);
                        add_tiles_init(input_dfb_id, input_dfb_id, true);
                        for (uint32_t k = 0; k < full_cnt; k += 2) {
                            add_tiles(input_dfb_id, input_dfb_id, start + k * stride, start + (k + 1) * stride, 0);
                        }
                    }
                    accum_dfb.pop_front(1);
                }
            } else {
                uint32_t k;
                if (full_cnt & 1u) {
                    copy_tile_init(input_dfb_id);
                    copy_tile(input_dfb_id, start, 0);
                    k = 1;
                } else {
                    add_tiles_init(input_dfb_id, input_dfb_id, false);
                    add_tiles(input_dfb_id, input_dfb_id, start, start + stride, 0);
                    k = 2;
                }
                add_tiles_init(input_dfb_id, input_dfb_id, true);
                for (; k < full_cnt; k += 2) {
                    add_tiles(input_dfb_id, input_dfb_id, start + k * stride, start + (k + 1) * stride, 0);
                }
                // partial: fold the LAST reduce-dim tile in, masked, ACCUMULATING into DST (acc_to_dest=1 at
                // init, clear_fp32_dst_acc=false at the op — the bcast shorthands would overwrite).
                if (has_partial) {
                    const uint32_t last = start + full_cnt * stride;
                    MATH((llk_math_eltwise_binary_init<ckernel::EltwiseBinaryType::ELWMUL, MASK_BCAST,
                                                       MATH_FIDELITY>(input_dfb_id, scaler_dfb_id, 1)));
                    UNPACK((llk_unpack_AB_init<MASK_BCAST>(input_dfb_id, scaler_dfb_id)));
                    UNPACK((llk_unpack_AB<MASK_BCAST>(input_dfb_id, scaler_dfb_id, last, mask_idx)));
                    MATH((llk_math_eltwise_binary<ckernel::EltwiseBinaryType::ELWMUL, MASK_BCAST, DST_ACCUM_MODE,
                                                  MATH_FIDELITY>(0, false)));
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
            if constexpr (reduce_type == PoolType::AVG) {
                mul_unary_tile(0, inv_bits);  // mean (no binop_with_scalar init needed after sfpu_reduce)
            }
            post_reduce_op(0);
        }

        tile_regs_commit();
        tile_regs_wait();
        output_dfb.reserve_back(1);
        pack_tile(0, output_dfb_id);  // output tile o -> page o
        output_dfb.push_back(1);
        tile_regs_release();
    }
    if constexpr (!streaming) {
        input_dfb.pop_front(in_tiles);
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
    UNPACK((llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE, p_dim_stride_target::IGNORE>(old_dfb_id, srca_dfb_id)));
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
    ReduceFp32Mode fp32_mode,
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
        (reduce_type != PoolType::MAX && reduce_type != PoolType::SUM) ||
            reduce_dim != ReduceDim::REDUCE_SCALAR || reduce_format != DataFormat::Int32,
        "Int32 MAX/SUM REDUCE_SCALAR is not supported (host decomposes Int32 HW reduce into W-then-H)");
    static_assert(
        reduce_type != PoolType::AVG || reduce_format != DataFormat::Int32,
        "Int32 AVG (mean) is not supported");
    static_assert(
        is_accumulation_type_v<AccumulateT>,
        "AccumulateT must be a valid accumulation type (NoAccumulation or Accumulate)");
    static_assert(
        is_post_reduce_op_v<PostReduceOp>,
        "PostReduceOp must be callable with a uint32_t argument");
    static_assert(
        !is_accumulate_v<AccumulateT> ||
            !(reduce_type == PoolType::MAX && reduce_dim == ReduceDim::REDUCE_SCALAR),
        "Accumulate with PoolType::MAX + REDUCE_SCALAR is not supported: the pack edge mask "
        "keeps only DST(0,0), but GMPOOL needs that running max broadcast across face-0 row 4 "
        "on the reload pass, which the current copy_tile reload cannot reproduce.");
#ifdef ARCH_QUASAR
    // The MAX + REDUCE_ROW accumulator reload relies on a within-16x16-face transpose during
    // copy_tile_to_dst_init_short (see reload_accumulator_if_needed). That transpose is rejected
    // by copy_tile_to_dst_init_short on Quasar ("Transpose within face not supported on Quasar"),
    // and there is no Quasar-compatible reload that restores the layout GMPOOL expects.
    static_assert(
        !is_accumulate_v<AccumulateT> ||
            !(reduce_type == PoolType::MAX && reduce_dim == ReduceDim::REDUCE_ROW),
        "Accumulate with PoolType::MAX + REDUCE_ROW is not supported on Quasar: the accumulator "
        "reload requires a within-16x16-face transpose, which copy_tile_to_dst_init_short asserts "
        "against on Quasar.");
#endif

    // =============================================================================
    // Algorithm selection. Auto resolves to a concrete datapath (for now always ReduceTile; a cost
    // heuristic will choose later). AccumulateViaAdd is a restricted, faster datapath for wide float
    // SUM/AVG reduces; anything it cannot express is rejected here (compile-time where possible) and
    // must use ReduceTile. ReduceTile (and Auto) fall through to the standard body below.
    // =============================================================================
    constexpr ReduceAlgorithm resolved_algorithm =
        (algorithm == ReduceAlgorithm::Auto) ? ReduceAlgorithm::ReduceTile : algorithm;
    if constexpr (resolved_algorithm == ReduceAlgorithm::AccumulateViaAdd) {
        static_assert(
            reduce_type == PoolType::SUM || reduce_type == PoolType::AVG,
            "AccumulateViaAdd: SUM/AVG only — an additive accumulate cannot express MAX/MIN. Use ReduceTile.");
        static_assert(
            reduce_format != DataFormat::Int32,
            "AccumulateViaAdd: float only (add_tiles + sfpu_reduce). Int32 must use ReduceTile.");
        static_assert(
            input_policy == ReduceInputPolicy::BulkWaitBulkPop ||
                input_policy == ReduceInputPolicy::WaitAndPopPerTile,
            "AccumulateViaAdd: only BulkWaitBulkPop (resident block, indexed) and WaitAndPopPerTile "
            "(streaming) are implemented.");
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
        if constexpr (input_policy == ReduceInputPolicy::WaitAndPopPerTile) {
            ASSERT(partial_scaler.valid_reduce_dim_elements == 0);  // streaming is aligned-only
        }
        // Partial (non-tile-aligned) reduce dims are supported for ROW/COL under BulkWaitBulkPop (the last
        // reduce-dim tile is folded in with a masked accumulating broadcast-mul; valid_reduce_dim_elements =
        // P + a 0/1 mask tile in scaler_dfb). REDUCE_SCALAR can be partial in BOTH axes at once (a single
        // row/col mask can't express the corner), so it is rejected — use ReduceTile.
        if constexpr (reduce_dim == ReduceDim::REDUCE_SCALAR) {
            ASSERT(partial_scaler.valid_reduce_dim_elements == 0);
        }
        // Cross-call Accumulate folds whole raw tiles; the masked partial-tile path is not wired into the
        // accumulator fold, so a partial reduce must be a standalone (NoAccumulation) reduce.
        if constexpr (is_accumulate_v<AccumulateT>) {
            ASSERT(partial_scaler.valid_reduce_dim_elements == 0);
        }
        detail::reduce_accumulate_via_add<reduce_type, reduce_dim, input_dfb_id, scaler_dfb_id, output_dfb_id,
                                          input_policy, AccumulateT, PostReduceOp>(
            input_block_shape, partial_scaler, accumulate, post_reduce_op);
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

    constexpr bool is_sfpu = is_sfpu_reduce_path<reduce_type, reduce_dim, reduce_format, fp32_mode>();

    DataflowBuffer input_dfb(input_dfb_id);
    DataflowBuffer scaler_dfb(scaler_dfb_id);
    DataflowBuffer output_dfb(output_dfb_id);
    DataflowBuffer accum_dfb([&]() -> uint32_t {
        if constexpr (enable_accumulation) { return accumulate.config.cb_accumulator; }
        else { return 0; }
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
                        reduce_tile<reduce_type, reduce_dim>(
                            input_dfb_id, scaler_dfb_id, tile_idx, 0, dst_idx);
                    } else {  // PreloadedPolicy or PersistentPolicy: indexed access
                        uint32_t tile_idx = batch_offset + ht * stride + wt;
                        reduce_tile<reduce_type, reduce_dim>(
                            input_dfb_id, scaler_dfb_id, tile_idx, 0, dst_idx);
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
                            reduce_tile<reduce_type, reduce_dim>(
                                input_dfb_id, scaler_dfb_id, 0, scaler_idx, dst_idx);
                            input_dfb.pop_front(onetile);
                        } else if constexpr (waits_bulk(input_policy)) {
                            // BulkWaitBulkPop: use indexed access
                            reduce_tile<reduce_type, reduce_dim>(
                                input_dfb_id, scaler_dfb_id, wt, scaler_idx, dst_idx);
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
                            reduce_tile<reduce_type, reduce_dim>(
                                input_dfb_id, scaler_dfb_id, 0, scaler_idx, dst_idx);
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

}  // namespace compute_kernel_lib
