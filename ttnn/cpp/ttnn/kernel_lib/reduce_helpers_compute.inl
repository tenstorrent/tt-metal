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
#include "tools/profiler/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dfb_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_common.hpp"

namespace compute_kernel_lib {

namespace detail {

// SFPU MAX fold
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

// SFPU MIN fold
template <DataFormat format>
ALWI void sfpu_reduce_min_fold_init() {
    static_assert(format == DataFormat::Int32, "SFPU reduce MIN fold: Int32 only");
    binary_min_int32_tile_init();
}

template <DataFormat format>
ALWI void sfpu_reduce_min_fold_tile(uint32_t a, uint32_t b, uint32_t out) {
    static_assert(format == DataFormat::Int32, "SFPU reduce MIN fold: Int32 only");
    binary_min_int32_tile(a, b, out);
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

// Pool-type dispatched cross-tile fold init (MAX -> binary_max, MIN -> binary_min, SUM -> add_int).
// Used by compute_kernel_lib::reduce() for the Int32 SFPU path.
template <PoolType pool_type, DataFormat format>
ALWI void sfpu_reduce_fold_init() {
    if constexpr (pool_type == PoolType::SUM) {
        sfpu_reduce_sum_fold_init<format>();
    } else if constexpr (pool_type == PoolType::MIN) {
        sfpu_reduce_min_fold_init<format>();
    } else {
        sfpu_reduce_max_fold_init<format>();
    }
}

// Copy one input tile into DST and fold into the running accumulator (first tile seeds dst_idx
// directly). Fold op is selected by pool_type: MAX -> running max, MIN -> running min, SUM -> running sum.
template <PoolType pool_type, DataFormat format>
ALWI void sfpu_copy_and_fold(
    uint32_t input_cb_id, uint32_t tile_idx, uint32_t dst_idx, uint32_t work_dst, bool is_first_tile) {
    if (is_first_tile) {
        copy_tile(input_cb_id, tile_idx, dst_idx);
    } else {
        copy_tile(input_cb_id, tile_idx, work_dst);
        if constexpr (pool_type == PoolType::SUM) {
            sfpu_reduce_sum_fold_tile<format>(dst_idx, work_dst, dst_idx);
        } else if constexpr (pool_type == PoolType::MIN) {
            sfpu_reduce_min_fold_tile<format>(dst_idx, work_dst, dst_idx);
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

// FoldViaAdd reads an accumulator through SrcA/SrcB, which is invalid for a 32-bit CB configured
// to unpack directly to DEST. PACK cannot inspect unpack_dst_format, so this check is UNPACK/MATH only.
ALWI bool dfb_unpacks_to_dest(uint32_t dfb_id) {
#if defined(UCK_CHLKC_PACK)
    (void)dfb_id;
    return false;
#else
    const uint32_t src = unpack_src_format[dfb_id];
    const bool src_is_32bit = src == (uint32_t)DataFormat::Float32 || src == (uint32_t)DataFormat::Int32;
    return src_is_32bit && src == unpack_dst_format[dfb_id];
#endif
}

// Indexed UNPACK addresses do not wrap at the end of a WH/BH circular buffer. Broadcast the
// UNPACK thread's bounds decision so every TRISC takes the same pair-vs-single control-flow path.
// Quasar's multi-tile-counter DFB layout does not yet expose an equivalent cross-TRISC query, so
// its streaming path conservatively consumes one page at a time.
ALWI bool stream_front_pair_is_contiguous(uint32_t dfb_id) {
#ifdef ARCH_QUASAR
    (void)dfb_id;
    return false;
#else
    bool contiguous = false;
    UNPACK({
        contiguous = cb_access_within_bounds(dfb_id, 0, 2);
        mailbox_write(ckernel::ThreadId::MathThreadId, static_cast<uint32_t>(contiguous));
        mailbox_write(ckernel::ThreadId::PackThreadId, static_cast<uint32_t>(contiguous));
    })
    MATH(contiguous = mailbox_read(ckernel::ThreadId::UnpackThreadId) != 0;)
    PACK(contiguous = mailbox_read(ckernel::ThreadId::UnpackThreadId) != 0;)
    return contiguous;
#endif
}

// Add the reduce-axis tiles into DST[0], then perform the within-tile collapse on the SFPU.
// With cross-call Accumulate, non-last calls instead pack the raw partial-sum tile and the final
// call reloads it, folds in the last chunk, and performs the collapse exactly once.
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
    const uint32_t Ht = shape.rows;
    const uint32_t Wt = shape.cols;
    const uint32_t NC = shape.batches;
    const uint32_t row_pitch = input_memory_layout.row_stride > 0u ? input_memory_layout.row_stride : Wt;
    const uint32_t in_tiles = Ht * row_pitch * NC;

    constexpr DataFormat dst_fmt = DST_ACCUM_MODE ? DataFormat::Float32 : DataFormat::Float16_b;
    constexpr bool is_row = reduce_dim == ReduceDim::REDUCE_ROW;
    constexpr bool is_col = reduce_dim == ReduceDim::REDUCE_COL;
    constexpr auto mask_bcast = is_col ? ckernel::BroadcastType::COL : ckernel::BroadcastType::ROW;
    constexpr bool streaming = input_policy == ReduceInputPolicy::WaitAndPopPerTile;
    constexpr bool has_accum = is_accumulate_v<AccumulateT>;

    constexpr bool should_pop_p =
        input_policy == ReduceInputPolicy::WaitAndPopPerTile || input_policy == ReduceInputPolicy::BulkWaitBulkPop;
    constexpr bool no_wait_p = input_policy == ReduceInputPolicy::NoWaitNoPop;
    constexpr bool helper_waits_block = !streaming && !no_wait_p;
    constexpr bool helper_pops_block = !streaming && should_pop_p;

    const uint32_t cnt = is_row ? Wt : (is_col ? Ht : Ht * Wt);
    const uint32_t stride = is_col ? row_pitch : 1u;
    const uint32_t n_out = is_row ? Ht * NC : (is_col ? Wt * NC : NC);
    const bool has_partial = partial_scaler.valid_reduce_dim_elements > 0;
    const uint32_t mask_idx = partial_scaler.mask_tile_idx;
    const uint32_t full_cnt = has_partial ? cnt - 1u : cnt;

    DataflowBuffer input_dfb(input_dfb_id);
    DataflowBuffer scaler_dfb(scaler_dfb_id);
    DataflowBuffer output_dfb(output_dfb_id);
    DataflowBuffer accum_dfb([&]() -> uint32_t {
        if constexpr (has_accum) {
            return accumulate.config.cb_accumulator;
        } else {
            return 0;
        }
    }());

    bool do_finalize = true;
    if constexpr (has_accum) {
        do_finalize = accumulate.is_last();
    }

    // A middle accumulation call commonly reads and rewrites the same exactly-sized accumulator CB. It cannot
    // bulk-reserve every output before consuming the old pages: that would wait forever for space which this
    // call itself must release. Cycle that CB one page at a time; retain the normal bulk output contract for
    // first/final no-pop calls whose output does not alias a live accumulator.
    bool cycles_accumulator = false;
    if constexpr (has_accum) {
        cycles_accumulator = !accumulate.is_first() && output_dfb_id == accumulate.config.cb_accumulator;
    }

    constexpr bool reconfig_in = reconfig_mode == ReduceDataFormatReconfigMode::INPUT ||
                                 reconfig_mode == ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT;
    constexpr bool reconfig_out = reconfig_mode == ReduceDataFormatReconfigMode::OUTPUT ||
                                  reconfig_mode == ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT;
    if constexpr (reconfig_in) {
        reconfig_data_format(input_dfb_id, input_dfb_id);
    }
    if constexpr (reconfig_out) {
        pack_reconfig_data_format(output_dfb_id);
    }
    sfpu_reduce_init<PoolType::SUM, dst_fmt>();

    ASSERT(input_dfb_id != output_dfb_id && Ht > 0 && Wt > 0 && NC > 0);
#ifndef ARCH_QUASAR
    UNPACK(ASSERT(is_valid_dfb_tile_page_size(input_dfb_id, (DataFormat)unpack_src_format[input_dfb_id])));
    PACK(ASSERT(is_valid_dfb_tile_page_size(output_dfb_id, (DataFormat)pack_dst_format[output_dfb_id])));
#endif
    if constexpr (no_wait_p) {
        UNPACK(ASSERT(get_dfb_num_pages(input_dfb_id) >= in_tiles));
    }
    if constexpr (streaming) {
        UNPACK(ASSERT(get_dfb_num_pages(input_dfb_id) >= 1u));
    }

    bool wait_scaler = has_partial;
    if constexpr (has_accum) {
        [[maybe_unused]] const uint32_t acc_cb = accumulate.config.cb_accumulator;
        ASSERT(input_dfb_id != acc_cb);
        // Every output owns one raw partial-sum page between calls. This is stricter than the per-page
        // waits below: a smaller ring could make the first call block after filling only part of the state.
        UNPACK(ASSERT(get_dfb_num_pages(acc_cb) >= n_out));
        ASSERT(accumulate.config.dst_index == 0u);
        UNPACK(ASSERT(accumulate.reload != AccumulateReloadMode::FoldViaAdd || !dfb_unpacks_to_dest(acc_cb)));
        ASSERT(accumulate.reload != AccumulateReloadMode::CopySeedZeroPair || !has_partial);
#ifdef ARCH_QUASAR
        ASSERT(accumulate.reload != AccumulateReloadMode::CopySeedSfpuAdd);
#endif
        wait_scaler = wait_scaler || accumulate.reload == AccumulateReloadMode::CopySeedZeroPair;
    }
    if (wait_scaler) {
        ASSERT(input_dfb_id != scaler_dfb_id && output_dfb_id != scaler_dfb_id);
        scaler_dfb.wait_front(mask_idx + 1);
    }
    if constexpr (helper_waits_block) {
        input_dfb.wait_front(in_tiles);
    }
    if constexpr (!should_pop_p) {
        if (!cycles_accumulator) {
            output_dfb.reserve_back(n_out);
        }
    }

    // Fold a masked final tile while preserving the sum already resident in DST[0].
    [[maybe_unused]] auto fold_partial_last = [&](uint32_t last_idx) {
        MATH((llk_math_eltwise_binary_init<ckernel::EltwiseBinaryType::ELWMUL, mask_bcast, MATH_FIDELITY>(
            input_dfb_id, scaler_dfb_id, 1)));
        UNPACK((llk_unpack_AB_init<mask_bcast>(input_dfb_id, scaler_dfb_id)));
        UNPACK((llk_unpack_AB<mask_bcast>(input_dfb_id, scaler_dfb_id, last_idx, mask_idx)));
        MATH((llk_math_eltwise_binary<ckernel::EltwiseBinaryType::ELWMUL, mask_bcast, DST_ACCUM_MODE, MATH_FIDELITY>(
            0, false)));
    };

    // Mask a partial-only stream into an otherwise empty DST[0].
    [[maybe_unused]] auto seed_partial_last = [&](uint32_t last_idx) {
        MATH((llk_math_eltwise_binary_init<ckernel::EltwiseBinaryType::ELWMUL, mask_bcast, MATH_FIDELITY>(
            input_dfb_id, scaler_dfb_id, 1)));
        UNPACK((llk_unpack_AB_init<mask_bcast>(input_dfb_id, scaler_dfb_id)));
        UNPACK((llk_unpack_AB<mask_bcast>(input_dfb_id, scaler_dfb_id, last_idx, mask_idx)));
        MATH((llk_math_eltwise_binary<ckernel::EltwiseBinaryType::ELWMUL, mask_bcast, DST_ACCUM_MODE, MATH_FIDELITY>(
            0, true)));
    };

    for (uint32_t output_idx = 0; output_idx < n_out; ++output_idx) {
        tile_regs_acquire();
        uint32_t deferred_stream_pop = 0;
        bool deferred_accum_pop = false;

        if constexpr (streaming) {
            bool dst_seeded = false;
            if constexpr (has_accum) {
                if (!accumulate.is_first()) {
                    const uint32_t acc_cb = accumulate.config.cb_accumulator;
                    accum_dfb.wait_front(1);
                    deferred_accum_pop = true;
                    reconfig_data_format_srca(input_dfb_id, acc_cb);
                    copy_tile_init(acc_cb);
                    copy_tile(acc_cb, 0, 0);
                    reconfig_data_format_srca(acc_cb, input_dfb_id);
                    dst_seeded = true;
                }
            }

            uint32_t remaining = full_cnt;
            while (remaining > 0u) {
                const bool consume_pair = remaining >= 2u && stream_front_pair_is_contiguous(input_dfb_id);
                uint32_t consumed_pages = 1;
                if (consume_pair) {
                    input_dfb.wait_front(2);
                    add_init(input_dfb_id, input_dfb_id, true);
                    add_tiles(input_dfb_id, input_dfb_id, 0, 1, 0);
                    dst_seeded = true;
                    consumed_pages = 2;
                } else {
                    input_dfb.wait_front(1);
                    if (dst_seeded) {
                        add_reuse_dest_init<ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCB>(input_dfb_id);
                        add_reuse_dest_tiles<ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCB>(input_dfb_id, 0, 0);
                    } else {
                        copy_tile_init(input_dfb_id);
                        copy_tile(input_dfb_id, 0, 0);
                        dst_seeded = true;
                    }
                }
                remaining -= consumed_pages;
                if (remaining > 0u || has_partial) {
                    input_dfb.pop_front(consumed_pages);
                } else {
                    deferred_stream_pop = consumed_pages;
                }
            }
            if (has_partial) {
                input_dfb.wait_front(1);
                if (dst_seeded) {
                    fold_partial_last(0);
                } else {
                    seed_partial_last(0);
                }
                deferred_stream_pop = 1;
            }
        } else {
            uint32_t start;
            if constexpr (is_row) {
                start = output_idx * row_pitch;
            } else if constexpr (is_col) {
                start = (output_idx / Wt) * (Ht * row_pitch) + output_idx % Wt;
            } else {
                start = output_idx * (Ht * row_pitch);
            }

            if constexpr (has_accum) {
                if (accumulate.is_first()) {
                    uint32_t tile = 0;
                    if (full_cnt & 1u) {
                        copy_tile_init(input_dfb_id);
                        copy_tile(input_dfb_id, start, 0);
                        tile = 1;
                    }
                    add_init(input_dfb_id, input_dfb_id, true);
                    for (; tile < full_cnt; tile += 2) {
                        add_tiles(input_dfb_id, input_dfb_id, start + tile * stride, start + (tile + 1) * stride, 0);
                    }
                    if (has_partial) {
                        fold_partial_last(start + full_cnt * stride);
                    }
                } else {
                    const uint32_t acc_cb = accumulate.config.cb_accumulator;
                    accum_dfb.wait_front(1);
                    deferred_accum_pop = true;
                    if (accumulate.reload == AccumulateReloadMode::FoldViaAdd) {
                        // Fold the running sum as an add operand. Odd chunks place it in the final add;
                        // even chunks reload it as the seed. This mode requires a Default accumulator CB.
                        if (full_cnt & 1u) {
                            if (full_cnt == 1u) {
                                reconfig_data_format_srcb(input_dfb_id, acc_cb);
                                add_init(input_dfb_id, acc_cb, true);
                                add_tiles(input_dfb_id, acc_cb, start, 0, 0);
                                reconfig_data_format_srcb(acc_cb, input_dfb_id);
                            } else {
                                add_init(input_dfb_id, input_dfb_id, true);
                                add_tiles(input_dfb_id, input_dfb_id, start, start + stride, 0);
                                add_init(input_dfb_id, input_dfb_id, true);
                                for (uint32_t tile = 2; tile + 1 < full_cnt; tile += 2) {
                                    add_tiles(
                                        input_dfb_id,
                                        input_dfb_id,
                                        start + tile * stride,
                                        start + (tile + 1) * stride,
                                        0);
                                }
                                reconfig_data_format_srcb(input_dfb_id, acc_cb);
                                add_init(input_dfb_id, acc_cb, true);
                                add_tiles(input_dfb_id, acc_cb, start + (full_cnt - 1u) * stride, 0, 0);
                                reconfig_data_format_srcb(acc_cb, input_dfb_id);
                            }
                        } else {
                            reconfig_data_format_srca(input_dfb_id, acc_cb);
                            copy_tile_init(acc_cb);
                            copy_tile(acc_cb, 0, 0);
                            reconfig_data_format_srca(acc_cb, input_dfb_id);
                            add_init(input_dfb_id, input_dfb_id, true);
                            for (uint32_t tile = 0; tile < full_cnt; tile += 2) {
                                add_tiles(
                                    input_dfb_id, input_dfb_id, start + tile * stride, start + (tile + 1) * stride, 0);
                            }
                        }
                    } else if (accumulate.reload == AccumulateReloadMode::CopySeedSfpuAdd) {
                        // Sum the new chunk without DEST reuse, reload the accumulator into DST[1], then
                        // combine the two fp32 DST tiles in the SFPU.
                        uint32_t tile = 0;
                        if (full_cnt & 1u) {
                            copy_tile_init(input_dfb_id);
                            copy_tile(input_dfb_id, start, 0);
                            tile = 1;
                        }
                        add_init(input_dfb_id, input_dfb_id, true);
                        for (; tile < full_cnt; tile += 2) {
                            add_tiles(
                                input_dfb_id, input_dfb_id, start + tile * stride, start + (tile + 1) * stride, 0);
                        }
                        reconfig_data_format_srca(input_dfb_id, acc_cb);
                        copy_tile_init(acc_cb);
                        copy_tile(acc_cb, 0, 1);
                        reconfig_data_format_srca(acc_cb, input_dfb_id);
#ifndef ARCH_QUASAR
                        add_binary_tile_init();
                        add_binary_tile(0, 1, 0);
                        sfpu_reduce_init<PoolType::SUM, dst_fmt>();
#else
                        ASSERT(false);
#endif
                    } else {
                        // CopySeed modes are safe for both Default and UnpackToDestFp32 accumulator CBs.
                        reconfig_data_format_srca(input_dfb_id, acc_cb);
                        copy_tile_init(acc_cb);
                        copy_tile(acc_cb, 0, 0);
                        reconfig_data_format_srca(acc_cb, input_dfb_id);
                        if (accumulate.reload == AccumulateReloadMode::CopySeedUniform) {
                            add_reuse_dest_init<ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCB>(input_dfb_id);
                            for (uint32_t tile = 0; tile < full_cnt; ++tile) {
                                add_reuse_dest_tiles<ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                                    input_dfb_id, start + tile * stride, 0);
                            }
                        } else if (accumulate.reload == AccumulateReloadMode::CopySeedZeroPair) {
                            uint32_t tile = 0;
                            if (full_cnt & 1u) {
                                add_init(input_dfb_id, scaler_dfb_id, true);
                                add_tiles(input_dfb_id, scaler_dfb_id, start, 0, 0);
                                tile = 1;
                            }
                            add_init(input_dfb_id, input_dfb_id, true);
                            for (; tile < full_cnt; tile += 2) {
                                add_tiles(
                                    input_dfb_id, input_dfb_id, start + tile * stride, start + (tile + 1) * stride, 0);
                            }
                        } else {
                            uint32_t tile = 0;
                            if (full_cnt & 1u) {
                                add_reuse_dest_init<ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCB>(input_dfb_id);
                                add_reuse_dest_tiles<ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                                    input_dfb_id, start, 0);
                                tile = 1;
                            }
                            add_init(input_dfb_id, input_dfb_id, true);
                            for (; tile < full_cnt; tile += 2) {
                                add_tiles(
                                    input_dfb_id, input_dfb_id, start + tile * stride, start + (tile + 1) * stride, 0);
                            }
                        }
                    }
                    if (has_partial) {
                        fold_partial_last(start + full_cnt * stride);
                    }
                }
            } else {
                uint32_t tile = 0;
                if (full_cnt & 1u) {
                    copy_tile_init(input_dfb_id);
                    copy_tile(input_dfb_id, start, 0);
                    tile = 1;
                }
                add_init(input_dfb_id, input_dfb_id, true);
                for (; tile < full_cnt; tile += 2) {
                    add_tiles(input_dfb_id, input_dfb_id, start + tile * stride, start + (tile + 1) * stride, 0);
                }
                if (has_partial) {
                    fold_partial_last(start + full_cnt * stride);
                }
            }
        }

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
                const uint32_t n =
                    (is_row || is_col) ? full_cnt * 32u + partial_scaler.valid_reduce_dim_elements : cnt * 1024u;
                const float inverse = 1.0f / static_cast<float>(n);
                uint32_t inverse_bits = 0;
                __builtin_memcpy(&inverse_bits, &inverse, sizeof(inverse_bits));
                mul_unary_tile(0, inverse_bits);
            }
            post_reduce_op(0);
        }

        tile_regs_commit();
        if (deferred_stream_pop > 0u) {
            input_dfb.pop_front(deferred_stream_pop);
        }
        if (deferred_accum_pop) {
            accum_dfb.pop_front(1);
        }
        tile_regs_wait();
        if constexpr (should_pop_p) {
            output_dfb.reserve_back(1);
            pack_tile(0, output_dfb_id);
            output_dfb.push_back(1);
        } else {
            if (cycles_accumulator) {
                output_dfb.reserve_back(1);
                pack_tile(0, output_dfb_id);
                output_dfb.push_back(1);
            } else {
                pack_tile(0, output_dfb_id, output_idx);
            }
        }
        tile_regs_release();
    }
    if constexpr (!should_pop_p) {
        if (!cycles_accumulator) {
            output_dfb.push_back(n_out);
        }
    }
    if constexpr (helper_pops_block) {
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
        UNPACK(ASSERT(get_dfb_num_pages(input_dfb_id) >= 1));
    } else if constexpr (waits_bulk(input_policy)) {
        UNPACK(ASSERT(get_dfb_num_pages(input_dfb_id) >= tiles_per_bulk));
        UNPACK(ASSERT(get_dfb_num_pages(input_dfb_id) % tiles_per_bulk == 0));
    } else {  // waits_upfront or no_wait
        UNPACK(ASSERT(get_dfb_num_pages(input_dfb_id) >= total_tiles));
    }
}

template <ReduceInputPolicy input_policy>
ALWI void assert_output_dfb_size(uint32_t output_dfb_id, uint32_t total_outputs) {
    if constexpr (should_pop(input_policy)) {
        // Per-tile reserve/push: only needs 1 page
        PACK(ASSERT(get_dfb_num_pages(output_dfb_id) >= 1));
    } else {
        // Bulk reserve upfront: needs all outputs
        PACK(ASSERT(get_dfb_num_pages(output_dfb_id) >= total_outputs));
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
    ReduceFp32Mode fp32_mode,
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
        reduce_type != PoolType::MIN || is_sfpu_reduce_path<reduce_type, reduce_dim, reduce_format, fp32_mode>(),
        "MIN is only valid on the Int32 SFPU reduce path; the FPU path implements MIN as -MAX(-x)");
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

    constexpr bool explicitly_accumulate_via_add = algorithm == ReduceAlgorithm::AccumulateViaAdd;
    constexpr bool input_policy_supports_accumulate_via_add =
        input_policy != ReduceInputPolicy::WaitAndPopPerTile || reduce_dim != ReduceDim::REDUCE_COL;
    constexpr bool auto_can_accumulate_via_add =
        algorithm == ReduceAlgorithm::Auto && reduce_type == PoolType::SUM &&
        input_policy_supports_accumulate_via_add &&
        (reconfig_mode == ReduceDataFormatReconfigMode::INPUT ||
         reconfig_mode == ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT) &&
        fp32_mode == ReduceFp32Mode::Fast && reduce_format != DataFormat::Int32;

    if constexpr (explicitly_accumulate_via_add) {
        static_assert(
            reduce_type == PoolType::SUM || reduce_type == PoolType::AVG,
            "AccumulateViaAdd supports SUM and standalone AVG only; MAX/MIN must use ReduceTile");
        static_assert(
            reduce_format != DataFormat::Int32,
            "AccumulateViaAdd is a floating-point add/SFPU datapath; Int32 must use ReduceTile");
        static_assert(
            fp32_mode != ReduceFp32Mode::Accurate,
            "AccumulateViaAdd does not support ReduceFp32Mode::Accurate; use ReduceTile for full-fp32 reduction");
        static_assert(
            !is_accumulate_v<AccumulateT> || reduce_type == PoolType::SUM,
            "AccumulateViaAdd cross-call Accumulate supports SUM only; use reduce_mean with a whole-reduction "
            "divisor for a cross-call mean");
        static_assert(
            input_policy_supports_accumulate_via_add,
            "AccumulateViaAdd REDUCE_COL cannot use the contiguous WaitAndPopPerTile stream");

        // The current branch's ReduceTile partial descriptor is kept intact.  AccumulateViaAdd uses the
        // orthogonal partial_mask() form because it must mask before adding tiles rather than scale after.
        ASSERT(!partial_scaler.uses_partial());
        ASSERT(partial_scaler.valid_reduce_dim_elements < 32);
        if constexpr (reduce_dim == ReduceDim::REDUCE_SCALAR) {
            ASSERT(partial_scaler.valid_reduce_dim_elements == 0);
        }
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

    if constexpr (auto_can_accumulate_via_add) {
        const bool contiguous =
            input_memory_layout.row_stride == 0 || input_memory_layout.row_stride == input_block_shape.cols;
        const bool tile_aligned = !partial_scaler.uses_partial() && partial_scaler.valid_reduce_dim_elements == 0;
        if (contiguous && tile_aligned) {
            DeviceZoneScopedN("AUTO_ACCUMULATE_VIA_ADD");
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
    }

    // =============================================================================
    // Runtime Assertions (parameter validation)
    // =============================================================================
    ASSERT(input_dfb_id != output_dfb_id);
    ASSERT(input_dfb_id != scaler_dfb_id);
    ASSERT(output_dfb_id != scaler_dfb_id);
#ifndef ARCH_QUASAR
    // is_valid_dfb_tile_page_size() is a debug validator only defined on WH/BH
    UNPACK(ASSERT(is_valid_dfb_tile_page_size(input_dfb_id, (DataFormat)unpack_src_format[input_dfb_id])));
    UNPACK(ASSERT(is_valid_dfb_tile_page_size(scaler_dfb_id, (DataFormat)unpack_src_format[scaler_dfb_id])));
    PACK(ASSERT(is_valid_dfb_tile_page_size(output_dfb_id, (DataFormat)pack_dst_format[output_dfb_id])));
#endif
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
    if constexpr (reduce_dim == ReduceDim::REDUCE_SCALAR) {
        ASSERT(!partial_scaler.uses_partial());
    }
    if constexpr (is_sfpu) {
        ASSERT(!partial_scaler.uses_partial());
    }
    if (partial_scaler.mode == ReducePartialScalerMode::OnlyTile) {
        if constexpr (reduce_dim == ReduceDim::REDUCE_ROW) {
            ASSERT(Wt == 1);
        } else if constexpr (reduce_dim == ReduceDim::REDUCE_COL) {
            ASSERT(Ht == 1);
        }
    }
    scaler_dfb.wait_front(partial_scaler.scaler_tile_count());
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
                        const uint32_t scaler_idx =
                            (wt == Wt - 1 && partial_scaler.uses_partial()) ? partial_scaler.partial_scaler_idx() : 0;
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
#ifndef ARCH_QUASAR
                    sfpu_reduce_init<reduce_type, reduce_format>();
                    sfpu_reduce<reduce_type, reduce_format, reduce_dim>(dst_idx, /*ct_dim=*/1, /*rt_dim=*/1);
#else
                    // The SFPU reduce path (Int32, or accurate-fp32 SUM) is unported on Quasar:
                    // sfpu_reduce/_init are ARCH_QUASAR-guarded out. is_sfpu_reduce_path() is false for the
                    // FPU/GMPOOL paths Quasar does support (e.g. avg_pool SUM, MAX), so this branch is dead
                    // there; static_assert makes an actual Quasar SFPU-reduce instantiation fail loudly
                    // rather than silently drop the finalize.
                    static_assert(!is_sfpu, "SFPU reduce path is not supported on Quasar");
#endif
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
                    [[maybe_unused]] const uint32_t scaler_idx =
                        (ht == Ht - 1 && partial_scaler.uses_partial()) ? partial_scaler.partial_scaler_idx() : 0;
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
#ifndef ARCH_QUASAR
                    const uint32_t sfpu_base_dst = get_dst_index(accumulate);
                    sfpu_reduce_init<reduce_type, reduce_format>();
                    for (uint32_t k = 0; k < current_chunk; ++k) {
                        sfpu_reduce<reduce_type, reduce_format, reduce_dim>(
                            sfpu_base_dst + k, /*ct_dim=*/1, /*rt_dim=*/1);
                    }
#else
                    // SFPU reduce path unported on Quasar (see the matching guard above); dead for the
                    // FPU/GMPOOL paths Quasar supports, static_assert catches a real Quasar SFPU reduce.
                    static_assert(!is_sfpu, "SFPU reduce path is not supported on Quasar");
#endif
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

template <
    ReduceDim reduce_dim,
    uint32_t input_dfb_id,
    uint32_t scaler_dfb_id,
    uint32_t output_dfb_id,
    ReduceInputPolicy input_policy,
    ReduceDataFormatReconfigMode reconfig_mode,
    ReduceFp32Mode fp32_mode,
    ReduceAlgorithm algorithm,
    typename AccumulateT>
ALWI void reduce_mean(
    ReduceInputBlockShape input_block_shape,
    uint32_t n_reduced,
    ReduceInputMemoryLayout input_memory_layout,
    AccumulateT accumulate,
    ReducePartialScaler partial_scaler) {
    ASSERT(n_reduced > 0);
    const float inverse = 1.0f / static_cast<float>(n_reduced);
    uint32_t inverse_bits = 0;
    __builtin_memcpy(&inverse_bits, &inverse, sizeof(inverse_bits));
    reduce<
        PoolType::SUM,
        reduce_dim,
        input_dfb_id,
        scaler_dfb_id,
        output_dfb_id,
        input_policy,
        reconfig_mode,
        fp32_mode,
        algorithm>(
        input_block_shape,
        input_memory_layout,
        accumulate,
        [inverse_bits](uint32_t dst) { mul_unary_tile(dst, inverse_bits); },
        partial_scaler);
}

template <
    ReduceDim reduce_dim,
    uint32_t input_dfb_id,
    uint32_t scaler_dfb_id,
    uint32_t output_dfb_id,
    ReduceInputPolicy input_policy,
    ReduceDataFormatReconfigMode reconfig_mode,
    ReduceFp32Mode fp32_mode,
    ReduceAlgorithm algorithm>
ALWI void reduce_mean(
    ReduceInputBlockShape input_block_shape,
    uint32_t n_reduced,
    ReduceInputMemoryLayout input_memory_layout,
    ReducePartialScaler partial_scaler) {
    reduce_mean<
        reduce_dim,
        input_dfb_id,
        scaler_dfb_id,
        output_dfb_id,
        input_policy,
        reconfig_mode,
        fp32_mode,
        algorithm,
        NoAccumulation>(input_block_shape, n_reduced, input_memory_layout, NoAccumulation{}, partial_scaler);
}

}  // namespace compute_kernel_lib
