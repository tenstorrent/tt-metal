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

template <ReduceDim dim, uint32_t output_dfb_id>
ALWI void configure_reduced_output_mask() {
    PACK((llk_pack_reduce_mask_config<dim, PackMode::Default>(output_dfb_id)));
}

// SFPU MAX fold
template <DataFormat format>
ALWI void sfpu_reduce_max_fold_init() {
    if constexpr (format == DataFormat::Int32) {
        binary_max_int32_tile_init();
    } else {
        binary_max_tile_init();
    }
}

template <DataFormat format>
ALWI void sfpu_reduce_max_fold_tile(uint32_t a, uint32_t b, uint32_t out) {
    if constexpr (format == DataFormat::Int32) {
        binary_max_int32_tile(a, b, out);
    } else {
        binary_max_tile(a, b, out);
    }
}

// SFPU MIN fold
template <DataFormat format>
ALWI void sfpu_reduce_min_fold_init() {
    if constexpr (format == DataFormat::Int32) {
        binary_min_int32_tile_init();
    } else {
        binary_min_tile_init();
    }
}

template <DataFormat format>
ALWI void sfpu_reduce_min_fold_tile(uint32_t a, uint32_t b, uint32_t out) {
    if constexpr (format == DataFormat::Int32) {
        binary_min_int32_tile(a, b, out);
    } else {
        binary_min_tile(a, b, out);
    }
}

// SFPU cross-tile add: add_int_tile for Int32, add_binary_tile for Float32 (not available on Quasar).
template <DataFormat format>
ALWI void sfpu_reduce_sum_fold_init() {
    if constexpr (format == DataFormat::Int32) {
        add_int_tile_init();
    } else {
#ifndef ARCH_QUASAR
        add_binary_tile_init();
#else
        static_assert(format == DataFormat::Int32, "Accurate fp32 SFPU reduce is not supported on Quasar");
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
        static_assert(format == DataFormat::Int32, "Accurate fp32 SFPU reduce is not supported on Quasar");
#endif
    }
}

// Cross-tile fold init for the SFPU paths: MAX -> binary_max, MIN -> binary_min, SUM -> add.
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

// Copy one input tile into DST and fold it into the accumulator; the first tile seeds dst_idx directly.
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

// mul_unary_tile is fp32-only, so Int32 is typecast around it (truncating toward zero).
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

// True for a 32-bit CB tagged UnpackToDestFp32, which cannot be a SrcA/SrcB operand. UNPACK/MATH only.
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

// ReduceAlgorithm::TileSum: add each output's reduce-dim tiles pairwise into one DST slot, then collapse within the
// tile, apply AVG and post_reduce_op. With Accumulate the accumulator holds the raw sum; only the last call
// finalizes, earlier calls pack the raw sum.
template <
    PoolType reduce_type,
    ReduceDim reduce_dim,
    uint32_t input_dfb_id,
    uint32_t scaler_dfb_id,
    uint32_t output_dfb_id,
    ReduceInputPolicy input_policy,
    ReduceDataFormatReconfigMode reconfig_mode,
    typename AccumulateT,
    typename PostReduceOp,
    ReduceWithinTile within_tile = ReduceWithinTile::Collapse,
    uint32_t scale_bits = REDUCE_UNIT_SCALE_BITS>
ALWI void reduce_tile_sum(
    ReduceInputBlockShape shape,
    ReduceInputMemoryLayout input_memory_layout,
    ReducePartialMode partial_mode,
    AccumulateT accumulate,
    PostReduceOp post_reduce_op,
    uint32_t output_group,
    uint32_t auxiliary_tile_offset) {
    const uint32_t Ht = shape.rows, Wt = shape.cols, NC = shape.batches;
    // Tile distance between resident rows; tiles in [Wt, row_pitch) are padding and never read.
    const uint32_t row_pitch = (input_memory_layout.row_stride > 0u) ? input_memory_layout.row_stride : Wt;
    const uint32_t batch_pitch =
        input_memory_layout.batch_stride > 0 ? input_memory_layout.batch_stride : Ht * row_pitch;
    const uint32_t in_tiles = batch_pitch * NC;

    // DST accumulation format drives the SFPU finalize (fp32 DST when fp32_dest_acc_en is on).
    constexpr DataFormat dst_fmt = DST_ACCUM_MODE ? DataFormat::Float32 : DataFormat::Float16_b;

    constexpr bool is_row = (reduce_dim == ReduceDim::REDUCE_ROW);
    constexpr bool is_col = (reduce_dim == ReduceDim::REDUCE_COL);
    constexpr auto MASK_BCAST = is_col ? ckernel::BroadcastType::COL : ckernel::BroadcastType::ROW;
    // Streaming consumes one output at a time; bulk COL input groups columns.
    constexpr bool grouped_col = is_col && input_policy == ReduceInputPolicy::BulkWaitBulkPop;
    constexpr bool streaming = input_policy == ReduceInputPolicy::WaitAndPopPerTile;
    constexpr bool streamed_input = streaming || grouped_col;
    constexpr bool has_accum = is_accumulate_v<AccumulateT>;  // cross-call CB accumulator (raw partial sum)

    // WaitUpfront waits for the resident block once; NoWait trusts the caller to have it resident.
    // Bulk ROW/SCALAR waits per output, and streaming/grouped COL synchronizes inside its loop.
    constexpr bool no_wait_p = (input_policy == ReduceInputPolicy::NoWaitNoPop);
    constexpr bool bulk_per_output = input_policy == ReduceInputPolicy::BulkWaitBulkPop && !is_col;
    constexpr bool helper_waits_block = (!streamed_input && !no_wait_p && !bulk_per_output);

    // tiles that collapse into one output, and their stride in the row-major (batch-major) input block.
    const uint32_t cnt = is_row ? Wt : (is_col ? Ht : (Ht * Wt));
    const uint32_t stride = is_col ? row_pitch : 1u;  // COL steps down a column by the row pitch
    const uint32_t n_out = is_row ? (Ht * NC) : (is_col ? (Wt * NC) : NC);

    // The caller's compile-time scale covers the whole reduction, never just this call's tiles.
    ASSERT(partial_mode == ReducePartialMode::None || partial_mode == ReducePartialMode::Mask);
    const bool has_partial = partial_mode == ReducePartialMode::Mask;
    // Each additive auxiliary slice contains an optional mask, then a required zero.
    const uint32_t mask_idx = auxiliary_tile_offset;
    const uint32_t full_cnt = has_partial ? (cnt - 1u) : cnt;  // tiles summed via pure add_tiles

    DataflowBuffer input_dfb(input_dfb_id), scaler_dfb(scaler_dfb_id), output_dfb(output_dfb_id);
    DataflowBuffer accum_dfb([&]() -> uint32_t {
        if constexpr (has_accum) {
            return accumulate.config.cb_accumulator;
        } else {
            return 0;
        }
    }());

    // With Accumulate, reload on every call but the first and finalize only on the last.
    bool do_finalize = true;
    if constexpr (has_accum) {
        do_finalize = accumulate.is_last();
    }

    // Per-call setup is format reconfig and the SFPU macro load only; hw_configure runs once at kernel start.
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
#ifdef ARCH_QUASAR
    // Quasar binds the output descriptor at op init, even when buffer formats match.
    ckernel::pack_init(output_dfb_id);
#endif
    // Load the SFPU reduce macro only when collapsing; an SFPU post_reduce_op still runs its own init.
    if constexpr (within_tile == ReduceWithinTile::Collapse && std::is_same_v<PostReduceOp, NoOp>) {
        if (do_finalize) {
            sfpu_reduce_init<PoolType::SUM, dst_fmt>();
        }
    }
    // reduce() asserts the compile-time restrictions; wait_front/reserve_back assert capacity.
    ASSERT(input_dfb_id != output_dfb_id && Ht > 0 && Wt > 0 && NC > 0);
    if constexpr (input_policy == ReduceInputPolicy::WaitAndPopPerTile) {
        UNPACK(ASSERT(get_dfb_num_pages(input_dfb_id) >= 2));
        if constexpr (is_col) {
            ASSERT(output_group <= 1);
        }
    }
#ifndef ARCH_QUASAR  // is_valid_dfb_tile_page_size is WH/BH only
    UNPACK(ASSERT(is_valid_dfb_tile_page_size(input_dfb_id, (DataFormat)unpack_src_format[input_dfb_id])));
    PACK(ASSERT(is_valid_dfb_tile_page_size(output_dfb_id, (DataFormat)pack_dst_format[output_dfb_id])));
#endif
    if constexpr (no_wait_p) {  // no wait/reserve to self-assert capacity: caller must have the block resident
        UNPACK(ASSERT(get_dfb_num_pages(input_dfb_id) >= in_tiles));
    }

    // The auxiliary CB is never popped here. The layout is [mask, zero] for a
    // partial axis and [zero] otherwise.
    // TODO: the zero is unused when every input tile has a pair partner (e.g. resident input, even count).
    const uint32_t zero_idx = auxiliary_tile_offset + (has_partial ? 1u : 0u);
    const uint32_t required_aux_tiles = zero_idx + 1u;
    if constexpr (has_accum) {
        // Indexed input uses the selected reload mode; streamed and grouped input always copy-seed.
        [[maybe_unused]] const uint32_t acc_cb = accumulate.config.cb_accumulator;
        ASSERT(input_dfb_id != acc_cb);
        if constexpr (!streamed_input) {
            // FoldViaAdd reads acc_cb via SrcB, which an UnpackToDestFp32 CB does not allow.
            UNPACK(ASSERT(accumulate.reload != AccumulateReloadMode::FoldViaAdd || !dfb_unpacks_to_dest(acc_cb)));
#ifdef ARCH_QUASAR
            ASSERT(accumulate.reload != AccumulateReloadMode::CopySeedSfpuAdd);  // needs add_binary_tile (WH/BH only)
#endif
        }
    }
    if (required_aux_tiles > 0) {
        ASSERT(input_dfb_id != scaler_dfb_id && output_dfb_id != scaler_dfb_id);
        scaler_dfb.wait_front(required_aux_tiles);
    }
    if constexpr (helper_waits_block) {
        input_dfb.wait_front(in_tiles);  // Bulk / WaitUpfront: whole resident block, indexed per output
    }

    auto add_input_with_zero = [&](uint32_t input_idx, uint32_t dst_idx = 0) {
        reconfig_data_format_srcb(input_dfb_id, scaler_dfb_id);
        add_tiles_init(input_dfb_id, scaler_dfb_id, true);
        add_tiles(input_dfb_id, scaler_dfb_id, input_idx, zero_idx, dst_idx);
        reconfig_data_format_srcb(scaler_dfb_id, input_dfb_id);
    };

    // Fold the masked last reduce-dim tile into DST. The bcast shorthands overwrite DST, so this calls the LLK
    // with acc_to_dest=1 and clear_fp32_dst_acc=false.
    [[maybe_unused]] auto fold_partial_last = [&](uint32_t last_idx, uint32_t dst_idx = 0) {
        // Compressed inputs use a float mask: switch SrcB to it and restore it afterwards.
        reconfig_data_format_srcb(input_dfb_id, scaler_dfb_id);
        MATH((llk_math_eltwise_binary_init<ckernel::EltwiseBinaryType::ELWMUL, MASK_BCAST, MATH_FIDELITY>(
            input_dfb_id, scaler_dfb_id, 1)));
        UNPACK((llk_unpack_AB_init<MASK_BCAST>(input_dfb_id, scaler_dfb_id)));
        UNPACK((llk_unpack_AB<MASK_BCAST>(input_dfb_id, scaler_dfb_id, last_idx, mask_idx)));
        MATH((llk_math_eltwise_binary<ckernel::EltwiseBinaryType::ELWMUL, MASK_BCAST, DST_ACCUM_MODE, MATH_FIDELITY>(
            dst_idx, false)));
        reconfig_data_format_srcb(scaler_dfb_id, input_dfb_id);
    };

    // Indexed by DST so grouped COL can finalize every output column.
    auto finalize_output = [&](uint32_t dst_idx) {
        if (!do_finalize) {
            return;
        }
        if constexpr (within_tile == ReduceWithinTile::Collapse) {
            // A preceding output's callback may have replaced the SFPU replay program.
            if constexpr (!std::is_same_v<PostReduceOp, NoOp>) {
                sfpu_reduce_init<PoolType::SUM, dst_fmt>();
            }
            if constexpr (is_row) {
                sfpu_reduce<PoolType::SUM, dst_fmt, ReduceDim::REDUCE_ROW>(dst_idx, 1, 1);
            } else if constexpr (is_col) {
                sfpu_reduce<PoolType::SUM, dst_fmt, ReduceDim::REDUCE_COL>(dst_idx, 1, 1);
            } else {
                sfpu_reduce<PoolType::SUM, dst_fmt, ReduceDim::REDUCE_ROW>(dst_idx, 1, 1);
                sfpu_reduce<PoolType::SUM, dst_fmt, ReduceDim::REDUCE_COL>(dst_idx, 1, 1);
            }
        }
        if constexpr (scale_bits != REDUCE_UNIT_SCALE_BITS) {
            if constexpr (within_tile == ReduceWithinTile::Skip) {
                binop_with_scalar_tile_init();
            }
            mul_unary_tile(dst_idx, scale_bits);
        }
        post_reduce_op(dst_idx);
    };

    // Finalized outputs have the same packer-zero contract as ReduceAlgorithm::Native.
    // Intermediate cross-call sums and Skip outputs retain all their elements.
    auto configure_output_mask = [&]() {
        if constexpr (within_tile == ReduceWithinTile::Collapse) {
            if (do_finalize) {
                configure_reduced_output_mask<reduce_dim, output_dfb_id>();
            }
        }
    };
    auto clear_output_mask = [&]() {
        if constexpr (within_tile == ReduceWithinTile::Collapse) {
            if (do_finalize) {
                PACK((llk_pack_reduce_mask_clear()));
            }
        }
    };

    configure_output_mask();
    if constexpr (grouped_col) {
        // A bulk contains all rows of an output-column group.
        const uint32_t default_output_group = Wt < DEST_AUTO_LIMIT ? Wt : DEST_AUTO_LIMIT;
        const uint32_t group = output_group > 0 ? output_group : default_output_group;
        ASSERT(group > 0 && group <= DEST_AUTO_LIMIT);

        for (uint32_t nc = 0; nc < NC; ++nc) {
            for (uint32_t wt = 0; wt < Wt; wt += group) {
                const uint32_t current_outputs = group < Wt - wt ? group : Wt - wt;
                tile_regs_acquire();

                if constexpr (has_accum) {
                    if (!accumulate.is_first()) {
                        const uint32_t acc_cb = accumulate.config.cb_accumulator;
                        accum_dfb.wait_front(current_outputs);
                        reconfig_data_format_srca(input_dfb_id, acc_cb);
                        copy_tile_init(acc_cb);
                        for (uint32_t out = 0; out < current_outputs; ++out) {
                            copy_tile(acc_cb, out, out);
                        }
                        accum_dfb.pop_front(current_outputs);
                        reconfig_data_format_srca(acc_cb, input_dfb_id);
                    }
                }

                const uint32_t input_tiles = Ht * current_outputs;
                input_dfb.wait_front(input_tiles);
                uint32_t h = 0;
                if (full_cnt & 1u) {
                    for (uint32_t out = 0; out < current_outputs; ++out) {
                        add_input_with_zero(out, out);
                    }
                    h = 1;
                }
                if (h < full_cnt) {
                    add_tiles_init(input_dfb_id, input_dfb_id, true);
                    for (; h < full_cnt; h += 2) {
                        for (uint32_t out = 0; out < current_outputs; ++out) {
                            add_tiles(
                                input_dfb_id,
                                input_dfb_id,
                                h * current_outputs + out,
                                (h + 1) * current_outputs + out,
                                out);
                        }
                    }
                }
                if (has_partial) {
                    for (uint32_t out = 0; out < current_outputs; ++out) {
                        fold_partial_last(full_cnt * current_outputs + out, out);
                    }
                }
                input_dfb.pop_front(input_tiles);

                for (uint32_t out = 0; out < current_outputs; ++out) {
                    finalize_output(out);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t out = 0; out < current_outputs; ++out) {
                    output_dfb.reserve_back(1);
                    pack_tile(out, output_dfb_id);
                    output_dfb.push_back(1);
                }
                tile_regs_release();
            }
        }
        clear_output_mask();
        return;
    }

    for (uint32_t o = 0; o < n_out; ++o) {
        if constexpr (bulk_per_output) {
            input_dfb.wait_front(cnt);
        }
        tile_regs_acquire();

        if constexpr (streaming) {
            // Stream the tiles in pairs from the CB front, adding a single tile wherever a pair would cross the wrap.
            bool loaded_accumulator = false;
            if constexpr (has_accum) {
                if (!accumulate.is_first()) {
                    const uint32_t acc_cb = accumulate.config.cb_accumulator;
                    accum_dfb.wait_front(1);
                    reconfig_data_format_srca(input_dfb_id, acc_cb);
                    copy_tile_init(acc_cb);
                    copy_tile(acc_cb, 0, 0);
                    reconfig_data_format_srca(acc_cb, input_dfb_id);
                    loaded_accumulator = true;
                }
            }

            // Only UNPACK owns the read pointer. It shares the pages until the wrap and the CB
            // size, so every thread derives the same sequence of zero and pair adds.
            uint32_t wrap_state = 0;
            UNPACK({
                const auto& cb = get_local_cb_interface(input_dfb_id);
                const uint32_t until_wrap = (cb.fifo_limit - cb.fifo_rd_ptr) / cb.fifo_page_size;
                wrap_state = (until_wrap << 16) | get_dfb_num_pages(input_dfb_id);
                mailbox_write(ckernel::ThreadId::MathThreadId, wrap_state);
                mailbox_write(ckernel::ThreadId::PackThreadId, wrap_state);
            });
            MATH(wrap_state = mailbox_read(ckernel::ThreadId::UnpackThreadId));
            PACK(wrap_state = mailbox_read(ckernel::ThreadId::UnpackThreadId));
            const uint32_t num_pages = wrap_state & 0xFFFFu;
            uint32_t until_wrap = wrap_state >> 16;

            bool pair_init = false;
            for (uint32_t consumed = 0; consumed < full_cnt;) {
                // A pair must not straddle the wrap: tile 1 is read at rd_ptr + page_size, unwrapped.
                const uint32_t step = (until_wrap == 1u || consumed + 1u == full_cnt) ? 1u : 2u;
                input_dfb.wait_front(step);
                if (step == 1u) {
                    add_input_with_zero(0);
                    pair_init = false;
                } else {
                    if (!pair_init) {
                        add_tiles_init(input_dfb_id, input_dfb_id, true);
                        pair_init = true;
                    }
                    add_tiles(input_dfb_id, input_dfb_id, 0, 1, 0);
                }
                input_dfb.pop_front(step);
                consumed += step;
                until_wrap = until_wrap == step ? num_pages : until_wrap - step;
            }
            if constexpr (has_accum) {
                if (loaded_accumulator) {
                    accum_dfb.pop_front(1);
                }
            }
            if (has_partial) {  // ROW partial: the LAST reduce-dim tile is now at the CB front; fold it masked
                input_dfb.wait_front(1);
                fold_partial_last(0);
                input_dfb.pop_front(1);
            }
        } else {
            // `start` is output o's first reduce-dim tile in the resident block.
            uint32_t start;
            if constexpr (bulk_per_output) {
                start = 0;
            } else if constexpr (is_row) {
                start = (o / Ht) * batch_pitch + (o % Ht) * row_pitch;
            } else if constexpr (is_col) {
                start = (o / Wt) * batch_pitch + (o % Wt);
            } else {
                start = o * batch_pitch;  // scalar: row_pitch == Wt (contiguous, asserted in reduce())
            }

            if constexpr (has_accum) {
                if (accumulate.is_first()) {
                    // A fresh DST reads 0, so the first acc_to_dest add starts the sum.
                    uint32_t k = 0;
                    if (full_cnt & 1u) {
                        add_input_with_zero(start);
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
                    // Fold the accumulator with this call's tiles as accumulate.reload selects.
                    const uint32_t acc_cb = accumulate.config.cb_accumulator;
                    accum_dfb.wait_front(1);
                    if (accumulate.reload == AccumulateReloadMode::FoldViaAdd) {
                        // add_tiles_init does not reconfigure formats, so switch SrcB around the accumulator add.
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
                    } else if (accumulate.reload == AccumulateReloadMode::CopySeedSfpuAdd && full_cnt > 0) {
                        // Sum the new tiles in DST[0], copy_tile the accumulator into DST[1], add them on the SFPU.
                        {
                            uint32_t k = 0;
                            if (full_cnt & 1u) {
                                add_input_with_zero(start);
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
                        add_binary_tile(0, 1, 0);  // DST[0] = DST[0] + DST[1] (fp32 SFPU add)
                        if constexpr (within_tile == ReduceWithinTile::Collapse) {
                            if (do_finalize && std::is_same_v<PostReduceOp, NoOp>) {
                                sfpu_reduce_init<PoolType::SUM, dst_fmt>();
                            }
                        }
#else
                        ASSERT(false);  // CopySeedSfpuAdd needs add_binary_tile (WH/BH only)
#endif
                    } else {
                        // CopySeedSfpuAdd also lands here when only a partial tile remains.
                        reconfig_data_format_srca(input_dfb_id, acc_cb);
                        copy_tile_init(acc_cb);
                        copy_tile(acc_cb, 0, 0);  // DST = accumulator
                        reconfig_data_format_srca(acc_cb, input_dfb_id);
                        uint32_t k = 0;
                        if (full_cnt & 1u) {
                            add_input_with_zero(start);
                            k = 1;
                        }
                        add_tiles_init(input_dfb_id, input_dfb_id, true);
                        for (; k < full_cnt; k += 2) {
                            add_tiles(input_dfb_id, input_dfb_id, start + k * stride, start + (k + 1) * stride, 0);
                        }
                    }
                    accum_dfb.pop_front(1);
                    if (has_partial) {  // ROW/COL partial: fold the masked last tile in after the accumulator
                        fold_partial_last(start + full_cnt * stride);
                    }
                }
            } else {
                // A fresh DST reads 0, so the first acc_to_dest add starts the sum.
                uint32_t k = 0;
                if (full_cnt & 1u) {
                    add_input_with_zero(start);
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

        finalize_output(0);

        tile_regs_commit();
        tile_regs_wait();
        // Reserve per tile so an in-place accumulator can replace each tile after consuming it.
        output_dfb.reserve_back(1);
        pack_tile(0, output_dfb_id);
        output_dfb.push_back(1);
        tile_regs_release();
        if constexpr (bulk_per_output) {
            input_dfb.pop_front(cnt);
        }
    }
    clear_output_mask();
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
constexpr bool manages_cb(ReduceInputPolicy p) { return p != ReduceInputPolicy::NoWaitNoPop; }

// =============================================================================
// Helper Function Implementations
// =============================================================================

template <PoolType reduce_type, ReduceDim reduce_dim>
ALWI void reduce_init_short_with_dt(uint32_t old_dfb_id, uint32_t input_dfb_id, uint32_t scaler_dfb_id) {
    constexpr bool swap_operands = reduce_swaps_operands<reduce_type, reduce_dim, false>();
    const uint32_t srca_dfb_id = swap_operands ? scaler_dfb_id : input_dfb_id;

    UNPACK(
        (llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE, p_dim_stride_target::IGNORE>(old_dfb_id, srca_dfb_id)));
    MATH((llk_math_reconfig_data_format_srca<DST_ACCUM_MODE>(old_dfb_id, srca_dfb_id)));

    UNPACK((llk_unpack_AB_reduce_init<reduce_type, reduce_dim>(input_dfb_id, scaler_dfb_id)));

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

template <PoolType reduce_type, ReduceDim reduce_dim, typename AccumulateT, bool is_sfpu = false>
ALWI void reload_accumulator_if_needed(
    DataflowBuffer& accum_dfb,
    uint32_t input_dfb_id,
    uint32_t scaler_dfb_id,
    const AccumulateT& accumulate,
    uint32_t dest_chunk_size = 1) {
    if constexpr (is_accumulate_v<AccumulateT>) {
        if (!accumulate.is_first()) {
            accum_dfb.wait_front(dest_chunk_size);
            constexpr bool swap_operands = reduce_swaps_operands<reduce_type, reduce_dim, is_sfpu>();
            const uint32_t prev_srca_cb = swap_operands ? scaler_dfb_id : input_dfb_id;

            // GMPOOL keeps the MAX + REDUCE_ROW accumulator in row 0 of each face, but the packed result sits in
            // column 0. Transposing within each face on reload restores that layout; the SFPU path never uses GMPOOL.
            constexpr bool reload_within_face_transpose =
                (reduce_type == PoolType::MAX && reduce_dim == ReduceDim::REDUCE_ROW && !is_sfpu);

            reconfig_data_format_srca(prev_srca_cb, accumulate.config.cb_accumulator);
            copy_init(
                accumulate.config.cb_accumulator,
                /*transpose_of_faces=*/0,
                /*transpose_within_16x16_face=*/reload_within_face_transpose ? 1u : 0u);
            for (uint32_t tile = 0; tile < dest_chunk_size; ++tile) {
                copy_tile(accumulate.config.cb_accumulator, tile, accumulate.config.dst_index + tile);
            }
            accum_dfb.pop_front(dest_chunk_size);

            // copy_tile changed the SrcA config; restore it for the input.
            if constexpr (is_sfpu) {
                // Point SrcA back at the input for the next fold; the caller does the fold init.
                reconfig_data_format_srca(accumulate.config.cb_accumulator, input_dfb_id);
                copy_init(input_dfb_id);
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
    } else {
        ASSERT(get_dfb_num_pages(input_dfb_id) >= total_tiles);
    }
}

ALWI void assert_output_dfb_size(uint32_t output_dfb_id) {
    // Outputs are reserved and pushed one tile at a time for every input policy.
    ASSERT(get_dfb_num_pages(output_dfb_id) >= 1);
}

// =============================================================================
// Main Reduce Function Implementation
// =============================================================================

template <
    PoolType reduce_type,
    ReduceDim reduce_dim,
    uint32_t input_dfb_id,
    uint32_t auxiliary_dfb_id,
    uint32_t output_dfb_id,
    ReduceInputPolicy input_policy,
    ReduceDataFormatReconfigMode reconfig_mode,
    ReduceFp32Mode fp32_mode,
    ReduceAlgorithm algorithm,
    ReduceWithinTile within_tile,
    uint32_t scale_bits,
    typename AccumulateT,
    typename PostReduceOp>
ALWI void reduce(
    ReduceInputBlockShape input_block_shape,
    ReduceInputMemoryLayout input_memory_layout,
    AccumulateT accumulate,
    PostReduceOp post_reduce_op,
    ReducePartialMode partial_mode,
    uint32_t output_group,
    uint32_t auxiliary_tile_offset) {
    // Int32, bf16 MIN and Accurate fp32 route to the SFPU via is_sfpu_reduce_path<>(); others use FPU/GMPOOL.
    constexpr DataFormat reduce_format = static_cast<DataFormat>(unpack_src_format[input_dfb_id]);
    constexpr bool has_auxiliary = auxiliary_dfb_id != REDUCE_NO_AUXILIARY_CB;
    // Fallback ID so unused branches compile without an auxiliary CB; assertions forbid reading through it.
    constexpr uint32_t scaler_dfb_id = has_auxiliary ? auxiliary_dfb_id : input_dfb_id;
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
        "MIN requires an SFPU path: Int32, bf16, or Accurate fp32, bf16 only on Quasar, on REDUCE_ROW/COL. "
        "The FPU has no min pool at all, so every other MIN is lowered to -MAX(-x) and arrives as PoolType::MAX");
    static_assert(
        input_policy == ReduceInputPolicy::WaitAndPopPerTile || input_policy == ReduceInputPolicy::BulkWaitBulkPop ||
            input_policy == ReduceInputPolicy::WaitUpfrontNoPop || input_policy == ReduceInputPolicy::NoWaitNoPop,
        "Unsupported reduction input policy");
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
    // The MAX + REDUCE_ROW reload needs a within-face transpose, which Quasar does not support.
    static_assert(
        !is_accumulate_v<AccumulateT> || !(reduce_type == PoolType::MAX && reduce_dim == ReduceDim::REDUCE_ROW),
        "Accumulate with PoolType::MAX + REDUCE_ROW is not supported on Quasar: the accumulator "
        "reload requires a within-16x16-face transpose, which copy_tile_to_dst_init_short asserts "
        "against on Quasar.");
#endif

    // =============================================================================
    // Algorithm selection. ReduceAlgorithm::TileSum is a restricted, faster datapath for wide float SUM/AVG reduces;
    // anything it cannot express is rejected here (compile-time where possible) and must use ReduceAlgorithm::Native.
    // =============================================================================
    constexpr bool is_sfpu = is_sfpu_reduce_path<reduce_type, reduce_dim, reduce_format, fp32_mode>();
    static_assert(
        has_auxiliary || is_sfpu,
        "Native and TileSum reductions require an auxiliary CB: Native reads its scaler and TileSum its zero tile");
    static_assert(
        scale_bits == REDUCE_UNIT_SCALE_BITS || algorithm == ReduceAlgorithm::TileSum || is_sfpu,
        "ReduceAlgorithm::Native carries its scale in the scaler tile, so scale_bits must be 1.0f");
    if constexpr (!has_auxiliary) {
        ASSERT(partial_mode == ReducePartialMode::None);
    }
    if constexpr (algorithm == ReduceAlgorithm::TileSum) {
        static_assert(
            reduce_type == PoolType::SUM || reduce_type == PoolType::AVG,
            "ReduceAlgorithm::TileSum computes SUM or AVG. AVG is the sum times the caller's scale_bits; "
            "MAX/MIN are not expressible via additive accumulate, so use ReduceAlgorithm::Native.");
        static_assert(
            reduce_format != DataFormat::Int32,
            "ReduceAlgorithm::TileSum: float only (add_tiles + sfpu_reduce). Int32 must use ReduceAlgorithm::Native.");
        // One row/column mask cannot express a 2-D partial corner, so REDUCE_SCALAR rejects partial.
        if constexpr (reduce_dim == ReduceDim::REDUCE_SCALAR) {
            ASSERT(partial_mode == ReducePartialMode::None);
        }
        // Skip inputs are already collapsed along the reduce axis, so a lane mask has nothing to mask.
        if constexpr (within_tile == ReduceWithinTile::Skip) {
            ASSERT(partial_mode == ReducePartialMode::None);
        }
        // REDUCE_SCALAR and streamed input cannot skip padded rows, so they require a contiguous layout.
        if (input_memory_layout.row_stride != 0) {
            ASSERT(input_memory_layout.row_stride >= input_block_shape.cols);
            if constexpr (reduce_dim == ReduceDim::REDUCE_SCALAR) {
                ASSERT(input_memory_layout.row_stride == input_block_shape.cols);
            }
            if constexpr (input_policy == ReduceInputPolicy::WaitAndPopPerTile) {
                ASSERT(input_memory_layout.row_stride == input_block_shape.cols);
            }
        }
        detail::reduce_tile_sum<
            reduce_type,
            reduce_dim,
            input_dfb_id,
            scaler_dfb_id,
            output_dfb_id,
            input_policy,
            reconfig_mode,
            AccumulateT,
            PostReduceOp,
            within_tile,
            scale_bits>(
            input_block_shape,
            input_memory_layout,
            partial_mode,
            accumulate,
            post_reduce_op,
            output_group,
            auxiliary_tile_offset);
        return;
    }

    if constexpr (is_sfpu && scale_bits != REDUCE_UNIT_SCALE_BITS) {
        // The SFPU fold reads no scaler, so the scale multiplies each finished output before the caller's op.
        reduce<
            reduce_type,
            reduce_dim,
            input_dfb_id,
            auxiliary_dfb_id,
            output_dfb_id,
            input_policy,
            reconfig_mode,
            fp32_mode,
            algorithm,
            within_tile,
            REDUCE_UNIT_SCALE_BITS>(
            input_block_shape,
            input_memory_layout,
            accumulate,
            [&](uint32_t dst) {
                detail::reduce_post_mul_tile<reduce_format>(dst, scale_bits);
                post_reduce_op(dst);
            },
            partial_mode,
            output_group,
            auxiliary_tile_offset);
        return;
    }

    // Code after the TileSum branch is still instantiated for TileSum calls, so the check names the algorithm.
    static_assert(
        algorithm == ReduceAlgorithm::TileSum || within_tile == ReduceWithinTile::Collapse,
        "ReduceWithinTile::Skip is ReduceAlgorithm::TileSum-only: on the ReduceAlgorithm::Native datapath the "
        "reduce_tile matmul-with-ones performs the within-tile collapse itself, so there is nothing to skip. "
        "Select ReduceAlgorithm::TileSum, or drop the Skip.");

    // =============================================================================
    // Runtime Assertions (parameter validation)
    // =============================================================================
    ASSERT(input_dfb_id != output_dfb_id);
    if constexpr (has_auxiliary) {
        ASSERT(input_dfb_id != scaler_dfb_id);
        ASSERT(output_dfb_id != scaler_dfb_id);
    }
#ifndef ARCH_QUASAR
    // is_valid_dfb_tile_page_size() is a debug validator only defined on WH/BH
    UNPACK(ASSERT(is_valid_dfb_tile_page_size(input_dfb_id, (DataFormat)unpack_src_format[input_dfb_id])));
    if constexpr (has_auxiliary) {
        UNPACK(ASSERT(is_valid_dfb_tile_page_size(scaler_dfb_id, (DataFormat)unpack_src_format[scaler_dfb_id])));
    }
    PACK(ASSERT(is_valid_dfb_tile_page_size(output_dfb_id, (DataFormat)pack_dst_format[output_dfb_id])));
#endif
    ASSERT(input_block_shape.rows > 0);
    ASSERT(input_block_shape.cols > 0);
    ASSERT(input_block_shape.batches > 0);
    if (input_memory_layout.row_stride != 0) {
        ASSERT(input_memory_layout.row_stride >= input_block_shape.cols);
    }

    constexpr bool enable_accumulation = is_accumulate_v<AccumulateT>;
    const bool run_post_op = [&]() {
        if constexpr (enable_accumulation) {
            return accumulate.is_last();
        } else {
            return true;
        }
    }();
    const uint32_t Ht = input_block_shape.rows;
    const uint32_t Wt = input_block_shape.cols;
    const uint32_t num_batches = input_block_shape.batches;

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
#ifdef ARCH_QUASAR
    // Format reconfiguration alone does not bind a different Quasar output buffer.
    ckernel::pack_init(output_dfb_id);
#endif
    if constexpr (is_sfpu) {
        // hw_startup and the reconfig above set the HW config; this only re-points the unpacker.
        copy_init(input_dfb_id);
    } else {
        reduce_init<reduce_type, reduce_dim>(input_dfb_id, scaler_dfb_id, output_dfb_id);
    }
    ASSERT(partial_mode == ReducePartialMode::None || partial_mode == ReducePartialMode::Scaler);
    if constexpr (is_sfpu) {
        // SFPU folds copy input directly and never read the scaler. Partial scalers cannot mask padding.
        ASSERT(partial_mode == ReducePartialMode::None);
    }
    // REDUCE_SCALAR can't use a partial scaler because it applies the scaler twice.
    if constexpr (reduce_dim == ReduceDim::REDUCE_SCALAR) {
        ASSERT(partial_mode == ReducePartialMode::None);
    }
    const bool has_partial_scaler = partial_mode == ReducePartialMode::Scaler;
    // This call's auxiliary slice is [full scaler, optional partial scaler].
    const uint32_t scaler_tile_count = has_partial_scaler ? 2u : 1u;
    const uint32_t full_scaler_idx = auxiliary_tile_offset;
    const uint32_t partial_scaler_idx = auxiliary_tile_offset + (has_partial_scaler ? 1u : 0u);
    if constexpr (has_auxiliary) {
        scaler_dfb.wait_front(auxiliary_tile_offset + scaler_tile_count);
    }
    if constexpr (is_sfpu) {
        // Every pack in this call targets the same reduced output layout.
        detail::configure_reduced_output_mask<reduce_dim, output_dfb_id>();
    }
    constexpr uint32_t onetile = 1;

    if constexpr (reduce_dim == ReduceDim::REDUCE_SCALAR) {
        // =================================================================
        // REDUCE_SCALAR: HW reduction - all tiles -> 1 output tile per batch
        // =================================================================
        const uint32_t stride = (input_memory_layout.row_stride > 0) ? input_memory_layout.row_stride : Wt;
        const uint32_t tiles_per_bulk = Ht * stride;
        const uint32_t batch_stride =
            input_memory_layout.batch_stride > 0 ? input_memory_layout.batch_stride : tiles_per_bulk;
        const uint32_t total_input_tiles = batch_stride * num_batches;
        UNPACK((assert_input_dfb_size<input_policy>(input_dfb_id, tiles_per_bulk, total_input_tiles)));
        PACK((assert_output_dfb_size(output_dfb_id)));

        if constexpr (waits_upfront(input_policy)) {
            input_dfb.wait_front(total_input_tiles);
        }

        uint32_t batch_offset = 0;
        for (uint32_t nc = 0; nc < num_batches; ++nc) {
            if constexpr (waits_bulk(input_policy)) {
                input_dfb.wait_front(tiles_per_bulk);
            }

            tile_regs_acquire();

            reload_accumulator_if_needed<reduce_type, reduce_dim, AccumulateT, is_sfpu>(
                accum_dfb, input_dfb_id, scaler_dfb_id, accumulate);

            const uint32_t dst_idx = get_dst_index(accumulate);
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                for (uint32_t wt = 0; wt < Wt; ++wt) {
                    if constexpr (waits_per_tile(input_policy)) {
                        input_dfb.wait_front(onetile);
                        reduce_tile<reduce_type, reduce_dim>(input_dfb_id, scaler_dfb_id, 0, full_scaler_idx, dst_idx);
                        input_dfb.pop_front(onetile);
                    } else if constexpr (waits_bulk(input_policy)) {
                        uint32_t tile_idx = ht * stride + wt;
                        reduce_tile<reduce_type, reduce_dim>(
                            input_dfb_id, scaler_dfb_id, tile_idx, full_scaler_idx, dst_idx);
                    } else {
                        uint32_t tile_idx = batch_offset + ht * stride + wt;
                        reduce_tile<reduce_type, reduce_dim>(
                            input_dfb_id, scaler_dfb_id, tile_idx, full_scaler_idx, dst_idx);
                    }
                }
            }

            if (run_post_op) {
                post_reduce_op(dst_idx);
            }

            output_dfb.reserve_back(onetile);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(get_dst_index(accumulate), output_dfb_id);
            tile_regs_release();
            output_dfb.push_back(onetile);

            if constexpr (waits_bulk(input_policy)) {
                input_dfb.pop_front(tiles_per_bulk);
            }

            if constexpr (!should_pop(input_policy)) {
                batch_offset += batch_stride;
            }
        }
    } else if constexpr (reduce_dim == ReduceDim::REDUCE_ROW) {
        // =================================================================
        // REDUCE_ROW: W reduction - each row -> 1 output tile (Ht outputs per batch)
        // =================================================================
        const uint32_t stride = (input_memory_layout.row_stride > 0) ? input_memory_layout.row_stride : Wt;
        const uint32_t batch_stride =
            input_memory_layout.batch_stride > 0 ? input_memory_layout.batch_stride : Ht * stride;
        const uint32_t total_input_tiles = batch_stride * num_batches;
        UNPACK((assert_input_dfb_size<input_policy>(input_dfb_id, Wt, total_input_tiles)));
        PACK((assert_output_dfb_size(output_dfb_id)));

        if constexpr (waits_upfront(input_policy)) {
            input_dfb.wait_front(total_input_tiles);
        }

        uint32_t index_offset = 0;
        for (uint32_t nc = 0; nc < num_batches; ++nc) {
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                if constexpr (waits_bulk(input_policy)) {
                    input_dfb.wait_front(Wt);
                }

                tile_regs_acquire();

                reload_accumulator_if_needed<reduce_type, reduce_dim, AccumulateT, is_sfpu>(
                    accum_dfb, input_dfb_id, scaler_dfb_id, accumulate);
                if constexpr (is_sfpu) {
                    // Fold needed if the axis has >1 tile, or Accumulate reloaded a result into DST.
                    if (Wt > 1 || !detail::sfpu_is_first_tile(0, accumulate)) {
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
                        const uint32_t scaler_idx = (wt == Wt - 1) ? partial_scaler_idx : full_scaler_idx;
                        if constexpr (waits_per_tile(input_policy)) {
                            input_dfb.wait_front(onetile);
                            reduce_tile<reduce_type, reduce_dim>(input_dfb_id, scaler_dfb_id, 0, scaler_idx, dst_idx);
                            input_dfb.pop_front(onetile);
                        } else if constexpr (waits_bulk(input_policy)) {
                            reduce_tile<reduce_type, reduce_dim>(input_dfb_id, scaler_dfb_id, wt, scaler_idx, dst_idx);
                        } else {
                            reduce_tile<reduce_type, reduce_dim>(
                                input_dfb_id, scaler_dfb_id, wt + index_offset, scaler_idx, dst_idx);
                        }
                    }
                }

                if constexpr (is_sfpu) {
                    sfpu_reduce_init<reduce_type, reduce_format>();
                    sfpu_reduce<reduce_type, reduce_format, reduce_dim>(dst_idx, /*ct_dim=*/1, /*rt_dim=*/1);
                }

                if (run_post_op) {
                    post_reduce_op(dst_idx);
                }

                output_dfb.reserve_back(onetile);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(dst_idx, output_dfb_id);
                tile_regs_release();
                output_dfb.push_back(onetile);

                if constexpr (waits_bulk(input_policy)) {
                    input_dfb.pop_front(Wt);
                }

                if constexpr (!should_pop(input_policy)) {
                    index_offset += stride;
                }
            }
            if constexpr (!should_pop(input_policy)) {
                index_offset = (nc + 1) * batch_stride;
            }
        }
    } else {
        // =================================================================
        // REDUCE_COL: H reduction - each column -> 1 output tile (Wt outputs per batch), chunked by DEST capacity
        // =================================================================

        // The reader must derive the same chunk size from DEST_AUTO_LIMIT.
        constexpr uint32_t default_chunk_size = is_sfpu ? (DEST_AUTO_LIMIT - 1) : DEST_AUTO_LIMIT;
        const uint32_t requested_chunk = output_group > 0 ? output_group : default_chunk_size;
        const uint32_t chunk_size = requested_chunk < default_chunk_size ? requested_chunk : default_chunk_size;
        const uint32_t stride = (input_memory_layout.row_stride > 0) ? input_memory_layout.row_stride : Wt;
        const uint32_t tiles_per_bulk = Ht * stride;
        const uint32_t batch_stride =
            input_memory_layout.batch_stride > 0 ? input_memory_layout.batch_stride : tiles_per_bulk;
        const uint32_t total_input_tiles = batch_stride * num_batches;
        UNPACK((assert_input_dfb_size<input_policy>(input_dfb_id, Ht * chunk_size, total_input_tiles)));
        PACK((assert_output_dfb_size(output_dfb_id)));

        if constexpr (waits_upfront(input_policy)) {
            input_dfb.wait_front(total_input_tiles);
        }

        uint32_t batch_offset = 0;
        for (uint32_t nc = 0; nc < num_batches; ++nc) {
            for (uint32_t wt = 0; wt < Wt; wt += chunk_size) {
                uint32_t chunk_end = (wt + chunk_size < Wt) ? (wt + chunk_size) : Wt;
                uint32_t current_chunk = chunk_end - wt;
                uint32_t tiles_in_chunk = Ht * current_chunk;

                if constexpr (waits_bulk(input_policy)) {
                    input_dfb.wait_front(tiles_in_chunk);
                }

                tile_regs_acquire();

                reload_accumulator_if_needed<reduce_type, reduce_dim, AccumulateT, is_sfpu>(
                    accum_dfb, input_dfb_id, scaler_dfb_id, accumulate, current_chunk);
                if constexpr (is_sfpu) {
                    // Fold needed if the axis has >1 tile, or Accumulate reloaded a result into DST.
                    if (Ht > 1 || !detail::sfpu_is_first_tile(0, accumulate)) {
                        detail::sfpu_reduce_fold_init<reduce_type, reduce_format>();
                    }
                }

                for (uint32_t ht = 0; ht < Ht; ++ht) {
                    uint32_t dst_idx = get_dst_index(accumulate);
                    // Last H-tile picks up the partial scaler when one was prepared by the reader.
                    [[maybe_unused]] const uint32_t scaler_idx = (ht == Ht - 1) ? partial_scaler_idx : full_scaler_idx;
                    for (uint32_t i = wt; i < chunk_end; ++i) {
                        if constexpr (is_sfpu) {
                            const bool is_first_tile = detail::sfpu_is_first_tile(ht, accumulate);
                            constexpr uint32_t sfpu_work_dst = default_chunk_size;
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
                            input_dfb.wait_front(onetile);
                            reduce_tile<reduce_type, reduce_dim>(input_dfb_id, scaler_dfb_id, 0, scaler_idx, dst_idx);
                            input_dfb.pop_front(onetile);
                        } else if constexpr (waits_bulk(input_policy)) {
                            uint32_t tile_idx = ht * current_chunk + (i - wt);
                            reduce_tile<reduce_type, reduce_dim>(
                                input_dfb_id, scaler_dfb_id, tile_idx, scaler_idx, dst_idx);
                        } else {
                            uint32_t tile_idx = batch_offset + ht * stride + i;
                            reduce_tile<reduce_type, reduce_dim>(
                                input_dfb_id, scaler_dfb_id, tile_idx, scaler_idx, dst_idx);
                        }
                        ++dst_idx;
                    }
                }

                if constexpr (is_sfpu) {
                    const uint32_t sfpu_base_dst = get_dst_index(accumulate);
                    sfpu_reduce_init<reduce_type, reduce_format>();
                    for (uint32_t k = 0; k < current_chunk; ++k) {
                        sfpu_reduce<reduce_type, reduce_format, reduce_dim>(
                            sfpu_base_dst + k, /*ct_dim=*/1, /*rt_dim=*/1);
                    }
                }

                const uint32_t base_dst = get_dst_index(accumulate);
                if (run_post_op) {
                    for (uint32_t i = 0; i < current_chunk; ++i) {
                        post_reduce_op(base_dst + i);
                    }
                }

                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < current_chunk; ++i) {
                    output_dfb.reserve_back(onetile);
                    pack_tile(base_dst + i, output_dfb_id);
                    output_dfb.push_back(onetile);
                }
                tile_regs_release();

                if constexpr (waits_bulk(input_policy)) {
                    input_dfb.pop_front(tiles_in_chunk);
                }
            }
            if constexpr (!should_pop(input_policy)) {
                batch_offset += batch_stride;
            }
        }
    }

    if constexpr (is_sfpu) {
        PACK((llk_pack_reduce_mask_clear()));
    } else {
        // Quasar restores its MXFP4 source-format override from the input metadata.
        reduce_uninit(input_dfb_id);
    }
}

namespace detail {

template <typename Call, typename PostReduceOp>
ALWI void reduce_planned_variant(PostReduceOp post_reduce_op) {
    auto shape = ReduceInputBlockShape::of(Call::rows, Call::columns, Call::batches);
    auto layout = Call::row_stride == 0 ? ReduceInputMemoryLayout::contiguous()
                                        : ReduceInputMemoryLayout::with_row_stride(Call::row_stride);
    if constexpr (Call::is_tail) {
        static_assert(
            Call::reduce_dim != ReduceDim::REDUCE_SCALAR ||
                (!should_pop(Call::input_policy) && Call::batches == 1 && Call::logical_h % 32 == 0 &&
                 Call::logical_w == Call::columns * 32),
            "HW tails require resident whole tile rows at the full width, in a single batch");
        shape = ReduceInputBlockShape::of((Call::logical_h + 31) / 32, (Call::logical_w + 31) / 32, Call::batches);
        if constexpr (!should_pop(Call::input_policy)) {
            // A resident tail keeps the pitches of the full block it lives in.
            constexpr uint32_t row_pitch = Call::row_stride == 0 ? Call::columns : Call::row_stride;
            layout = ReduceInputMemoryLayout::with_strides(row_pitch, Call::rows * row_pitch);
        }
    }
    if constexpr (Call::accumulation_mode != ttnn::kernel_lib::ReduceAccumulationMode::None) {
        // Each call packs every partial before the next call reloads any, on this same thread.
        [[maybe_unused]] const uint32_t outputs =
            Call::reduce_dim == ReduceDim::REDUCE_ROW   ? shape.rows * shape.batches
            : Call::reduce_dim == ReduceDim::REDUCE_COL ? shape.cols * shape.batches
                                                        : shape.batches;
        UNPACK(ASSERT(get_dfb_num_pages(Call::accumulator_cb_id) >= outputs));
    }
    // Out of line, the call keeps its shape and accumulation as runtime values and compiles the generic
    // reduce body, several times the inlined size per call. TODO: forcing it inline grows kernel_main past
    // GCC's large-function limits, which then stops inlining further down; find a structure that keeps
    // planned calls constant-folded without forcing it.
    auto issue = [&](auto accumulate, auto post_op) __attribute__((always_inline)) {
        reduce<
            Call::reduce_type,
            Call::reduce_dim,
            Call::input_cb_id,
            Call::auxiliary_cb_id,
            Call::output_cb_id,
            Call::input_policy,
            Call::reconfig_mode,
            Call::fp32_mode,
            Call::algorithm,
            Call::within_tile,
            Call::scale_bits>(
            shape,
            layout,
            accumulate,
            post_op,
            Call::partial_mode,
            Call::output_chunk_tiles,
            Call::auxiliary_tile_offset);
    };

    if constexpr (Call::accumulation_mode == ttnn::kernel_lib::ReduceAccumulationMode::None) {
        issue(NoAccumulation{}, post_reduce_op);
    } else if constexpr (Call::accumulation_mode == ttnn::kernel_lib::ReduceAccumulationMode::Final) {
        issue(
            Accumulate::at_last(Call::accumulator_cb_id, Call::accumulation_index).with_reload(Call::reload_mode),
            post_reduce_op);
    } else {
        static_assert(
            Call::accumulation_mode == ttnn::kernel_lib::ReduceAccumulationMode::Intermediate,
            "Unknown planned reduction accumulation mode");
        issue(Accumulate::at(Call::accumulator_cb_id, Call::accumulation_index).with_reload(Call::reload_mode), NoOp{});
    }
}

}  // namespace detail

template <typename Call, typename PostReduceOp>
ALWI void reduce(PostReduceOp post_reduce_op) {
    if constexpr (Call::has_tail_variant) {
        // One compiled call serves the whole grid; the runtime override selects the tail's traversal,
        // masks, auxiliary slice and normalization together.
        if (Call::use_tail()) {
            detail::reduce_planned_variant<typename Call::Tail>(post_reduce_op);
        } else {
            detail::reduce_planned_variant<Call>(post_reduce_op);
        }
    } else {
        detail::reduce_planned_variant<Call>(post_reduce_op);
    }
}

}  // namespace compute_kernel_lib
