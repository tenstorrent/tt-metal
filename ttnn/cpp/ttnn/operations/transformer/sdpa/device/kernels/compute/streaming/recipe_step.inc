// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The streaming recipes' K-chunk step: the max reduce, the exps, normalization, the fused chunk and the K-chunk
// loop. recipe_streaming.hpp builds it twice, in namespaces recipe_cold_step and recipe_hot_step, with
// SDPA_RECIPE_COLD defined around the include as these functions' attributes: with fused chunks the reduce path (a Q
// chunk's first K chunk, redone groups) and normalization are cold, so out of line and size-optimized (also their
// lambdas, which take the enclosing function's optimization) to fit the ring kernels in the kernel config buffer;
// every other build keeps them at the kernel's level. recipe::fused_active picks the copy (sdpa_segment_v2).

// abs_row: pack the maxima at that absolute row of an already-published out_cb (a fused chunk's redo)
// instead of appending them.
template <uint32_t in0_cb, uint32_t scale_cb, uint32_t row_stride, uint32_t ref_scale_fp32 = 0>
SDPA_RECIPE_COLD void reduce_c_row_group(
    uint32_t out_cb,
    uint32_t prev_cb,
    uint32_t row_group_index,
    bool do_eltwise_max,
    uint32_t sbh,
    uint32_t reduce_cols,
    bool respect_trigger = false,
    bool overlap_first_half = false,
    uint32_t abs_row = UINT32_MAX) {
    const uint32_t group_size = sbh;
    const uint32_t row_start = row_group_index * group_size;

    // row_stride: physical row width in the CB (may exceed cols on the reduced path).
    const uint32_t cumulative_input_tiles = (row_group_index + 1) * group_size * row_stride;
    const uint32_t cumulative_prev_tiles = (row_group_index + 1) * group_size;

    // scale_cb assumed ready (waited once at kernel init)

    tile_regs_acquire();

    if (do_eltwise_max) {
        CircularBuffer(prev_cb).wait_front(cumulative_prev_tiles);
        if constexpr (recipe::fp32) {
            sdpa_stream_reconfig_srca(prev_cb);
        }
        sdpa_reduce_copy_tile_to_dst_init_short(prev_cb);
        for (uint32_t i = 0; i < group_size; i++) {
            copy_tile(prev_cb, row_start + i, i);
        }
    }

    // Deferred: wait for in0_cb just before its first use (reduce_block_max_row).
    // When do_eltwise_max=true, the prev_cb wait + copy_tile work above can overlap
    // with in0_cb data arrival.
    // When respect_trigger=true, the unpack MOP is split into two halves with a
    // HW semaphore wait in between, so we don't need a wait-front here.
    if (!respect_trigger) {
        CircularBuffer(in0_cb).wait_front(cumulative_input_tiles);
    }

    if constexpr (recipe::fp32) {
        sdpa_stream_reconfig(in0_cb, scale_cb);
    }
    reduce_block_max_row_init_runtime(out_cb, reduce_cols, in0_cb, scale_cb, respect_trigger);
    for (uint32_t i = 0; i < group_size; i++) {
        const uint32_t input_tile_start = (row_start + i) * row_stride;
        reduce_block_max_row_runtime(in0_cb, scale_cb, input_tile_start, i, respect_trigger, overlap_first_half);
    }
    reduce_block_max_row_uninit_runtime(in0_cb, respect_trigger, overlap_first_half);
    if constexpr (!recipe::fp32) {
        // The reference (m_ref) for row i in slot i + 2 (a row pair, or a single-row tail group), for the
        // conditional rescale below.
        if (do_eltwise_max) {
            copy_init(prev_cb);
            for (uint32_t i = 0; i < group_size; i++) {
                copy_tile(prev_cb, row_start + i, 2 + i);
            }
        }
    }

    tile_regs_commit();
    tile_regs_wait();
    if constexpr (!recipe::fp32) {
        if (do_eltwise_max) {
            for (uint32_t i = 0; i < group_size; i++) {
                PACK((SFPU_UNARY_CALL(
                    DST_SYNC_MODE,
                    DST_ACCUM_MODE,
                    calculate_sdpa_ref_max_select,
                    (ref_scale_fp32, 2),
                    i,
                    VectorMode::C)));
            }
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        }
        if constexpr (recipe::fused_active) {
            for (uint32_t i = 0; i < group_size; i++) {
                PACK((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
                    DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_trunc7, i, VectorMode::C)));
            }
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        }
    }
    if constexpr (recipe::fp32) {
        configure_single_tile_pack(out_cb);
    }

    for (uint32_t i = 0; i < group_size; i++) {
        if (abs_row == UINT32_MAX) {
            pack_tile<false>(i, out_cb);
        } else {
            pack_tile<true>(i, out_cb, abs_row + i);
        }
    }

    tile_regs_release();
}

// ACCURATE: apply the row maximum once to the entire FP32 score row using
// the L1 FP32 adder, then reload four score tiles without a resident maximum.
static void sdpa_subtract_max_l1(uint32_t inout_cb, uint32_t max_cb, uint32_t row, uint32_t cols) {
    tile_regs_acquire();
    sdpa_stream_reconfig(max_cb, max_cb);
    unary_bcast_init<BroadcastType::COL>(max_cb);
    unary_bcast<BroadcastType::COL>(max_cb, row, 0);
    unary_bcast_uninit<BroadcastType::COL>(max_cb);
    tile_regs_commit();
    tile_regs_wait();
    PACK((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_negate_max, 0, VectorMode::None)));
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
    configure_single_tile_pack(inout_cb);
    PACK((llk_pack_relu_config(ReluConfig::none())));
    PACK((llk_pack_reconfig_l1_acc(1)));
    for (uint32_t col = 0; col < cols; ++col) {
        pack_tile<true>(0, inout_cb, row * cols + col);
    }
    PACK((llk_pack_reconfig_l1_acc(0)));
    tile_regs_release();
    recipe_pack_to_unpack_barrier();
}

/**
 * In-place sub_exp on cb_qkt_im: subtracts max, applies exp with ReLU clamping,
 * writes back to same positions. Accumulates row sums into reduce_cb.
 */
template <uint32_t scale_fp32>
SDPA_RECIPE_COLD void sub_exp_block_bcast_cols(
    uint32_t inout_cb,
    uint32_t max_cb,
    uint32_t reduce_cb,
    uint32_t cols_in_row,
    uint32_t q_subblock,
    uint32_t global_col_base,
    uint32_t sbh,
    uint32_t sbw,
    bool skip_pack_configure = false,
    bool wait_max = true) {
    const uint32_t tiles_per_row = sbh;
    const uint32_t tiles_per_column = sbw;
    const uint32_t max_row_base = q_subblock * tiles_per_row;

    if constexpr (recipe::accurate) {
        // One batch per QK subblock column group (the subblock width divides the K chunk).
        constexpr uint32_t score_batch = recipe::qk_w;
        static_assert(score_batch == 1 || score_batch == 2 || score_batch == 4);
        CircularBuffer(max_cb).wait_front((q_subblock + 1) * tiles_per_row);
        if (global_col_base == 0) {
            sdpa_subtract_max_l1(inout_cb, max_cb, max_row_base, cols_in_row);
        }
        cb_alias_read_ptr(7, inout_cb);
        configure_pack_width(inout_cb, score_batch);
        PACK((llk_pack_relu_config(ReluConfig::zero())));
        for (uint32_t i = 0; i < tiles_per_row; ++i) {
            for (uint32_t j = 0; j < tiles_per_column; j += score_batch) {
                const uint32_t index = (max_row_base + i) * cols_in_row + global_col_base + j;
                tile_regs_acquire();
                sdpa_stream_reconfig(7, 7);
                unary_bcast_init<BroadcastType::NONE>(7);
                // Two adjacent FP32 score tiles' eight contiguous faces in one unpack-to-DST handshake.
                sdpa::streaming::state_unpack_mop(4 * score_batch);
                unary_bcast<BroadcastType::NONE>(7, index, 0);
                // The generic one-tile helper cleared tile 0's zero flags. Preserve
                // the Blackhole unpack-to-DST workaround for tile 1 as well.
                MATH(for (uint32_t tile = 1; tile < score_batch; ++tile) {
                    for (uint32_t face = 0; face < 4; ++face) {
                        TT_ZEROACC(p_zeroacc::CLR_16, 1, 1, ADDR_MOD_3, get_dest_index_in_faces(tile, face));
                    }
                })
                sdpa::streaming::state_unpack_mop(4);
                unary_bcast_uninit<BroadcastType::NONE>(7);
                tile_regs_commit();
                tile_regs_wait();
                PACK((ckernel::sfpu::restore_sdpa_grid_macro_instructions()));
                PACK((ckernel::sfpu::init_sdpa_exp_grid<scale_fp32>()));
                PACK((SFPU_UNARY_CALL(
                    DST_SYNC_MODE,
                    DST_ACCUM_MODE,
                    calculate_sdpa_exp_grid_batch,
                    (32 * score_batch),
                    0,
                    VectorMode::None)));
                PACK((SFPU_UNARY_CALL(
                    DST_SYNC_MODE,
                    DST_ACCUM_MODE,
                    calculate_sdpa_exp_refine_loadmacro,
                    (32 * score_batch, true),
                    0,
                    VectorMode::None)));
                PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                pack_tile<true>(0, inout_cb, index);
                tile_regs_release();
            }
        }
        sdpa_stream_reconfig(inout_cb, max_cb);
        configure_row_pack_width(inout_cb, tiles_per_column);
        PACK((llk_pack_relu_config(ReluConfig::none())));
        PACK((llk_pack_reconfig_l1_acc(0)));
        return;
    } else {
        {
            if constexpr (recipe::fp32) {
                // Both sources: after the QK matmul srcB still holds Q's BF16 format, and the maxima are FP32.
                sdpa_stream_reconfig(inout_cb, max_cb);
            }
            sub_bcast_cols_init_short_custom(inout_cb, max_cb, tiles_per_column);
        }

        // inout_cb assumed ready (max_cb was already computed from it). A fused chunk's redo reads unpublished
        // scratch maxima (wait_max = false).
        if (wait_max) {
            CircularBuffer(max_cb).wait_front((q_subblock + 1) * tiles_per_row);
        }

        tile_regs_acquire();
        {
            uint32_t dst_index = 0;
            for (uint32_t i = 0; i < tiles_per_row; i++) {
                uint32_t in0_tile_index = (max_row_base + i) * cols_in_row + global_col_base;
                sub_tiles_bcast_cols_custom(
                    inout_cb, max_cb, in0_tile_index, max_row_base + i, dst_index, tiles_per_column);
                dst_index += tiles_per_column;
            }
        }
        tile_regs_commit();

        tile_regs_wait();
        PACK((llk_pack_relu_config(ReluConfig::zero())));
        {
            uint32_t dst_index = 0;
            constexpr int iterations = 32;
            constexpr VectorMode vector_mode_exp = VectorMode::None;
            if constexpr (!recipe::fp32) {
                for (uint32_t i = 0; i < tiles_per_row; i++) {
                    for (uint32_t j = 0; j < tiles_per_column; j++) {
                        exp_packthread_tile<true, false, InputClamping::None, iterations>(dst_index++, vector_mode_exp);
                    }
                }
            } else if (tiles_per_row * tiles_per_column == 4) {
                PACK((ckernel::sfpu::restore_sdpa_grid_macro_instructions()));
                PACK((ckernel::sfpu::init_sdpa_exp_grid<scale_fp32>()));
                PACK((SFPU_UNARY_CALL(
                    DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_exp_grid_batch, (128), 0, vector_mode_exp)));
                PACK((SFPU_UNARY_CALL(
                    DST_SYNC_MODE,
                    DST_ACCUM_MODE,
                    calculate_sdpa_exp_refine_loadmacro,
                    (128, false),
                    0,
                    vector_mode_exp)));
            } else {
                for (uint32_t i = 0; i < tiles_per_row; i++) {
                    for (uint32_t j = 0; j < tiles_per_column; j++) {
                        PACK((ckernel::sfpu::restore_sdpa_grid_macro_instructions()));
                        PACK((ckernel::sfpu::init_sdpa_exp_grid<scale_fp32>()));
                        PACK((SFPU_UNARY_CALL(
                            DST_SYNC_MODE,
                            DST_ACCUM_MODE,
                            calculate_sdpa_exp_grid_batch,
                            (iterations),
                            dst_index,
                            vector_mode_exp)));
                        PACK((SFPU_UNARY_CALL(
                            DST_SYNC_MODE,
                            DST_ACCUM_MODE,
                            calculate_sdpa_exp_refine_loadmacro,
                            (iterations, false),
                            dst_index,
                            vector_mode_exp)));
                        ++dst_index;
                    }
                }
            }
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        }

        {
            // Pack back to inout_cb at the same absolute positions.
            // In Phase 1, the caller pre-configures (cb_qkt_im, actual_sbw) before the kt loop
            // and blocked_matmul_and_pack restores it after each sub_exp. Skip the redundant
            // reconfigure here when the caller guarantees the state.
            if (skip_pack_configure) {
                pack_contiguous_rows_nocfg(
                    inout_cb, max_row_base, tiles_per_row, cols_in_row, global_col_base, tiles_per_column);
            } else {
                pack_contiguous_rows(
                    inout_cb, max_row_base, tiles_per_row, cols_in_row, global_col_base, tiles_per_column);
            }
            if constexpr (!recipe::fp32) {
                configure_single_tile_pack(reduce_cb);
                {
                    uint32_t dst_index = 0;
#pragma GCC unroll 1
                    for (uint32_t i = 0; i < tiles_per_row; i++) {
                        if (global_col_base > 0) {
                            PACK((llk_pack_reconfig_l1_acc(1)));
                        } else {
                            PACK((llk_pack_reconfig_l1_acc(0)));
                        }
#pragma GCC unroll 1
                        for (uint32_t j = 0; j < tiles_per_column; ++j) {
                            pack_tile<true>(dst_index++, reduce_cb, max_row_base + i);
                            if (global_col_base == 0 && j == 0) {
                                PACK((llk_pack_reconfig_l1_acc(1)));
                            }
                        }
                    }
                }
            }
        }

        tile_regs_release();

        // Restore packer ReLU config after all exp operations complete
        PACK((llk_pack_relu_config(ReluConfig::none())));
        PACK((llk_pack_reconfig_l1_acc(0)));
    }
}

// BF16 recipes; UNPACK reads the maxima (get_tile_l1_byte_address is UNPACK-only).
inline uint32_t sdpa_scan_identity_maxima(
    uint32_t in0_cb, uint32_t in1_cb, uint32_t q_subblock, uint32_t tiles_per_row) {
    uint32_t identical = 1;
    UNPACK({
        const uint32_t global_row_base = q_subblock * tiles_per_row;
        for (uint32_t t = 0; t < tiles_per_row && identical; ++t) {
            auto* old_max = reinterpret_cast<volatile uint32_t*>(
                get_tile_l1_byte_address(get_operand_id(in0_cb), global_row_base + t));
            auto* new_max = reinterpret_cast<volatile uint32_t*>(
                get_tile_l1_byte_address(get_operand_id(in1_cb), global_row_base + t));
            uint32_t difference = 0;
            uint32_t nonfinite = 0;
            // Same 32 row values, but fixed native-face strides reduce
            // per-row index and branch/control bookkeeping. Checking
            // old's exponent is sufficient once raw old/new bits are equal.
            for (uint32_t face = 0; face < 2; ++face) {
                _Pragma("GCC unroll 16") for (uint32_t r = 0; r < 16; ++r) {
                    const uint32_t offset = face * 256 + r * 8;
                    const uint32_t a = old_max[offset] & 0xffffu;
                    const uint32_t b = new_max[offset] & 0xffffu;
                    difference |= a ^ b;
                    nonfinite |= ((a & 0x7f80u) == 0x7f80u);
                }
            }
            identical &= ((difference | nonfinite) == 0);
        }
    })

    return identical;
}

/**
 * Column-only exp(prev_max - cur_max) for SALAD corrections.
 * Operates on first-column subset of tiles.
 */
// The BF16 recipes pass their Phase-1 identity scan's verdict (precomputed_identity); the FP32 recipes compare here.
template <uint32_t scale_fp32, typename... PrecomputedIdentity>
bool sub_exp_first_col_blocks(
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    uint32_t q_subblock,
    uint32_t sbh,
    PrecomputedIdentity... precomputed_identity) {
    static_assert(sizeof...(PrecomputedIdentity) == (recipe::fp32 ? 0 : 1));
    const uint32_t tiles_per_row = sbh;
    const uint32_t global_row_base = q_subblock * tiles_per_row;
    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;

    if constexpr (!recipe::fp32) {
        sub_init(in0_cb, in1_cb);
    }

    CircularBuffer(in0_cb).wait_front((q_subblock + 1) * tiles_per_row);
    CircularBuffer(in1_cb).wait_front((q_subblock + 1) * tiles_per_row);

    uint32_t identical = 1;
    UNPACK({
        if constexpr (recipe::fp32) {
            for (uint32_t t = 0; t < tiles_per_row && identical; ++t) {
                auto* old_max = reinterpret_cast<volatile uint32_t*>(
                    get_tile_l1_byte_address(get_operand_id(in0_cb), global_row_base + t));
                auto* new_max = reinterpret_cast<volatile uint32_t*>(
                    get_tile_l1_byte_address(get_operand_id(in1_cb), global_row_base + t));
                // The 32 FP32 first-column comparisons (sdpa_max_word), four independent loads per loop
                // control iteration.
                for (uint32_t r = 0; r < 32; r += 4) {
                    const uint32_t offset = sdpa_max_word(r);
                    const uint32_t difference =
                        (old_max[offset] ^ new_max[offset]) | (old_max[offset + 16] ^ new_max[offset + 16]) |
                        (old_max[offset + 32] ^ new_max[offset + 32]) | (old_max[offset + 48] ^ new_max[offset + 48]);
                    if (difference != 0) {
                        identical = 0;
                        break;
                    }
                }
            }
        } else {
            identical = (precomputed_identity, ...);
        }
        mailbox_write(ckernel::ThreadId::MathThreadId, identical);
        mailbox_write(ckernel::ThreadId::PackThreadId, identical);
    })
    MATH(identical = mailbox_read(ckernel::ThreadId::UnpackThreadId);)
    PACK(identical = mailbox_read(ckernel::ThreadId::UnpackThreadId);)
    if (identical) {
        return true;
    }
    if constexpr (recipe::fp32) {
        // An exact identity correction never consumes either subtraction input.
        // Preserve the comparison and arithmetic, but skip unused operation setup.
        sdpa_stream_reconfig(in0_cb, in1_cb);
        sub_init(in0_cb, in1_cb);
    }

    {
        tile_regs_acquire();
        for (uint32_t i = 0; i < tiles_per_row; i++) {
            uint32_t tile_index = global_row_base + i;
            sub_tiles(in0_cb, in1_cb, tile_index, tile_index, i);
        }
        tile_regs_commit();
    }

    {
        tile_regs_wait();
        for (uint32_t dst_index = 0; dst_index < tiles_per_row; dst_index++) {
            PACK((SFPU_UNARY_CALL(
                DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_exp_correction, (scale_fp32), dst_index, VectorMode::C)));
        }
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));

        configure_single_tile_pack(out_cb);
        for (uint32_t i = 0; i < tiles_per_row; i++) {
            pack_tile<false>(i, out_cb);
        }

        tile_regs_release();
    }
    return false;
}

/**
 * Fused SALAD correction: output correction + sum correction in one init cycle.
 * Folds the sum correction tile(s) into the output correction's last DEST batch
 * when there's room, or appends a minimal extra batch otherwise.
 * Eliminates the separate init + acquire/release overhead for sum correction.
 *
 * ob_q_subblock controls read offset for out_in_cb and bcast_cb (0 when popped row-by-row).
 * sum_q_subblock controls read offset for sum_in_cb (cumulative when not popped per-row).
 * FP32 recipes.
 */
template <uint32_t sbh_t, uint32_t sbw_t, uint32_t dst_size>
void salad_correct_fused(
    uint32_t out_in_cb,
    uint32_t sum_in_cb,
    uint32_t bcast_cb,
    uint32_t out_out_cb,
    uint32_t sum_out_cb,
    uint32_t ob_q_subblock,
    uint32_t sum_q_subblock,
    uint32_t write_q_subblock,
    bool current_sum_popped = false,
    bool identity_correction = false) {
    constexpr uint32_t tiles_per_row = sbh_t;
    constexpr uint32_t tiles_per_column = sbw_t;

    // out_in_cb and bcast_cb may be popped row-by-row (ob_q_subblock=0) while
    // sum_in_cb retains cumulative indexing (sum_q_subblock=salad_row).
    const uint32_t ob_row_base = ob_q_subblock * tiles_per_row;
    const uint32_t sum_row_base = sum_q_subblock * tiles_per_row;
    const uint32_t write_row_base = write_q_subblock * tiles_per_row;

    CircularBuffer(out_in_cb).wait_front((ob_q_subblock + 1) * tiles_per_row * tiles_per_column);
    CircularBuffer(sum_in_cb).wait_front((sum_q_subblock + 1) * tiles_per_row);
    CircularBuffer(bcast_cb).wait_front((ob_q_subblock + 1) * tiles_per_row);
    PACK((llk_pack_reconfig_l1_acc(1)));
    configure_single_tile_pack(out_out_cb);
    for (uint32_t i = 0; i < tiles_per_row; ++i) {
        if (identity_correction && tiles_per_column == 4) {
            // Identity needs no correction tile, so all four FP32 DST slots
            // hold independent numerator tiles. L1 adds remain unchanged.
            sdpa::streaming::rescale_and_accumulate<4>(
                out_in_cb,
                out_out_cb,
                bcast_cb,
                (ob_row_base + i) * tiles_per_column,
                (write_row_base + i) * tiles_per_column,
                ob_row_base + i,
                true);
        } else {
            for (uint32_t j = 0; j + 1 < tiles_per_column; j += 2) {
                sdpa::streaming::rescale_and_accumulate<2>(
                    out_in_cb,
                    out_out_cb,
                    bcast_cb,
                    (ob_row_base + i) * tiles_per_column + j,
                    (write_row_base + i) * tiles_per_column + j,
                    ob_row_base + i,
                    identity_correction);
            }
            if (tiles_per_column % 2 != 0) {
                // Odd head-dim width: the last numerator column on its own.
                const uint32_t j = tiles_per_column - 1;
                sdpa::streaming::rescale_and_accumulate<1>(
                    out_in_cb,
                    out_out_cb,
                    bcast_cb,
                    (ob_row_base + i) * tiles_per_column + j,
                    (write_row_base + i) * tiles_per_column + j,
                    ob_row_base + i,
                    identity_correction);
            }
        }
        sdpa::streaming::rescale_and_accumulate<1, true>(
            sum_in_cb,
            sum_out_cb,
            bcast_cb,
            sum_row_base + i,
            write_row_base + i,
            ob_row_base + i,
            identity_correction);
    }
    PACK((llk_pack_reconfig_l1_acc(1)));
}

/**
 * Per-row streaming normalization: matmul_reduce + recip-in-DST + mul_bcast_cols.
 * Consumes (pops) sum and output tiles, writes normalized output.
 * scratch_cb is a 1-tile CB reused for the reciprocal intermediate.
 */
template <
    uint32_t head_dim_t_,
    uint32_t dst_size,
    uint32_t col_identity_cb,
    uint32_t scratch_cb,
    uint32_t normalized_out_cb>
static __attribute__((noinline, noclone)) SDPA_RECIPE_COLD void normalize_row_streaming(
    uint32_t cur_sum_cb, uint32_t cur_out_cb, uint32_t sbh) {
    if constexpr (recipe::fp32) {
        sdpa::streaming::normalize_rows<head_dim_t_, col_identity_cb>(
            cur_sum_cb, cur_out_cb, scratch_cb, normalized_out_cb, sbh, sdpa_pack);
    } else {
        configure_single_tile_pack(scratch_cb);
        for (uint32_t s = 0; s < sbh; s++) {
            // 1+2. Fused matmul_reduce + recip: sum × col_identity → recip → 1/sum in scratch
            {
                constexpr uint32_t N = 1;
                // The matmul unpack path does not read Float32 into a 16-bit dest: round the accumulated
                // denominator to BF16 through dest into the one-tile CB 8 first.
                constexpr uint32_t sum_bf16_cb = 8;
                CircularBuffer(cur_sum_cb).wait_front(1);
                reconfig_data_format_srca(cur_sum_cb);
                copy_init(cur_sum_cb);
                CircularBuffer(sum_bf16_cb).reserve_back(1);
                tile_regs_acquire();
                copy_tile(cur_sum_cb, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(sum_bf16_cb);
                configure_single_tile_pack(sum_bf16_cb);
                pack_tile(0, sum_bf16_cb);
                tile_regs_release();
                CircularBuffer(sum_bf16_cb).push_back(1);
                reconfig_data_format_srca(sum_bf16_cb);
                configure_single_tile_pack(scratch_cb);
                const uint32_t norm_sum_cb = sum_bf16_cb;
                if constexpr (recipe::lofi) {
                    // The no-MOP matmul the chunks already use: FAST's ring kernels then carry no MOP matmul code.
                    recipe_mm_init(norm_sum_cb, col_identity_cb, false, N, 1, N);
                } else {
                    matmul_block_init(norm_sum_cb, col_identity_cb, 0, N, 1, N);
                }
                sdpa_maybe_reconfig_data_format<normalized_out_cb, col_identity_cb, normalized_out_cb, scratch_cb>();
                // Pack format follows scratch_cb for the reciprocal intermediate. The old/new form folds away
                // when scratch and normalized output formats match, and reconfigures after rows that packed output.
                sdpa_maybe_pack_reconfig_data_format<normalized_out_cb, scratch_cb>();

                CircularBuffer(col_identity_cb).wait_front(N);
                CircularBuffer(norm_sum_cb).wait_front(1);

                CircularBuffer(scratch_cb).reserve_back(1);
                tile_regs_acquire();
                if constexpr (recipe::lofi) {
                    matmul_block_no_mop(norm_sum_cb, col_identity_cb, 0, 0, 0, false, N, 1, N);
                } else {
                    matmul_block(norm_sum_cb, col_identity_cb, 0, 0, 0, 0, N, 1, N);
                }

                recip_tile_init();
                MATH((recip_tile(0 /*dst_index*/, VectorMode::C)));
                tile_regs_commit();

                tile_regs_wait();
                pack_tile(0, scratch_cb);
                tile_regs_release();
                CircularBuffer(scratch_cb).push_back(1);

                CircularBuffer(norm_sum_cb).pop_front(1);
                CircularBuffer(cur_sum_cb).pop_front(1);
            }

            // 3. Normalize: multiply output tiles by bcast_cols(1/sum)
            // Process in batches of up to dst_size tiles (DST capacity).
            {
                constexpr uint32_t batch = (head_dim_t_ < dst_size) ? head_dim_t_ : dst_size;
                reconfig_data_format_srca(cur_out_cb);
                recipe_output_scale_init(cur_out_cb, scratch_cb);
                // Pack output to normalized_out_cb; old/new skips when it has the same format as scratch.
                sdpa_maybe_pack_reconfig_data_format<scratch_cb, normalized_out_cb>();
                CircularBuffer(cur_out_cb).wait_front(head_dim_t_ * sdpa_out_stride);
                CircularBuffer(scratch_cb).wait_front(1);

                CircularBuffer(normalized_out_cb).reserve_back(head_dim_t_);
                for (uint32_t base = 0; base < head_dim_t_; base += batch) {
                    constexpr uint32_t last_batch = head_dim_t_ % batch;
                    const uint32_t cur_batch = (base + batch <= head_dim_t_) ? batch : last_batch;
                    tile_regs_acquire();
                    for (uint32_t j = 0; j < cur_batch; ++j) {
                        recipe_output_scale_tile(cur_out_cb, scratch_cb, base + j, 0, j);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t j = 0; j < cur_batch; ++j) {
                        pack_tile(j, normalized_out_cb);
                    }
                    tile_regs_release();
                }
                CircularBuffer(normalized_out_cb).push_back(head_dim_t_);

                CircularBuffer(scratch_cb).pop_front(1);
                CircularBuffer(cur_out_cb).pop_front(head_dim_t_ * sdpa_out_stride);
                reconfig_data_format_srca(normalized_out_cb);
            }
        }
        // Restore pack format to scratch_cb (im_df = Float16_b) so that subsequent ops
        // (e.g. salad_correct_fused on the next K-chunk's drain row) pack to F16b CBs
        // with the right format. Without this, when normalized_out_cb has a different
        // format (e.g. Bfp8 output dtype), the format register stays Bfp8 and the next
        // pack to a F16b CB writes garbage that's later mis-decoded by F16b unpacks.
        sdpa_maybe_pack_reconfig_data_format<normalized_out_cb, scratch_cb>();
    }
}

// Dense additive attn_mask (legacy SDPA contract): L1-accumulate one QK row group's mask tiles onto
// the packed scores in cb_qkt_im, before the row max. The reader streams one Q chunk x K chunk of
// mask per K chunk into CB 15, one Q tile row at a time. The add runs in the packer's L1
// accumulator (the same adder legacy SDPA uses); no recipe SFPU/FPU arithmetic changes, and
// unmasked builds compile this out.
constexpr uint32_t kRecipeMaskCb = 15;
// The reader always pushes whole group_rows groups (an odd chunk's single-row tail group carries a
// zero padding row), so waits/pops stay group-aligned in a one- or two-group CB.
template <uint32_t cb_qkt_im, uint32_t k_tiles, uint32_t group_rows>
static __attribute__((noinline, noclone)) void recipe_add_attn_mask(uint32_t row_start, uint32_t rows) {
    constexpr uint32_t batch = compute_kernel_lib::DEST_AUTO_LIMIT;
    const uint32_t tiles = rows * k_tiles;
    CircularBuffer(kRecipeMaskCb).wait_front(group_rows * k_tiles);
    configure_single_tile_pack(cb_qkt_im);
    sdpa_stream_reconfig_srca(kRecipeMaskCb);
    copy_init(kRecipeMaskCb);
    PACK((llk_pack_reconfig_l1_acc(1)));
    const uint32_t out_base = row_start * k_tiles;
    for (uint32_t i = 0; i < tiles; i += batch) {
        const uint32_t n = tiles - i < batch ? tiles - i : batch;
        tile_regs_acquire();
        for (uint32_t j = 0; j < n; ++j) {
            copy_tile(kRecipeMaskCb, i + j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < n; ++j) {
            pack_tile<true>(j, cb_qkt_im, out_base + i + j);
        }
        tile_regs_release();
    }
    PACK((llk_pack_reconfig_l1_acc(0)));
    CircularBuffer(kRecipeMaskCb).pop_front(group_rows * k_tiles);
    // Callers expect srcA configured for the score CB (max reduce / next row group).
    sdpa_stream_reconfig_srca(cb_qkt_im);
}

// The fused chunk: size-optimized UNPACK copy for the ring kernels (unpack only issues and waits there, and its image
// is the one nearest the kernel config buffer limit), every other copy out of line at the kernel's level.
namespace fused_os {
#define SDPA_FUSED_CHUNK_ATTR __attribute__((noinline, optimize("Os")))
#include "recipe_fused_chunk.inc"
#undef SDPA_FUSED_CHUNK_ATTR
}  // namespace fused_os
namespace fused_plain {
#define SDPA_FUSED_CHUNK_ATTR __attribute__((noinline))
#include "recipe_fused_chunk.inc"
#undef SDPA_FUSED_CHUNK_ATTR
}  // namespace fused_plain

/**
 * One K-chunk iteration of the streaming SDPA algorithm (v2 — no row buffers).
 * Phase 1: Q@KT directly into cb_qkt_im with cb_push_back_hold_wr_ptr, in-place sub_exp.
 * Phase 2: Drain + QKT@V with SALAD corrections, streaming normalization on last K iter.
 */
template <
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t DHt,
    uint32_t vDHt,
    uint32_t scale_fp32,
    uint32_t qkt_subblock_h,
    uint32_t qkt_subblock_w,
    uint32_t qktv_subblock_h,
    uint32_t qktv_subblock_w,
    uint32_t cb_q_in,
    uint32_t cb_kt_in,
    uint32_t cb_v_in,
    uint32_t cb_qkt_im,
    uint32_t cb_identity_scale_in,
    uint32_t cb_exp_max_diff,
    uint32_t cb_col_identity,
    uint32_t cb_recip_scratch,
    uint32_t cb_normalized_out,
    bool independent_q_release = false>
static SDPA_RECIPE_COLD bool sdpa_inner_loop_step(
    AccumulatorHalf& prev, AccumulatorHalf& cur, bool is_last_iter, bool is_first_iter, bool release_q = false) {
    constexpr uint32_t KT_stride = Sk_chunk_t;
    constexpr uint32_t active_Sk = Sk_chunk_t;
    constexpr uint32_t actual_sbw = qkt_subblock_w;
    constexpr bool reduce_trigger = reduce_trigger_supported && Sk_chunk_t % qkt_subblock_w == 0 &&
                                    Sk_chunk_t / qkt_subblock_w > 1 && Sk_chunk_t % 2 == 0;
    if constexpr (recipe::fp32) {
        PACK(sdpa_pack.format_cb = INVALID_CB; sdpa_pack.width = 0;)
        sdpa_skip_prev_sum_pop = false;
    } else {
        // Row groups pair two query tile rows; an odd Q chunk ends with a single-row group.
        static_assert(
            Sq_chunk_t >= 1 && Sq_chunk_t <= kRecipeMaxQTiles && vDHt >= 1 && qkt_subblock_h == 2 &&
            qktv_subblock_h == 2);
    }
    const uint32_t kt_num_full_subblocks = active_Sk / actual_sbw;
    constexpr uint32_t dst_size = compute_kernel_lib::DEST_AUTO_LIMIT;
    constexpr uint32_t in0_block_w = DHt;
    // An odd Q chunk (BF16 recipes, two-row subblocks) ends with a one-row subblock. Helpers
    // address rows as index * height, so that subblock uses row index 2 * (n - 1) and height 1.
    constexpr bool q_tail_row = Sq_chunk_t % qkt_subblock_h != 0;
    constexpr uint32_t q_num_subblocks = Sq_chunk_t / qkt_subblock_h + (q_tail_row ? 1 : 0);
    constexpr uint32_t q_subblock_num_tiles = qkt_subblock_h * in0_block_w;
    auto qk_rows = [](uint32_t q_subblock) -> uint32_t {
        return q_tail_row && q_subblock == q_num_subblocks - 1 ? 1 : qkt_subblock_h;
    };
    auto qk_index = [](uint32_t q_subblock) -> uint32_t {
        return q_tail_row && q_subblock == q_num_subblocks - 1 ? q_subblock * qkt_subblock_h : q_subblock;
    };

    if constexpr (recipe::fp32) {
        static_assert(qkt_subblock_h == 1, "Early identity scan handles one query tile row per QK subblock");
    } else {
        static_assert(
            qkt_subblock_h == 2 && qktv_subblock_h == 2,
            "Early guard pairs query tile rows (odd chunks end with one row)");
    }
    // FP32 recipes: UNPACK's early identity scan of each row's maxima.
    uint32_t sdpa_early_identity = !is_first_iter;
    auto scan_identity_row = [&](uint32_t row) {
        if constexpr (recipe::fp32) {
            UNPACK({
                if (sdpa_early_identity) {
                    CircularBuffer(prev.max).wait_front(row + 1);
                    CircularBuffer(cur.max).wait_front(row + 1);
                    auto* old_max =
                        reinterpret_cast<volatile uint32_t*>(get_tile_l1_byte_address(get_operand_id(prev.max), row));
                    auto* new_max =
                        reinterpret_cast<volatile uint32_t*>(get_tile_l1_byte_address(get_operand_id(cur.max), row));
                    for (uint32_t r = 0; r < 32; r += 4) {
                        constexpr uint32_t inf = 0x7f800000u;
                        const uint32_t offset = sdpa_max_word(r);
                        const uint32_t a0 = old_max[offset], a1 = old_max[offset + 16];
                        const uint32_t a2 = old_max[offset + 32], a3 = old_max[offset + 48];
                        const uint32_t diff = (a0 ^ new_max[offset]) | (a1 ^ new_max[offset + 16]) |
                                              (a2 ^ new_max[offset + 32]) | (a3 ^ new_max[offset + 48]);
                        if (diff || (a0 & inf) == inf || (a1 & inf) == inf || (a2 & inf) == inf || (a3 & inf) == inf) {
                            sdpa_early_identity = 0;
                            break;
                        }
                    }
                }
            })
        }
    };
    // BF16 recipes: per-K-step flags; only UNPACK fills these. Other threads obtain the
    // matching decision at the original correction mailbox rendezvous.
    uint32_t sdpa_identity_flags[Sq_chunk_t] = {};
    // Row sums target: the running state (FP32 recipes) or this chunk's sum CB (folded into l later).
    auto chunk_sum_cb = [&]() -> uint32_t {
        if constexpr (recipe::fp32) {
            return cur.sum;
        } else {
            return kRefMaxChunkSumCb;
        }
    };
    uint32_t pushed_rows = 0;
    // Q lives at [q_base_tiles, q_base_tiles + Sq_chunk_t*DHt) from the CB front. wait_front counts
    // from the front, so the wait target includes the chunks of earlier passes that stay resident.
    uint32_t q_wait_tiles = q_subblock_num_tiles;
    uint32_t q_index_offset = 0;
    uint32_t kt_index_offset = 0;

    exp_packthread_tile_init<true, scale_fp32, InputClamping::None>();
    if constexpr (recipe::fp32) {
        PACK((ckernel::sfpu::init_sdpa_refine_loadmacros()));
    }
    if constexpr (!recipe::fp32) {
        // Exp headroom: P = exp(scale * (s - m_ref) - tau). The fast exp computes y = 256 * log2(e) * scale * x +
        // (B - C) as INT16; lowering its constant (LREG13) by 256 per octave of tau moves saturation from ~0.72
        // to tau + 0.72 above m_ref while s - m_ref stays small (precise in BF16).
        PACK({
            constexpr float exp_c = 32500.818359375f - 256.0f * kRefMaxExpOctaves;
            constexpr uint32_t exp_c_bits = __builtin_bit_cast(uint32_t, exp_c);
            TTI_SFPLOADI(0, 0xA, exp_c_bits & 0xFFFF);
            TTI_SFPLOADI(0, 0x8, exp_c_bits >> 16);
            TTI_SFPCONFIG(0, 13, 0);
        })
    }
    if constexpr (recipe::fused_active) {
        if (!is_first_iter) {
            if constexpr (recipe::unpack_thread && recipe::ring) {
                fused_os::sdpa_fused_chunk<
                    Sq_chunk_t,
                    Sk_chunk_t,
                    DHt,
                    vDHt,
                    scale_fp32,
                    qkt_subblock_w,
                    qktv_subblock_w,
                    cb_q_in,
                    cb_kt_in,
                    cb_v_in,
                    cb_qkt_im,
                    cb_identity_scale_in,
                    cb_exp_max_diff,
                    cb_col_identity,
                    cb_recip_scratch,
                    cb_normalized_out,
                    independent_q_release>(prev, cur, is_last_iter, release_q);
            } else {
                fused_plain::sdpa_fused_chunk<
                    Sq_chunk_t,
                    Sk_chunk_t,
                    DHt,
                    vDHt,
                    scale_fp32,
                    qkt_subblock_w,
                    qktv_subblock_w,
                    cb_q_in,
                    cb_kt_in,
                    cb_v_in,
                    cb_qkt_im,
                    cb_identity_scale_in,
                    cb_exp_max_diff,
                    cb_col_identity,
                    cb_recip_scratch,
                    cb_normalized_out,
                    independent_q_release>(prev, cur, is_last_iter, release_q);
            }
            return true;
        }
    }

    // Use KT_stride for cb_qkt_im layout to keep CB pointers aligned across iterations
    CircularBuffer(cb_qkt_im).reserve_back(Sq_chunk_t * KT_stride);

    CircularBuffer(cur.sum).reserve_back(Sq_chunk_t);
    if constexpr (!recipe::fp32) {
        CircularBuffer(kRefMaxChunkSumCb).reserve_back(Sq_chunk_t);
    }
    // ========== PHASE 1: Q@KT directly into cb_qkt_im ==========
    // All matmul output goes to cb_qkt_im at absolute offsets via pack_tile<true>.
    // cb_push_back_hold_wr_ptr makes each row visible to UNPACK without advancing wr_ptr.
    CircularBuffer(cb_kt_in).wait_front(DHt * KT_stride);

    for (uint32_t q_subblock = 0; q_subblock < q_num_subblocks; q_subblock++) {
        const uint32_t cur_qk_h = qk_rows(q_subblock);
        const uint32_t cur_qk_index = qk_index(q_subblock);
        CircularBuffer(cb_q_in).wait_front(q_wait_tiles - (qkt_subblock_h - cur_qk_h) * in0_block_w);
        kt_index_offset = 0;

        sdpa_maybe_pack_reconfig_data_format<cb_normalized_out, cb_qkt_im>();
        if constexpr (recipe::fp32) {
            sdpa_stream_reconfig(cb_kt_in, cb_q_in);
        } else {
            sdpa_maybe_reconfig_data_format<cb_qkt_im, cb_kt_in, cb_identity_scale_in, cb_q_in>();
        }
        recipe_mm_init(cb_q_in, cb_kt_in, true, actual_sbw, cur_qk_h, in0_block_w);
        if constexpr (kBalancedQk) {
            MATH((llk_math_matmul_init_no_mop<kBalancedQkFidelity, MM_THROTTLE>(
                cb_q_in, cb_kt_in, true, actual_sbw, qkt_subblock_h)));
        }
        // Configure pack once before the kt loop for cb_qkt_im. Both sub_exp
        // and blocked_matmul_and_pack skip their internal configure (same cb+width).
        // sub_exp's configure_single_tile_pack(reduce_cb) clobbers the global to 1,
        // so blocked_matmul_and_pack must reconfigure when q_subblock > 0.
        // When q_subblock == 0, no sub_exp → global stays set → skip there too.
        configure_row_pack_width(cb_qkt_im, actual_sbw);

        // With an attn_mask, the first-half max reduce must not read scores before the mask lands on them.
        constexpr bool overlap_first_half = !recipe::mask && reduce_trigger;
        // PACK posts the first-half token after the subblock covering the last first-half column
        // [active_Sk/2 - 1]; committing a superset of [0, active_Sk/2) is safe (run()#1's cols are a subset).
        const uint32_t first_half_last_sb = (active_Sk / 2 - 1) / actual_sbw;

        {
            for (uint32_t kt_subblock = 0; kt_subblock < kt_num_full_subblocks; ++kt_subblock) {
                if (q_subblock > 0) {
                    uint32_t prev_q_subblock = q_subblock - 1;
                    if constexpr (!recipe::fp32) {
                        sdpa_maybe_reconfig_data_format<cb_kt_in, cb_qkt_im, cb_q_in, cb_qkt_im>();
                    }
                    sub_exp_block_bcast_cols<scale_fp32>(
                        cb_qkt_im,
                        cur.max,
                        chunk_sum_cb(),
                        KT_stride,
                        prev_q_subblock,
                        kt_subblock * actual_sbw,
                        qkt_subblock_h,
                        actual_sbw,
                        /*skip_pack_configure=*/true);
                    sdpa_maybe_pack_reconfig_data_format<cb_recip_scratch, cb_qkt_im>();
                    if constexpr (recipe::fp32) {
                        // sub_exp left the FP32 maxima's format on srcB: restore Q's.
                        sdpa_stream_reconfig(cb_kt_in, cb_q_in);
                    } else {
                        sdpa_maybe_reconfig_data_format<cb_qkt_im, cb_kt_in, cb_qkt_im, cb_q_in>();
                    }
                    recipe_mm_reinit(cb_q_in, cb_kt_in, true, actual_sbw, cur_qk_h, in0_block_w);
                    if constexpr (kBalancedQk) {
                        MATH((llk_math_matmul_reinit_no_mop<kBalancedQkFidelity, MM_THROTTLE>(
                            cb_q_in, cb_kt_in, true, actual_sbw, qkt_subblock_h)));
                    }
                }
                {
                    blocked_matmul_and_pack<true, KT_stride, KT_stride>(
                        cb_q_in,
                        cb_kt_in,
                        cb_qkt_im,
                        q_index_offset,
                        kt_index_offset,
                        cur_qk_index,
                        kt_subblock * actual_sbw,
                        actual_sbw,
                        cur_qk_h,
                        in0_block_w,
                        in0_block_w,
                        /*skip_pack_configure=*/recipe::fp32 ? true : q_subblock == 0);
                    if constexpr (recipe::fp32) {
                        // The previous row's max was already consumed by sub_exp
                        // above. Scan it after issuing current QK so scalar L1 loads
                        // can overlap the FPU, without a per-row cross-RISC mailbox.
                        if (q_subblock > 0 && kt_subblock == 0) {
                            scan_identity_row(q_subblock - 1);
                        }
                    }
                    // Signal the first half is committed so run()#1 overlaps the second-half
                    // pack. STALL_PACK drains the L1 writes before the token retires.
                    if (overlap_first_half && kt_subblock == first_half_last_sb) {
                        PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::UNPACK_MATH_DONE)));
                    }
                    if constexpr (!recipe::fp32) {
                        UNPACK({
                            if (!is_first_iter && q_subblock > 0 && kt_subblock == 0) {
                                sdpa_identity_flags[q_subblock - 1] =
                                    sdpa_scan_identity_maxima(prev.max, cur.max, q_subblock - 1, qkt_subblock_h);
                            }
                        })
                    }
                    kt_index_offset += actual_sbw;
                }
            }
        }
        // The FP32 max-reduce helper configures its own input views.
        if constexpr (!recipe::fp32) {
            sdpa_maybe_reconfig_data_format<cb_kt_in, cb_qkt_im, cb_q_in, cb_qkt_im>();
        }

        if constexpr (recipe::mask) {
            recipe_add_attn_mask<cb_qkt_im, KT_stride, qkt_subblock_h>(q_subblock * qkt_subblock_h, cur_qk_h);
        }

        // Push row (visible for UNPACK reads) but keep wr_ptr stable
        cb_push_back_hold_wr_ptr(cb_qkt_im, cur_qk_h * KT_stride);

        // reduce_trigger barrier. Posted after pack + push so it dominates every
        // cb_qkt_im writer; gates run()#2 (and run()#1 on the non-overlap path). STALL_PACK drains
        // the writes; one post / one uninit get stays balanced (wait_on_zero is non-consuming).
        if (reduce_trigger) {
            PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::FPU_SFPU)));
        }

        // Max reduce: reads from cb_qkt_im at q_subblock position
        {
            CircularBuffer(cur.max).reserve_back(cur_qk_h);
            configure_single_tile_pack(cur.max);
            // Use reduce_trigger to enable early reduce start (before all matmul output is ready).
            // When reduce_trigger=true, the packer signals the unpacker via semaphore after partial output.
            reduce_c_row_group<cb_qkt_im, cb_identity_scale_in, KT_stride, scale_fp32>(
                cur.max,
                prev.max,
                cur_qk_index,
                !is_first_iter /*do_eltwise_max*/,
                cur_qk_h,
                active_Sk,
                reduce_trigger,
                overlap_first_half);
            CircularBuffer(cur.max).push_back(cur_qk_h);
        }

        q_index_offset += qkt_subblock_h * in0_block_w;
        q_wait_tiles += q_subblock_num_tiles;
    }

    {
        CircularBuffer(cb_kt_in).pop_front(DHt * KT_stride);
    }

    // Release after the last local use, independently of final normalization
    // when a ring caller needs to reload Q for the next pass.
    {
        if (independent_q_release ? release_q : is_last_iter) {
            sdpa_cb_pop_front_out_of_line(cb_q_in, Sq_chunk_t * DHt);
        }
    }

    // ========== PHASE 2: Drain last row + QKT@V + SALAD ==========
    // After Phase 1: all rows are pushed (via hold_wr_ptr) in cb_qkt_im.
    // Rows 0..N-2 are softmax'd in-place; row N-1 has raw matmul output.
    {
        // FP32 recipes use single-row PV groups (their in-place numerator and row-wise FP32
        // state updates assume one row per group), including narrower D64 subblocks.
        constexpr uint32_t qktv_h = recipe::fp32 ? qktv_subblock_h
                                                 : ttnn::transformer::sdpa::streaming_qktv_h(
                                                       qktv_subblock_h, qktv_subblock_w, dst_size, Sq_chunk_t);
        constexpr uint32_t qktv_remainder_h = Sq_chunk_t % qktv_h;
        // QK and PV row groups coincide; an odd BF16 chunk ends with one single-row group.
        static_assert(qktv_h == qkt_subblock_h && qktv_remainder_h <= 1 && Sq_chunk_t >= 1);

        static_assert(vDHt % qktv_subblock_w == 0, "vDHt must be evenly divisible by qktv_subblock_w");
        static_assert(qktv_h * qktv_subblock_w <= dst_size, "qktv subblock must fit in dest register file");
        constexpr uint32_t qktv_q_num_subblocks = Sq_chunk_t / qktv_h + (qktv_remainder_h ? 1 : 0);
        constexpr uint32_t last_group = qktv_q_num_subblocks - 1;
        constexpr uint32_t last_h = qktv_remainder_h ? qktv_remainder_h : qktv_h;
        // Group g's row index in units of its own height.
        auto pv_rows = [](uint32_t group) -> uint32_t { return group == last_group ? last_h : qktv_h; };
        auto pv_index = [](uint32_t group) -> uint32_t {
            return group == last_group && last_h != qktv_h ? group * qktv_h : group;
        };
        constexpr uint32_t qktv_v_num_subblocks = vDHt / qktv_subblock_w;
        constexpr uint32_t qktv_output_num_tiles = Sq_chunk_t * vDHt * sdpa_out_stride;
        // cb_qkt_im row width is KT_stride (for pointer alignment), not Sk_chunk_t
        constexpr uint32_t qktv_in0_row_tiles = qktv_h * KT_stride;

        uint32_t qktv_in0_index_offset = 0;
        uint32_t qktv_in0_wait_tiles = qktv_in0_row_tiles;

        // FP32 recipes: this chunk accumulates onto the previous numerator in place when no row maximum changed.
        uint32_t inplace_numerator = !is_first_iter;
        if constexpr (recipe::fp32) {
            static_assert(qktv_h == 1);
            // FP32 recipes use single-row QK and PV groups, so odd Q chunks need no remainder group.
            if (inplace_numerator) {
                CircularBuffer(prev.max).wait_front(Sq_chunk_t);
                CircularBuffer(cur.max).wait_front(Sq_chunk_t);
                UNPACK({
                    scan_identity_row(Sq_chunk_t - 1);
                    inplace_numerator = sdpa_early_identity;
                    mailbox_write(ckernel::ThreadId::MathThreadId, inplace_numerator);
                    mailbox_write(ckernel::ThreadId::PackThreadId, inplace_numerator);
                })
                MATH(inplace_numerator = mailbox_read(ckernel::ThreadId::UnpackThreadId);)
                PACK(inplace_numerator = mailbox_read(ckernel::ThreadId::UnpackThreadId);)
            }
            if (inplace_numerator) {
                // Both physical banks retain their original 32-tile capacity.
                // The old bank is fully published. Pop it as a whole, preserving
                // its L1 bytes, then reuse it as this chunk's reserved output bank.
                // The alternate bank was empty and remains empty as prev.out.
                CircularBuffer(prev.out).wait_front(qktv_output_num_tiles);
                CircularBuffer(prev.out).pop_front(qktv_output_num_tiles);
                std::swap(prev.out, cur.out);
                // Phase 1's cur.sum reserve did not advance its write pointer
                // or publish tiles. It is safe to leave that empty bank unused.
                CircularBuffer(prev.sum).wait_front(Sq_chunk_t);
                CircularBuffer(prev.sum).pop_front(Sq_chunk_t);
                std::swap(prev.sum, cur.sum);
                CircularBuffer(cur.sum).reserve_back(Sq_chunk_t);
                sdpa_skip_prev_sum_pop = true;
            }
        }
        const uint32_t out_cb = cur.out;
        // BF16 recipes: groups whose reference max is unchanged (bitwise, UNPACK's Phase-1 scans), sent to MATH
        // and PACK.
        uint32_t unchanged_groups = 0;
        if constexpr (!recipe::fp32) {
            if (!is_first_iter) {
                // Groups 0..N-2 were scanned during Phase 1; the last group is scanned in the drain and sent
                // after it (below), so nothing waits for the last row's maximum here.
                if constexpr (q_num_subblocks == 1) {
                    CircularBuffer(prev.max).wait_front(Sq_chunk_t);
                    CircularBuffer(cur.max).wait_front(Sq_chunk_t);
                }
                UNPACK({
                    if constexpr (q_num_subblocks == 1) {
                        sdpa_identity_flags[0] = sdpa_scan_identity_maxima(prev.max, cur.max, qk_index(0), qk_rows(0));
                    }
                    const uint32_t scanned = q_num_subblocks == 1 ? 1 : q_num_subblocks - 1;
                    for (uint32_t g = 0; g < scanned; ++g) {
                        unchanged_groups |= (sdpa_identity_flags[g] ? 1u : 0u) << g;
                    }
                    mailbox_write(ckernel::ThreadId::MathThreadId, unchanged_groups);
                    mailbox_write(ckernel::ThreadId::PackThreadId, unchanged_groups);
                })
                MATH(unchanged_groups = mailbox_read(ckernel::ThreadId::UnpackThreadId);)
                PACK(unchanged_groups = mailbox_read(ckernel::ThreadId::UnpackThreadId);)
            }
        }
        auto group_unchanged = [&](uint32_t g) -> bool {
            if constexpr (!recipe::fp32) {
                return !is_first_iter && ((unchanged_groups >> g) & 1u);
            } else {
                return false;
            }
        };
        // Changed groups put this chunk's PV in plane 1 and fold it below; others accumulate in plane 0.
        auto pv_plane_offset = [&](uint32_t g) -> uint32_t {
            if constexpr (!recipe::fp32) {
                return (is_first_iter || group_unchanged(g)) ? 0 : vDHt;
            } else {
                return 0;
            }
        };
        [[maybe_unused]] auto pv_accumulates = [&](uint32_t g) -> bool {
            if constexpr (!recipe::fp32) {
                return group_unchanged(g);
            } else {
                return false;
            }
        };
        // Fold group g (rows Q tile rows) into the Float32 state: l (CB 13) from the chunk-local sums
        // (CB 12), and for changed groups O = O * c + PV (plane 1). In the last chunk, normalization pops
        // each group, so reads index from the current front. Group g starts at Q tile row g * qkt_subblock_h,
        // also for an odd chunk's single-row tail group.
        auto fold_group = [&](uint32_t g, uint32_t rows, bool first, bool ident) {
            if constexpr (!recipe::fp32) {
                constexpr uint32_t lsum_cb = kRefMaxChunkSumCb;
                const uint32_t read_row = is_last_iter ? 0 : g * qkt_subblock_h;
                CircularBuffer(lsum_cb).push_back(rows);
                CircularBuffer(lsum_cb).wait_front(rows);
                if (first || ident) {
                    reconfig_data_format_srca(lsum_cb);
                    copy_init(lsum_cb);
                    tile_regs_acquire();
                    for (uint32_t r = 0; r < rows; ++r) {
                        copy_tile(lsum_cb, r, r);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_reconfig_data_format(cur.sum);
                    configure_single_tile_pack(cur.sum);
                    PACK((llk_pack_reconfig_l1_acc(first ? 0 : 1)));
                    for (uint32_t r = 0; r < rows; ++r) {
                        pack_tile<true>(r, cur.sum, r);
                    }
                    PACK((llk_pack_reconfig_l1_acc(0)));
                    pack_reconfig_data_format(cb_qkt_im);
                    tile_regs_release();
                } else {
                    for (uint32_t r = 0; r < rows; ++r) {
                        tile_regs_acquire();
                        reconfig_data_format_srca(cur.sum);
                        copy_init(cur.sum);
                        copy_tile(cur.sum, read_row + r, 0);
                        reconfig_data_format_srca(lsum_cb);
                        copy_init(lsum_cb);
                        copy_tile(lsum_cb, r, 1);
                        unary_bcast_init<BroadcastType::COL>(cb_exp_max_diff);
                        unary_bcast<BroadcastType::COL>(cb_exp_max_diff, r, 2);
                        unary_bcast_uninit<BroadcastType::COL>(cb_exp_max_diff);
                        tile_regs_commit();
                        tile_regs_wait();
                        PACK((SFPU_UNARY_CALL(
                            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_rescale_add, (1), 0, VectorMode::None)));
                        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                        pack_reconfig_data_format(cur.sum);
                        configure_single_tile_pack(cur.sum);
                        PACK((llk_pack_reconfig_l1_acc(0)));
                        pack_tile<true>(0, cur.sum, r);
                        pack_reconfig_data_format(cb_qkt_im);
                        tile_regs_release();
                    }
                    for (uint32_t r = 0; r < rows; ++r) {
                        for (uint32_t j = 0; j < vDHt; j += 2) {
                            const bool pair = j + 1 < vDHt;
                            tile_regs_acquire();
                            reconfig_data_format_srca(out_cb);
                            copy_init(out_cb);
                            const uint32_t rbase = (read_row + r) * 2 * vDHt + j;
                            copy_tile(out_cb, rbase, 0);
                            if (pair) {
                                copy_tile(out_cb, rbase + 1, 1);
                            }
                            copy_tile(out_cb, rbase + vDHt, pair ? 2 : 1);
                            if (pair) {
                                copy_tile(out_cb, rbase + vDHt + 1, 3);
                            }
                            reconfig_data_format_srca(cb_exp_max_diff);
                            unary_bcast_init<BroadcastType::COL>(cb_exp_max_diff);
                            unary_bcast<BroadcastType::COL>(cb_exp_max_diff, r, pair ? 4 : 2);
                            unary_bcast_uninit<BroadcastType::COL>(cb_exp_max_diff);
                            tile_regs_commit();
                            tile_regs_wait();
                            if (pair) {
                                PACK((SFPU_UNARY_CALL(
                                    DST_SYNC_MODE,
                                    DST_ACCUM_MODE,
                                    calculate_sdpa_rescale_add,
                                    (2),
                                    0,
                                    VectorMode::None)));
                            } else {
                                PACK((SFPU_UNARY_CALL(
                                    DST_SYNC_MODE,
                                    DST_ACCUM_MODE,
                                    calculate_sdpa_rescale_add,
                                    (1),
                                    0,
                                    VectorMode::None)));
                            }
                            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                            pack_reconfig_data_format(out_cb);
                            configure_single_tile_pack(out_cb);
                            PACK((llk_pack_reconfig_l1_acc(0)));
                            const uint32_t wbase = r * 2 * vDHt + j;
                            pack_tile<true>(0, out_cb, wbase);
                            if (pair) {
                                pack_tile<true>(1, out_cb, wbase + 1);
                            }
                            pack_reconfig_data_format(cb_qkt_im);
                            tile_regs_release();
                        }
                    }
                }
                CircularBuffer(lsum_cb).pop_front(rows);
                reconfig_data_format_srca(cb_qkt_im);
            }
        };
        // The first chunk's sums seed l directly (no state to rescale).
        auto fold_first = [&](uint32_t rows) {
            if constexpr (!recipe::fp32) {
                if (is_first_iter) {
                    fold_group(0, rows, true, false);
                }
            }
        };

        // V wait deferred: don't block here. The sub_exp drain loop below
        // doesn't touch V, so the reader's V DMA can overlap with the drain.
        CircularBuffer(out_cb).reserve_back(qktv_output_num_tiles);

        // q_subblock 0: drain last row's sub_exp in-place + first QKT@V matmul
        {
            const uint32_t matmul_inner = actual_sbw;
            const uint32_t drain_subblocks = kt_num_full_subblocks;
            // Group 0 is short only when the whole chunk is one single-row group (a one-tile
            // BF16 chunk); cb_qkt_im then holds one row, not qktv_h.
            constexpr uint32_t first_h = last_group == 0 ? last_h : qktv_h;

            {
                // Interleave each column-subblock's sub_exp with its partial V matmul; partial products
                // accumulate across kt_sub via L1.
                for (uint32_t kt_sub = 0; kt_sub < drain_subblocks; ++kt_sub) {
                    sub_exp_block_bcast_cols<scale_fp32>(
                        cb_qkt_im,
                        cur.max,
                        chunk_sum_cb(),
                        KT_stride,
                        qk_index(q_num_subblocks - 1),
                        kt_sub * matmul_inner,
                        qk_rows(q_num_subblocks - 1),
                        matmul_inner);
                    if constexpr (q_num_subblocks == 1) {
                        // Phase 1 published no earlier row, so make the in-place exponential
                        // visible to the PV unpack explicitly.
                        recipe_pack_to_unpack_barrier();
                    }

                    if (kt_sub == 0) {
                        CircularBuffer(cb_qkt_im).wait_front(qktv_in0_wait_tiles - (qktv_h - first_h) * KT_stride);
                        CircularBuffer(cb_v_in).wait_front(Sk_chunk_t * vDHt);
                    }
                    if constexpr (recipe::fp32) {
                        if (kt_sub > 0 || inplace_numerator) {
                            PACK((llk_pack_reconfig_l1_acc(1)));
                        }
                    } else {
                        REF_MAX_PV_BEGIN(out_cb, kt_sub > 0 || pv_accumulates(0));
                    }
                    {
                        uint32_t v_index_offset = 0;
                        sdpa_maybe_reconfig_data_format<cb_normalized_out, cb_v_in, cb_normalized_out, cb_qkt_im>(
                            out_cb, out_cb);
                        // cb_qkt_im rows are laid out at KT_stride even when this kt_sub only consumes a
                        // narrower logical width. Keep unpack init on the physical stride; inner_dim below
                        // still limits how many V rows are multiplied.
                        recipe_mm_reinit(cb_qkt_im, cb_v_in, false, qktv_subblock_w, first_h, KT_stride);
                        configure_row_pack_width(out_cb, qktv_subblock_w);
                        for (uint32_t v_subblock = 0; v_subblock < qktv_v_num_subblocks; ++v_subblock) {
                            const uint32_t pv_inner_offset = kt_sub * matmul_inner;
                            const uint32_t qktv_in1_index = pv_inner_offset * vDHt + v_index_offset;
                            blocked_matmul_and_pack<false, vDHt, vDHt>(
                                cb_qkt_im,
                                cb_v_in,
                                out_cb,
                                qktv_in0_index_offset + pv_inner_offset,
                                qktv_in1_index,
                                0,
                                v_subblock * qktv_subblock_w + (recipe::fp32 ? 0 : pv_plane_offset(0)),
                                qktv_subblock_w,
                                first_h,
                                matmul_inner,
                                KT_stride,
                                /*skip_pack_configure=*/true);
                            if constexpr (!recipe::fp32) {
                                UNPACK({
                                    if (!is_first_iter && kt_sub == 0 && v_subblock == 0) {
                                        sdpa_identity_flags[q_num_subblocks - 1] = sdpa_scan_identity_maxima(
                                            prev.max,
                                            cur.max,
                                            qk_index(q_num_subblocks - 1),
                                            qk_rows(q_num_subblocks - 1));
                                    }
                                })
                            }
                            v_index_offset += qktv_subblock_w;
                        }
                        sdpa_maybe_reconfig_data_format<cb_v_in, cb_qkt_im, cb_qkt_im, cb_qkt_im>();
                    }

                    if constexpr (recipe::fp32) {
                        if (kt_sub > 0 || inplace_numerator) {
                            PACK((llk_pack_reconfig_l1_acc(0)));
                        }
                    } else {
                        REF_MAX_PV_END(cb_qkt_im);
                    }
                }
            }
            qktv_in0_index_offset += qktv_h * KT_stride;
            qktv_in0_wait_tiles += qktv_in0_row_tiles;
        }

        // PACK wrote the drain row's P in place; UNPACK must see it before the PV matmul.
        recipe_pack_to_unpack_barrier();
        if constexpr (!recipe::fp32 && q_num_subblocks > 1) {
            if (!is_first_iter) {
                UNPACK({
                    unchanged_groups |= (sdpa_identity_flags[q_num_subblocks - 1] ? 1u : 0u) << (q_num_subblocks - 1);
                    mailbox_write(ckernel::ThreadId::MathThreadId, unchanged_groups);
                    mailbox_write(ckernel::ThreadId::PackThreadId, unchanged_groups);
                })
                MATH(unchanged_groups = mailbox_read(ckernel::ThreadId::UnpackThreadId);)
                PACK(unchanged_groups = mailbox_read(ckernel::ThreadId::UnpackThreadId);)
            }
        }

        if constexpr (recipe::fp32) {
            // All P rows are committed by the drain barrier above. Use exactly the
            // effective HiFi2 weights in the denominator, without SFPU mantissa masking.
            {
                CircularBuffer(cb_col_identity).wait_front(1);
                sdpa_stream_reconfig(cb_col_identity, cb_qkt_im);
                matmul_block_init(cb_qkt_im, cb_col_identity, 0, 1, 4, KT_stride);
                constexpr auto denom_fidelity = recipe::accurate ? MathFidelity::HiFi2 : MathFidelity::LoFi;
                // Rows are batched four at a time (the FP32 half-sync dest capacity). A Q
                // chunk that is not a multiple of four finishes with a two-row batch;
                // batching only changes which rows share a dest pass, not any element's sum.
                auto denominator_init = [&](uint32_t rows) {
                    MATH((llk_math_matmul_init<denom_fidelity, MM_THROTTLE>(cb_qkt_im, cb_col_identity, 0, 1, rows)));
                    if constexpr (recipe::accurate) {
                        static_assert(
                            denom_fidelity == MathFidelity::HiFi2, "Phase-0/2 denominator requires the two-phase MOP");
                        // SrcA is exactly zero/one: its low mantissa is zero. Execute phases
                        // 0 and 2, not ordinary HiFi2's 0 and 1, to retain all SrcB bits.
                        MATH((addr_mod_t{
                            .srca = {.incr = 0, .clr = 1, .cr = 1},
                            .srcb = {.incr = 0, .clr = 1, .cr = 1},
                            .dest = {.incr = 0, .clr = 1, .cr = 1},
                            .fidelity = {.incr = 2, .clr = 0},
                        }
                                  .set(ADDR_MOD_5)));
                    }
                };
                denominator_init(4);
                configure_single_tile_pack(cur.sum);
                PACK((llk_pack_reconfig_l1_acc(inplace_numerator ? 1 : 0)));
                for (uint32_t row = 0; row < Sq_chunk_t; row += 4) {
                    const uint32_t rows = Sq_chunk_t - row < 4 ? Sq_chunk_t - row : 4;
                    if (rows != 4) {
                        UNPACK((llk_unpack_AB_matmul_init(cb_qkt_im, cb_col_identity, 0, 1, rows, KT_stride)));
                        denominator_init(rows);
                    }
                    tile_regs_acquire();
                    for (uint32_t col = 0; col < active_Sk; ++col) {
                        UNPACK((llk_unpack_AB_matmul(
                            cb_qkt_im, cb_col_identity, row * KT_stride + col, 0, 1, rows, KT_stride)));
                        MATH((llk_math_matmul<denom_fidelity, MM_THROTTLE>(0, 1, rows)));
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t j = 0; j < rows; ++j) {
                        pack_tile<true>(j, cur.sum, row + j);
                    }
                    tile_regs_release();
                }
                MATH((llk_math_matmul_init_no_mop<MATH_FIDELITY, MM_THROTTLE>(
                    cb_qkt_im, cb_v_in, false, qktv_subblock_w, qktv_h)));
            }
        }

        auto normalize_row = [&](uint32_t& pushed, uint32_t sbh) {
            fold_first(sbh);
            CircularBuffer(cur.sum).push_back(sbh);
            CircularBuffer(out_cb).push_back(sbh * vDHt * sdpa_out_stride);
            normalize_row_streaming<vDHt, dst_size, cb_col_identity, cb_recip_scratch, cb_normalized_out>(
                cur.sum, out_cb, sbh);

            pushed++;
        };

        bool identity_corrections[Sq_chunk_t] = {};

        // Correct the previous row; normalization is guarded separately.
        // prev.out is consumed row-by-row: always read from CB front, then pop after use.
        auto salad_correct_row = [&](uint32_t salad_row, uint32_t w_salad, uint32_t sbh) {
            if constexpr (!recipe::fp32) {
                CircularBuffer(cb_exp_max_diff).wait_front(sbh);
                fold_group(salad_row, sbh, false, group_unchanged(salad_row));
                CircularBuffer(cb_exp_max_diff).pop_front(sbh);
            } else {
                if (inplace_numerator) {
                    CircularBuffer(cb_exp_max_diff).wait_front(sbh);
                }
                if (inplace_numerator) {
                    // Preserve the existing correction CB publication/consumption
                    // handshake; only its unused arithmetic disappears.
                    CircularBuffer(cb_exp_max_diff).pop_front(sbh);
                    PACK((llk_pack_reconfig_l1_acc(0)));
                    return;
                }
                PACK((llk_pack_reconfig_l1_acc(1)));
                {
                    // ob_q_subblock=0: prev.out and cb_exp_max_diff are popped row-by-row (read from front).
                    // sum_q_subblock=salad_row: prev.sum uses cumulative indexing (not popped per-row).
                    salad_correct_fused<qktv_h, vDHt, dst_size>(
                        prev.out,
                        prev.sum,
                        cb_exp_max_diff,
                        out_cb,
                        cur.sum,
                        0,
                        salad_row,
                        w_salad,
                        is_last_iter,
                        identity_corrections[salad_row]);
                }
                CircularBuffer(cb_exp_max_diff).pop_front(sbh);
                if (!inplace_numerator) {
                    CircularBuffer(prev.out).pop_front(sbh * vDHt * sdpa_out_stride);
                }
                PACK((llk_pack_reconfig_l1_acc(0)));
            }
        };

        // Rows 1..N-1: correction of the previous row overlaps the current PV matmul.
        constexpr uint32_t total_v_row_groups = qktv_q_num_subblocks;
        // No score exp follows. This restores the generic SFPU state (config register, ADDR_MOD_7 with no
        // DST increment, counters) that the SFPI correction exp assumes: the BALANCED/ACCURATE refine
        // leaves ADDR_MOD_7 at a DST increment of 4, and the FP32 correction does not reset it itself.
        exp_packthread_tile_init<EXP_APPROX_MODE>();
        for (uint32_t q_subblock = 1; q_subblock < total_v_row_groups; ++q_subblock) {
            uint32_t cur_h = pv_rows(q_subblock);
            recipe_opaque_tail_rows<q_tail_row>(cur_h);
            uint32_t salad_row = q_subblock - 1;
            uint32_t w_salad = salad_row - pushed_rows;
            uint32_t w_q = q_subblock - pushed_rows;

            // SALAD for previous group (always a full group, h=qktv_h)
            if (!is_first_iter) {
                CircularBuffer(cb_exp_max_diff).reserve_back(qktv_h);
                if constexpr (recipe::fp32) {
                    identity_corrections[salad_row] =
                        inplace_numerator ||
                        sub_exp_first_col_blocks<scale_fp32>(prev.max, cur.max, cb_exp_max_diff, salad_row, qktv_h);
                } else {
                    identity_corrections[salad_row] = sub_exp_first_col_blocks<scale_fp32>(
                        prev.max, cur.max, cb_exp_max_diff, salad_row, qktv_h, sdpa_identity_flags[salad_row]);
                }
                CircularBuffer(cb_exp_max_diff).push_back(qktv_h);
            }

            // V matmul for the current row group.
            {
                CircularBuffer(cb_qkt_im).wait_front(qktv_in0_wait_tiles - (qktv_h - cur_h) * KT_stride);
            }
            {
                if constexpr (recipe::fp32) {
                    if (inplace_numerator) {
                        PACK((llk_pack_reconfig_l1_acc(1)));
                    }
                }
                if constexpr (!recipe::fp32) {
                    REF_MAX_PV_BEGIN(out_cb, pv_accumulates(q_subblock));
                }
                uint32_t v_index_offset = 0;
                sdpa_maybe_reconfig_data_format<cb_normalized_out, cb_v_in, cb_normalized_out, cb_qkt_im>(
                    out_cb, out_cb);
                // See the q_subblock-0 V matmul above: active_Sk can be narrower than the physical
                // cb_qkt_im row stride, but the unpacker is configured for the physical layout.
                recipe_mm_reinit(cb_qkt_im, cb_v_in, false, qktv_subblock_w, cur_h, KT_stride);
                // Configure once before v_subblock loop; skip inside.
                configure_row_pack_width(out_cb, qktv_subblock_w);
                for (uint32_t v_subblock = 0; v_subblock < qktv_v_num_subblocks; ++v_subblock) {
                    const uint32_t qktv_in1_index = v_index_offset;
                    blocked_matmul_and_pack<false, vDHt, vDHt>(
                        cb_qkt_im,
                        cb_v_in,
                        out_cb,
                        qktv_in0_index_offset,
                        qktv_in1_index,
                        cur_h == qktv_h ? w_q : w_q * qktv_h,
                        v_subblock * qktv_subblock_w + (recipe::fp32 ? 0 : pv_plane_offset(q_subblock)),
                        qktv_subblock_w,
                        cur_h,
                        active_Sk,
                        KT_stride,
                        /*skip_pack_configure=*/true);
                    v_index_offset += qktv_subblock_w;
                }
                sdpa_maybe_reconfig_data_format<cb_v_in, cb_qkt_im, cb_qkt_im, cb_qkt_im>();
                if constexpr (!recipe::fp32) {
                    REF_MAX_PV_END(cb_qkt_im);
                }
            }

            // SALAD corrections for previous group (always full, h=qktv_h) + row-by-row push
            if (!is_first_iter) {
                // Last main-loop iteration: hoist drain's sub_exp so both salads
                // (current row and drain row) chain back-to-back with one FPU init.
                if (q_subblock == total_v_row_groups - 1) {
                    constexpr uint32_t drain_h = last_h;
                    const uint32_t drain_salad_row = last_group;

                    CircularBuffer(cb_exp_max_diff).reserve_back(drain_h);
                    if constexpr (recipe::fp32) {
                        identity_corrections[drain_salad_row] =
                            inplace_numerator || sub_exp_first_col_blocks<scale_fp32>(
                                                     prev.max, cur.max, cb_exp_max_diff, drain_salad_row, drain_h);
                    } else {
                        identity_corrections[drain_salad_row] = sub_exp_first_col_blocks<scale_fp32>(
                            prev.max,
                            cur.max,
                            cb_exp_max_diff,
                            pv_index(drain_salad_row),
                            drain_h,
                            sdpa_identity_flags[drain_salad_row]);
                    }
                    CircularBuffer(cb_exp_max_diff).push_back(drain_h);

                    salad_correct_row(salad_row, w_salad, qktv_h);
                    if (is_last_iter) {
                        normalize_row(pushed_rows, qktv_h);
                    } else {
                        fold_first(qktv_h);
                        CircularBuffer(cur.sum).push_back(qktv_h);
                        CircularBuffer(out_cb).push_back(qktv_h * vDHt * sdpa_out_stride);
                        pushed_rows++;
                    }

                    const uint32_t drain_w = last_group - pushed_rows;
                    salad_correct_row(drain_salad_row, drain_w, drain_h);
                    if (is_last_iter) {
                        normalize_row(pushed_rows, drain_h);
                    } else {
                        fold_first(drain_h);
                        CircularBuffer(cur.sum).push_back(drain_h);
                        CircularBuffer(out_cb).push_back(drain_h * vDHt * sdpa_out_stride);
                        pushed_rows++;
                    }
                } else {
                    salad_correct_row(salad_row, w_salad, qktv_h);
                    if (is_last_iter) {
                        normalize_row(pushed_rows, qktv_h);
                    } else {
                        fold_first(qktv_h);
                        CircularBuffer(cur.sum).push_back(qktv_h);
                        CircularBuffer(out_cb).push_back(qktv_h * vDHt * sdpa_out_stride);
                        pushed_rows++;
                    }
                }
            } else if (is_last_iter) {
                normalize_row(pushed_rows, qktv_h);
            } else {
                fold_first(qktv_h);
                CircularBuffer(cur.sum).push_back(qktv_h);
                CircularBuffer(out_cb).push_back(qktv_h * vDHt * sdpa_out_stride);
                pushed_rows++;
            }

            qktv_in0_index_offset += cur_h * KT_stride;
            qktv_in0_wait_tiles += cur_h * KT_stride;
        }

        // Pipeline drain: SALAD for the last group
        {
            constexpr uint32_t drain_h = last_h;
            if constexpr (total_v_row_groups == 1) {
                // Single row group: the main loop never ran, so the drain performs the
                // full correction (as the legacy streaming kernel does).
                if (!is_first_iter) {
                    CircularBuffer(cb_exp_max_diff).reserve_back(drain_h);
                    if constexpr (recipe::fp32) {
                        identity_corrections[0] =
                            inplace_numerator || sub_exp_first_col_blocks<scale_fp32>(
                                                     prev.max, cur.max, cb_exp_max_diff, pv_index(0), drain_h);
                    } else {
                        identity_corrections[0] = sub_exp_first_col_blocks<scale_fp32>(
                            prev.max, cur.max, cb_exp_max_diff, pv_index(0), drain_h, sdpa_identity_flags[0]);
                    }
                    CircularBuffer(cb_exp_max_diff).push_back(drain_h);
                    salad_correct_row(0, 0, drain_h);
                }
                if (is_last_iter) {
                    normalize_row(pushed_rows, drain_h);
                } else {
                    fold_first(drain_h);
                    CircularBuffer(cur.sum).push_back(drain_h);
                    CircularBuffer(out_cb).push_back(drain_h * vDHt * sdpa_out_stride);
                    pushed_rows++;
                }
            } else {
                // Drain was hoisted into the last main-loop iteration above.
                // For is_first_iter (no SALAD), the drain row still needs push/normalize.
                if (is_first_iter) {
                    if (is_last_iter) {
                        normalize_row(pushed_rows, drain_h);
                    } else {
                        fold_first(drain_h);
                        CircularBuffer(cur.sum).push_back(drain_h);
                        CircularBuffer(out_cb).push_back(drain_h * vDHt * sdpa_out_stride);
                        pushed_rows++;
                    }
                }
            }
        }

        CircularBuffer(cb_v_in).pop_front(KT_stride * vDHt);
        CircularBuffer(cb_qkt_im).pop_front(Sq_chunk_t * KT_stride);
    }
    return false;
}

// One Q chunk over k_num_chunks K chunks: the online-softmax loop for recipes B-E. Called once per Q chunk
// by sdpa_standard_v2 (dense and joint). A caller may split one Q chunk's K range into several segments,
// setting final_segment on the last.
template <
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t DHt,
    uint32_t vDHt,
    uint32_t scale_fp32,
    uint32_t qkt_subblock_h,
    uint32_t qkt_subblock_w,
    uint32_t qktv_subblock_h,
    uint32_t qktv_subblock_w,
    uint32_t cb_q_in,
    uint32_t cb_kt_in,
    uint32_t cb_v_in,
    uint32_t cb_qkt_im,
    uint32_t cb_identity_scale_in,
    uint32_t cb_exp_max_diff,
    uint32_t cb_col_identity,
    uint32_t cb_recip_scratch,
    uint32_t cb_normalized_out,
    bool independent_q_release = false>
ALWI void sdpa_segment_v2(RecipeAccumulatorState& state, uint32_t k_num_chunks, bool final_segment, bool release_q) {
    static_assert(Sq_chunk_t >= 1 && Sq_chunk_t <= kRecipeMaxQTiles && DHt == vDHt && DHt >= 1);
    ASSERT(k_num_chunks > 0);
    auto& prev = state.prev;
    auto& cur = state.cur;
    // BF16 recipes: O stays in one bank for the whole Q chunk; only the maxima ping-pong.
    const uint32_t cb_out_im_A = prev.out;
    const uint32_t cb_out_im_B = cur.out;
    for (uint32_t k_chunk = 0; k_chunk < k_num_chunks; ++k_chunk) {
        if constexpr (recipe::k_tail) {
            recipe_k_tile_offset = (state.processed_chunks + k_chunk) * Sk_chunk_t;
        }
        const bool is_first = state.processed_chunks == 0 && k_chunk == 0;
        const bool last_local = k_chunk == k_num_chunks - 1;
        const bool is_last = final_segment && last_local;
        [[maybe_unused]] const bool fused = sdpa_inner_loop_step<
            Sq_chunk_t,
            Sk_chunk_t,
            DHt,
            vDHt,
            scale_fp32,
            qkt_subblock_h,
            qkt_subblock_w,
            qktv_subblock_h,
            qktv_subblock_w,
            cb_q_in,
            cb_kt_in,
            cb_v_in,
            cb_qkt_im,
            cb_identity_scale_in,
            cb_exp_max_diff,
            cb_col_identity,
            cb_recip_scratch,
            cb_normalized_out,
            independent_q_release>(prev, cur, is_last, is_first, release_q && last_local);
        if constexpr (recipe::fp32) {
            // prev.out and cb_exp_max_diff are already popped row-by-row inside salad_correct_row.
            if (!is_first) {
                sdpa_cb_pop_front_out_of_line(prev.max, Sq_chunk_t);
                if (!sdpa_skip_prev_sum_pop) {
                    sdpa_cb_pop_front_out_of_line(prev.sum, Sq_chunk_t);
                }
            }
        } else {
            // O and l each stay in one Float32 bank for the whole Q chunk: recycle the CB slots (the bytes
            // survive in L1 and the next chunk accumulates onto them).
            if (!is_last) {
                sdpa_cb_pop_front_out_of_line(cur.out, Sq_chunk_t * vDHt * sdpa_out_stride);
                sdpa_cb_pop_front_out_of_line(cur.sum, Sq_chunk_t);
            }
            // A fused chunk keeps the reference max in place in prev.max (no ping-pong).
            if (!is_first && !fused) {
                sdpa_cb_pop_front_out_of_line(prev.max, Sq_chunk_t);
            }
        }

        if (is_last) {
            if constexpr (recipe::fp32) {
                sdpa_cb_pop_front_out_of_line(cur.max, Sq_chunk_t);
            } else {
                sdpa_cb_pop_front_out_of_line(fused ? prev.max : cur.max, Sq_chunk_t);
            }
        } else {
            if constexpr (!recipe::fp32) {
                if (fused) {
                    continue;
                }
            }
            std::swap(prev, cur);
            if constexpr (!recipe::fp32) {
                prev.out = cb_out_im_A;
                cur.out = cb_out_im_B;
                std::swap(prev.sum, cur.sum);
            }
        }
    }
    state.processed_chunks += k_num_chunks;
}
