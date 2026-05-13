// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

namespace compute_kernel_lib {

namespace detail {

template <matmul_config::InitMode init_mode, typename Buf>
ALWI void MatmulSubblockStep::operator()(
    Buf& in0_buf,
    Buf& in1_buf,
    uint32_t in0_index,
    uint32_t in1_index,
    uint32_t dst_index,
    bool transpose,
    uint32_t ct_dim,
    uint32_t rt_dim,
    uint32_t kt_dim) const {

    static_assert(
        init_mode != matmul_config::InitMode::Full,
        "MatmulSubblockStep is a fine-grained primitive without an out_buf — Full init has no out_cb_id to bind. "
        "Use the kernel's outer mm_block_init for boot-time configuration; pass InitMode::Short here when the "
        "call site needs cheap restore between op kinds.");

    ASSERT(ct_dim * rt_dim <= compute_kernel_lib::DEST_AUTO_LIMIT);
    ASSERT(kt_dim > 0);

    const uint32_t in0_cb_id = buf_id(in0_buf);
    const uint32_t in1_cb_id = buf_id(in1_buf);

    if constexpr (init_mode == matmul_config::InitMode::Short) {
        mm_block_init_short(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim);
    }

#ifndef SKIP_COMPUTE
    ckernel::matmul_block(in0_cb_id, in1_cb_id, in0_index, in1_index, dst_index, transpose, ct_dim, rt_dim, kt_dim);
#else
    (void)in0_index;
    (void)in1_index;
    (void)dst_index;
    (void)transpose;
#endif
}

template <typename Buf, typename KStepFn>
ALWI void matmul_segmented_kloop(
    MatmulSubblockStep& step,
    Buf& in1_cb_buf,
    SegmentedKLoopShape shape,
    KStepFn& k_step) {

    ASSERT(shape.num_blocks > 0 || shape.last_block_tiles > 0);
    ASSERT(shape.tiles_per_block > 0);
    ASSERT(shape.ct_dim > 0);
    ASSERT(shape.last_block_tiles <= shape.tiles_per_block);

    for (uint32_t block_id = 0; block_id < shape.num_blocks; ++block_id) {
        in1_cb_buf.wait_front(shape.tiles_per_block);
        for (uint32_t k = 0; k < shape.tiles_per_block; k += shape.ct_dim) {
            k_step(step, shape, block_id, k);
        }
        in1_cb_buf.pop_front(shape.tiles_per_block);
    }

    if (shape.last_block_tiles > 0) {
        // Producer pushes a full tiles_per_block; FMA loop only iterates over
        // the populated prefix.
        in1_cb_buf.wait_front(shape.tiles_per_block);
        for (uint32_t k = 0; k < shape.last_block_tiles; k += shape.ct_dim) {
            k_step(step, shape, shape.num_blocks, k);
        }
        if (!shape.last_block_no_pop) {
            in1_cb_buf.pop_front(shape.tiles_per_block);
        }
    }
}

}  // namespace detail

template <typename Buf>
ALWI void KStepDefault<Buf>::operator()(
    detail::MatmulSubblockStep& step,
    SegmentedKLoopShape shape,
    uint32_t /*block_id*/,
    uint32_t k) {
    step(in0_buf, in1_buf, in0_index++, k, /*dst_index=*/0,
         transpose, shape.ct_dim, shape.rt_dim, shape.kt_dim);
}

template <typename Buf>
ALWI void KStepWithBias<Buf>::operator()(
    detail::MatmulSubblockStep& step,
    SegmentedKLoopShape shape,
    uint32_t /*block_id*/,
    uint32_t k) {
    if (k_tracker == bias_at) {
        // Bias FMA: ones × bias_row at the K position past the regular weights.
        step(bias_buf, in1_buf, /*in0_index=*/0, k, /*dst_index=*/0,
             transpose, shape.ct_dim, shape.rt_dim, shape.kt_dim);
        ++k_tracker;
        return;
    }
    if (k_tracker > bias_at) {
        // Padding K-slot past the bias position — no FMA, but advance k_tracker
        // so subsequent calls stay in sync with the producer's tile-per-block
        // count (ceiling-divided by the producer to round up to a txn boundary).
        ++k_tracker;
        return;
    }
    step(in0_buf, in1_buf, in0_index++, k, /*dst_index=*/0,
         transpose, shape.ct_dim, shape.rt_dim, shape.kt_dim);
    ++k_tracker;
}

template <bool HasBias, typename Buf, typename RingStepFn>
ALWI void KStepWithRing<HasBias, Buf, RingStepFn>::operator()(
    detail::MatmulSubblockStep& step,
    SegmentedKLoopShape shape,
    uint32_t /*block_id*/,
    uint32_t k) {
    if (k_tracker >= bias_at) {
        if constexpr (HasBias) {
            if (k_tracker == bias_at) {
                step(bias_buf, in1_buf, /*in0_index=*/0, k, /*dst_index=*/0,
                     transpose, shape.ct_dim, shape.rt_dim, shape.kt_dim);
            }
        }
        // Past bias (HasBias) or past the K limit (no bias): padding skip.
        // The bias FMA itself does not consume a ring slot — it reads from
        // bias_buf, not from the ring data buffer — so dm1_tiles_remaining
        // stays untouched.
        ++k_tracker;
        return;
    }
    // Regular FMA from the ring data buffer. If the active ring slot's tile
    // budget is exhausted, advance to the next slot via ring_step_fn — the
    // callback owns the in0_index advance convention (monotonic vs cyclic).
    if (dm1_tiles_remaining == 0) {
        ring_cb_buf.pop_front(1);
        ring_cb_buf.wait_front(1);
        ++dm1_step;
        const RingStepResult next = ring_step_fn(dm1_step);
        in0_index = next.in0_index;
        dm1_tiles_remaining = next.tiles_remaining;
    }
    --dm1_tiles_remaining;
    step(in0_buf, in1_buf, in0_index++, k, /*dst_index=*/0,
         transpose, shape.ct_dim, shape.rt_dim, shape.kt_dim);
    ++k_tracker;
}

template <typename Buf>
ALWI void SimplePack<Buf>::operator()() const {
    tile_regs_wait();
    const uint32_t out_cb_id = buf_id(out_buf);
    cb_reserve_back(out_cb_id, dst_count);
    pack_tile_block(0, out_cb_id, dst_count);
    cb_push_back(out_cb_id, dst_count);
}

template <typename Buf, typename KStepFn, typename PackBody, typename PostKBody>
ALWI void matmul_kloop_pack(
    Buf& in1_buf,
    SegmentedKLoopShape shape,
    KStepFn k_step,
    PackBody pack_body,
    PostKBody post_k_body) {

    detail::MatmulSubblockStep step;

    tile_regs_acquire();

    detail::matmul_segmented_kloop(step, in1_buf, shape, k_step);

    if constexpr (!std::is_same_v<std::decay_t<PostKBody>, NoOp>) {
        post_k_body();
    }

    tile_regs_commit();

    pack_body();

    tile_regs_release();
}

}  // namespace compute_kernel_lib
