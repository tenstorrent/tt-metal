// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused per-head QK RMSNorm + RoPE after the all-gather matmul (FUSE_QK_NORM_ROPE).
//
// The matmul output columns are heads of CHUNK_HEAD_DT tiles in the order of CHUNK_HEAD_PERM
// (entry h = kind << 16 | index, kind 0 = q, 1 = k, 2 = v). For every q / k head the post-pass
// below runs the same sequence of LLK calls, operand formats and math fidelity as
// dit_rmsnorm_fused_compute.cpp's per-head path (per_head_norm, fuse_rope, broadcast RoPE), so the
// result is bit-identical to the matmul writing bf16 followed by that op:
//   0. fp32 matmul result -> bf16 (the copy the matmul does today)
//   1. x^2 accumulated over the head's tiles (packer L1 accumulate, fp32)
//   2. row SUM -> stat tile; 3. row AVG + eps + rsqrt on DEST
//   4. x * (1/rms) (bcast cols); 5. * gamma (bcast rows); 6. rotate (matmul with trans_mat)
//   7. x * cos + rotate(x) * sin (DEST accumulate) -> output format
// v heads take step 0 and, when the output is not bf16, a second copy into the output format (what
// a typecast of the bf16 matmul output does).
//
// The kernel is built at the norm's fidelity (HiFi4) and approx mode (off); the matmul itself keeps
// the caller's fidelity through the agmm_matmul_* helpers (same LLK calls as api/compute/matmul.h with
// MM_MATH_FIDELITY_VALUE as the fidelity).

#pragma once

#include <cstdint>

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

#ifndef MM_MATH_FIDELITY_VALUE
#error "FUSE_QK_NORM_ROPE needs MM_MATH_FIDELITY_VALUE (the matmul's own math fidelity)"
#endif
#if !defined(CHUNK_HEAD_PERM) || !defined(CHUNK_HEAD_DT)
#error "FUSE_QK_NORM_ROPE needs CHUNK_HEAD_PERM and CHUNK_HEAD_DT"
#endif

namespace qk_norm_rope {

#ifdef TRISC_MATH
// Only used inside MATH((...)), like MATH_FIDELITY in api/compute/matmul.h.
constexpr ckernel::MathFidelity MM_FIDELITY = static_cast<ckernel::MathFidelity>(MM_MATH_FIDELITY_VALUE);
#endif

// CB ids (the factory creates them only when fused).
constexpr uint32_t staging_cb = tt::CBIndex::c_8;          // bf16, step 0 output
constexpr uint32_t pre_intermediate_cb = tt::CBIndex::c_9;  // fp32, x^2 accumulator
constexpr uint32_t stats_cb = tt::CBIndex::c_10;            // fp32, per-head SUM
constexpr uint32_t reduce_result_cb = tt::CBIndex::c_11;    // fp32, 1/rms
constexpr uint32_t reduce_scalar_sum_cb = tt::CBIndex::c_12;
constexpr uint32_t reduce_scalar_avg_cb = tt::CBIndex::c_13;
constexpr uint32_t epsilon_cb = tt::CBIndex::c_14;  // made by the shared scalar setup; the math uses eps bits
constexpr uint32_t transformation_mat_cb = tt::CBIndex::c_15;
constexpr uint32_t gamma_cb = tt::CBIndex::c_16;         // this core's q / k heads, column order
constexpr uint32_t rope_cos_cb = tt::CBIndex::c_17;      // M_block rows x DT tiles
constexpr uint32_t rope_sin_cb = tt::CBIndex::c_18;
constexpr uint32_t norm_intermediate_cb = tt::CBIndex::c_19;  // fp32 (mul_rms / mul_weight result)
constexpr uint32_t rotated_input_cb = tt::CBIndex::c_20;      // fp32

constexpr uint32_t DT = CHUNK_HEAD_DT;
constexpr uint32_t KIND_V = 2;
constexpr uint32_t head_perm[] = {CHUNK_HEAD_PERM};
constexpr uint32_t num_perm_heads = sizeof(head_perm) / sizeof(head_perm[0]);

FORCE_INLINE uint32_t head_kind(uint32_t col) {
    const uint32_t h = col / DT;
    return h < num_perm_heads ? (head_perm[h] >> 16) : KIND_V;  // padding columns: copied like v
}

// q / k heads that start in [col_begin, col_end).
FORCE_INLINE uint32_t qk_heads_in(uint32_t col_begin, uint32_t col_end) {
    uint32_t count = 0;
    for (uint32_t c = col_begin; c < col_end; c++) {
        if (c % DT == 0 && head_kind(c) != KIND_V) {
            count++;
        }
    }
    return count;
}

// ---- The matmul at its own fidelity (api/compute/matmul.h with MM_FIDELITY) ----
ALWI void agmm_matmul_init(uint32_t in0_cb_id, uint32_t in1_cb_id, const uint32_t transpose = 0) {
    state_configure(in1_cb_id, in0_cb_id, __builtin_LINE());
    MATH((llk_math_matmul_init<MM_FIDELITY, MM_THROTTLE>(in0_cb_id, in1_cb_id, transpose)));
    UNPACK((llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose)));
}

ALWI void agmm_matmul_block_init(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    const uint32_t transpose,
    uint32_t ct_dim,
    uint32_t rt_dim,
    uint32_t kt_dim) {
    state_configure(in1_cb_id, in0_cb_id, __builtin_LINE());
    UNPACK((llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim)));
    MATH((llk_math_matmul_init<MM_FIDELITY, MM_THROTTLE>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim)));
#ifdef ARCH_BLACKHOLE
    MATH((throttled_mop_status = 0));
#endif
}

#ifdef ARCH_BLACKHOLE
ALWI void agmm_matmul_block_math_dynamic_throttle(
    uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t idst, const uint32_t transpose, uint32_t ct_dim, uint32_t rt_dim) {
    volatile std::uint32_t mm_throttle_en = *(throttle_ptr) % 2;
    if (mm_throttle_en) {
        if (throttled_mop_status != 1) {
            MATH((llk_math_matmul_init<MM_FIDELITY, MM_THROTTLE_MAX>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim)));
            throttled_mop_status = 1;
        }
        MATH((llk_math_matmul<MM_FIDELITY, MM_THROTTLE_MAX>(idst, ct_dim, rt_dim)));
    } else {
        if (throttled_mop_status != 0) {
            MATH((llk_math_matmul_init<MM_FIDELITY, MM_THROTTLE>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim)));
            throttled_mop_status = 0;
        }
        MATH((llk_math_matmul<MM_FIDELITY, MM_THROTTLE>(idst, ct_dim, rt_dim)));
    }
}
#endif

ALWI void agmm_matmul_block(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t in0_tile_index,
    uint32_t in1_tile_index,
    uint32_t idst,
    const uint32_t transpose,
    uint32_t ct_dim,
    uint32_t rt_dim,
    uint32_t kt_dim) {
    state_configure(in1_cb_id, in0_cb_id, __builtin_LINE());
    UNPACK((llk_unpack_AB_matmul(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, ct_dim, rt_dim, kt_dim)));
#ifdef ARCH_BLACKHOLE
    MATH((agmm_matmul_block_math_dynamic_throttle(in0_cb_id, in1_cb_id, idst, transpose, ct_dim, rt_dim)));
#else
    MATH((llk_math_matmul<MM_FIDELITY, MM_THROTTLE>(idst, ct_dim, rt_dim)));
#endif
}

// Once per kernel: the singletons the writer pushes (scalers, eps, trans_mat) and the resident gamma tiles.
FORCE_INLINE void wait_resident_inputs(uint32_t num_gamma_tiles) {
    CircularBuffer(reduce_scalar_sum_cb).wait_front(1);
    CircularBuffer(reduce_scalar_avg_cb).wait_front(1);
    CircularBuffer(epsilon_cb).wait_front(1);
    CircularBuffer(transformation_mat_cb).wait_front(1);
    if (num_gamma_tiles > 0) {
        CircularBuffer(gamma_cb).wait_front(num_gamma_tiles);
    }
}

// Step 0 for one tile: fp32 matmul result -> bf16 staging (copy_block's calls, one tile).
FORCE_INLINE void stage_tile(uint32_t interm_cb_id, uint32_t interm_idx) {
    CircularBuffer cb_staging(staging_cb);
    tile_regs_acquire();
    copy_tile(interm_cb_id, interm_idx, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_staging.reserve_back(1);
    pack_tile(0, staging_cb);
    cb_staging.push_back(1);
    tile_regs_release();
}

// v / padding tiles of one row: step 0 straight into the output when it is bf16; otherwise step 0 into
// staging for all of the row's v tiles, then the typecast copy (bf16 -> output format) for each.
template <uint32_t N_block_tiles>
FORCE_INLINE void copy_v_tiles(uint32_t interm_cb_id, uint32_t out_cb_id, uint32_t row_base, uint32_t n_tile) {
#ifdef QK_NORM_OUT_IS_BF16
    copy_init(interm_cb_id);
    reconfig_data_format_srca(interm_cb_id);
    pack_reconfig_data_format(out_cb_id);
    for (uint32_t c = 0; c < N_block_tiles; c++) {
        if (head_kind(n_tile + c) != KIND_V) {
            continue;
        }
        tile_regs_acquire();
        copy_tile(interm_cb_id, row_base + c, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, out_cb_id, c);
        tile_regs_release();
    }
#else
    CircularBuffer cb_staging(staging_cb);
    copy_init(interm_cb_id);
    reconfig_data_format_srca(interm_cb_id);
    pack_reconfig_data_format(staging_cb);
    uint32_t staged = 0;
    for (uint32_t c = 0; c < N_block_tiles; c++) {
        if (head_kind(n_tile + c) != KIND_V) {
            continue;
        }
        stage_tile(interm_cb_id, row_base + c);
        staged++;
    }
    if (staged == 0) {
        return;
    }
    copy_init(staging_cb);
    reconfig_data_format_srca(staging_cb);
    pack_reconfig_data_format(out_cb_id);
    cb_staging.wait_front(staged);
    uint32_t s = 0;
    for (uint32_t c = 0; c < N_block_tiles; c++) {
        if (head_kind(n_tile + c) != KIND_V) {
            continue;
        }
        tile_regs_acquire();
        copy_tile(staging_cb, s, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, out_cb_id, c);
        tile_regs_release();
        s++;
    }
    cb_staging.pop_front(staged);
#endif
}

// One q / k head of one row: steps 0-7, output packed at out-block column c .. c + DT - 1.
// gamma_base: this head's first tile in gamma_cb; rope_base: this row's first tile in the cos / sin CBs.
template <uint32_t eps_bits>
FORCE_INLINE void norm_rope_head(
    uint32_t interm_cb_id, uint32_t out_cb_id, uint32_t interm_base, uint32_t c, uint32_t gamma_base, uint32_t rope_base) {
    static_assert(DT <= 4, "one head's tiles must fit one fp32 half-sync DEST (the norm's block_size)");
    CircularBuffer cb_staging(staging_cb);
    CircularBuffer cb_pre_intermediate(pre_intermediate_cb);
    CircularBuffer cb_reduce_result(reduce_result_cb);
    CircularBuffer cb_intermediate(norm_intermediate_cb);
    CircularBuffer cb_rotated_input(rotated_input_cb);

    // ---- 0. matmul result -> bf16 (the norm op's input) ----
    copy_init(interm_cb_id);
    reconfig_data_format_srca(interm_cb_id);
    pack_reconfig_data_format(staging_cb);
    for (uint32_t i = 0; i < DT; i++) {
        stage_tile(interm_cb_id, interm_base + c + i);
    }
    cb_staging.wait_front(DT);

    // ---- 1. sum(x^2) over the head (dit_rmsnorm_fused_compute.cpp PHASE 1, per_head_norm group) ----
    reconfig_data_format(staging_cb, staging_cb);
    pack_reconfig_data_format(pre_intermediate_cb);
    PACK((llk_pack_reconfig_l1_acc(0)));
    mul_init(staging_cb, staging_cb);
    cb_pre_intermediate.reserve_back(1);
    tile_regs_acquire();
    for (uint32_t i = 0; i < DT; i++) {
        mul_tiles(staging_cb, staging_cb, i, i, i);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < DT; i++) {
        pack_tile<true>(i, pre_intermediate_cb, 0);
        if (i == 0) {
            PACK((llk_pack_reconfig_l1_acc(1)));
        }
    }
    tile_regs_release();
    cb_pre_intermediate.push_back(1);
    PACK((llk_pack_reconfig_l1_acc(0)));

    // ---- 2. row SUM -> stat tile ----
    compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, pre_intermediate_cb, reduce_scalar_sum_cb, stats_cb>(
        compute_kernel_lib::ReduceInputBlockShape::single());

    // ---- 3. row AVG + eps + rsqrt on DEST ----
    auto eps_rsqrt = [](uint32_t dst_idx) {
        binop_with_scalar_tile_init();
        add_unary_tile(dst_idx, eps_bits);
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    };
    compute_kernel_lib::reduce<PoolType::AVG, ReduceDim::REDUCE_ROW, stats_cb, reduce_scalar_avg_cb, reduce_result_cb>(
        compute_kernel_lib::ReduceInputBlockShape::single(),
        compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
        compute_kernel_lib::NoAccumulation{},
        eps_rsqrt);
    cb_reduce_result.wait_front(1);

    // ---- 4. x * (1/rms) -> norm intermediate ----
    reconfig_data_format(staging_cb, reduce_result_cb);
    pack_reconfig_data_format(norm_intermediate_cb);
    mul_bcast_cols_init(staging_cb, reduce_result_cb);
    cb_intermediate.reserve_back(DT);
    tile_regs_acquire();
    for (uint32_t i = 0; i < DT; i++) {
        mul_tiles_bcast_cols(staging_cb, reduce_result_cb, i, 0, i);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < DT; i++) {
        pack_tile(i, norm_intermediate_cb);
    }
    tile_regs_release();
    cb_intermediate.push_back(DT);
    cb_reduce_result.pop_front(1);
    cb_staging.pop_front(DT);

    // ---- 5. * gamma (row broadcast) ----
    reconfig_data_format(norm_intermediate_cb, gamma_cb);
    pack_reconfig_data_format(norm_intermediate_cb);
    mul_bcast_rows_init(norm_intermediate_cb, gamma_cb);
    cb_intermediate.wait_front(DT);
    tile_regs_acquire();
    for (uint32_t i = 0; i < DT; i++) {
        mul_tiles_bcast_rows(norm_intermediate_cb, gamma_cb, i, gamma_base + i, i);
    }
    tile_regs_commit();
    cb_intermediate.pop_front(DT);
    cb_intermediate.reserve_back(DT);
    tile_regs_wait();
    for (uint32_t i = 0; i < DT; i++) {
        pack_tile(i, norm_intermediate_cb);
    }
    tile_regs_release();
    cb_intermediate.push_back(DT);

    // ---- 6. rotate: matmul(intermediate, trans_mat) at the kernel fidelity (the norm's) ----
    reconfig_data_format(transformation_mat_cb, norm_intermediate_cb);
    pack_reconfig_data_format(rotated_input_cb);
    matmul_block_init(norm_intermediate_cb, transformation_mat_cb, /*transpose=*/0, /*ct_dim=*/1, /*rt_dim=*/DT, /*kt_dim=*/1);
    cb_intermediate.wait_front(DT);
    cb_rotated_input.reserve_back(DT);
    tile_regs_acquire();
    matmul_block(
        norm_intermediate_cb,
        transformation_mat_cb,
        /*in0_idx=*/0,
        /*in1_idx=*/0,
        /*idst=*/0,
        /*transpose=*/0,
        /*ct_dim=*/1,
        /*rt_dim=*/DT,
        /*kt_dim=*/1);
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < DT; i++) {
        pack_tile(i, rotated_input_cb);
    }
    tile_regs_release();
    cb_rotated_input.push_back(DT);

    // ---- 7. x * cos + rotate(x) * sin (DEST accumulate) -> output ----
    cb_rotated_input.wait_front(DT);
    reconfig_data_format(norm_intermediate_cb, rope_cos_cb);
    pack_reconfig_data_format(out_cb_id);
    tile_regs_acquire();
    binary_tiles_init<true, EltwiseBinaryType::ELWMUL>(norm_intermediate_cb, rope_cos_cb, false);
    for (uint32_t i = 0; i < DT; i++) {
        mul_tiles(norm_intermediate_cb, rope_cos_cb, i, rope_base + i, i);
    }
    binary_tiles_init<false, EltwiseBinaryType::ELWMUL>(rotated_input_cb, rope_sin_cb, true);
    for (uint32_t i = 0; i < DT; i++) {
        mul_tiles(rotated_input_cb, rope_sin_cb, i, rope_base + i, i);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < DT; i++) {
        pack_tile<true>(i, out_cb_id, c + i);
    }
    tile_regs_release();
    cb_intermediate.pop_front(DT);
    cb_rotated_input.pop_front(DT);
}

// The post-pass of one (M block, N block): replaces copy_block. interm holds M_block x N_block fp32 tiles
// (row stride N_block_tiles); out_cb has the whole block reserved; one push per row, as copy_block does.
// gamma_slot0: gamma tile slot of the first q / k head of this N block.
template <uint32_t M_block_tiles, uint32_t N_block_tiles, uint32_t eps_bits>
void norm_rope_block(CircularBuffer& interm_cb, CircularBuffer& out_cb, uint32_t n_tile, uint32_t gamma_slot0) {
    CircularBuffer cb_cos(rope_cos_cb);
    CircularBuffer cb_sin(rope_sin_cb);
    cb_cos.wait_front(M_block_tiles * DT);
    cb_sin.wait_front(M_block_tiles * DT);
    const uint32_t interm_id = interm_cb.get_cb_id();
    const uint32_t out_id = out_cb.get_cb_id();
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        const uint32_t row_base = m * N_block_tiles;
        copy_v_tiles<N_block_tiles>(interm_id, out_id, row_base, n_tile);
        uint32_t gamma_slot = gamma_slot0;
        for (uint32_t c = 0; c < N_block_tiles; c++) {
            const uint32_t col = n_tile + c;
            if (col % DT != 0 || head_kind(col) == KIND_V) {
                continue;
            }
            norm_rope_head<eps_bits>(interm_id, out_id, row_base, c, gamma_slot * DT, m * DT);
            gamma_slot++;
        }
        out_cb.push_back(N_block_tiles);
    }
    cb_cos.pop_front(M_block_tiles * DT);
    cb_sin.pop_front(M_block_tiles * DT);
}

}  // namespace qk_norm_rope
