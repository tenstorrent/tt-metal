// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Implementation file for reduce_sfpu_helpers_compute.hpp.
// Do not include directly -- include reduce_sfpu_helpers_compute.hpp instead.

#include <cstdint>

#include "api/compute/binary_max_min.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"

#ifdef TRISC_PACK
#include "llk_pack_api.h"
#endif

namespace compute_kernel_lib {

namespace detail {

// Apply the packer reduce mask so non-result lanes are zeroed on pack.
// Mirrors what the FPU `reduce_init` does -- sfpu_reduce_init does NOT touch
// the packer, so we configure the mask once around the helper's main body.
template <ckernel::ReduceDim reduce_dim>
ALWI void sfpu_reduce_pack_mask_config() {
    PACK((llk_pack_reduce_mask_config<false /*untilize*/, reduce_dim>()));
}

ALWI void sfpu_reduce_pack_mask_clear() { PACK((llk_pack_reduce_mask_clear())); }

// Cross-tile fold primitives: pick MAX or MIN based on pool_type.  Format-aware
// dispatch is done at the helper level (Phase 1 = Int32 only).
//
// Note: `DataFormat` is at global scope (defined in tensix_types.h, no namespace),
// while `PoolType` lives in `namespace ckernel`.  Keep the spelling consistent with
// the LLK API (compute_kernel_api.h), which uses unqualified `DataFormat` inside
// `namespace ckernel`.
template <ckernel::PoolType pool_type, DataFormat format>
ALWI void sfpu_reduce_binary_fold_init() {
    static_assert(
        format == DataFormat::Int32,
        "Phase 1 of issue #43736 only implements DataFormat::Int32 -- other formats are reserved for "
        "follow-up phases.  Add the matching binary fold primitive when extending.");
    if constexpr (pool_type == ckernel::PoolType::MAX) {
        binary_max_int32_tile_init();
    } else {
        binary_min_int32_tile_init();
    }
}

template <ckernel::PoolType pool_type, DataFormat format>
ALWI void sfpu_reduce_binary_fold_tile(uint32_t a, uint32_t b, uint32_t out) {
    static_assert(
        format == DataFormat::Int32,
        "Phase 1 of issue #43736 only implements DataFormat::Int32 -- other formats are reserved for "
        "follow-up phases.  Add the matching binary fold primitive when extending.");
    if constexpr (pool_type == ckernel::PoolType::MAX) {
        binary_max_int32_tile(a, b, out);
    } else {
        binary_min_int32_tile(a, b, out);
    }
}

}  // namespace detail

// =============================================================================
// reduce_sfpu - main entry point
// =============================================================================
//
// Per-output algorithm (streaming, WaitAndPopPerTile):
//   (a) cross-tile fold via binary_*_int32_tile          <- binary init
//   (b) within-tile reduction via sfpu_reduce            <- reduce init
//
// The two SFPU primitives BOTH overwrite SFPCONFIG slots 4-5 (the LOADMACRO
// templates) and each leaves a different replay buffer at slot 0, so calling
// them in a fixed order at startup makes the second call clobber the first.
// We re-issue the matching init right before each of (a) and (b) inside the
// per-output loop body to stay correct.  (See the original reduce_int32.cpp
// commit message and #43736.)
//
// The binary fold init is skipped at compile time when there is only one
// input tile per output (Wt==1 for W-reduce, Ht==1 for H-reduce) since no
// fold is needed -- the seed tile is the result and we only need the
// within-tile sfpu_reduce.

template <ckernel::PoolType pool_type, ckernel::ReduceDim reduce_dim, DataFormat format>
ALWI void reduce_sfpu(
    uint32_t input_cb_id,
    uint32_t scaler_cb_id,
    uint32_t output_cb_id,
    ReduceInputBlockShape input_block_shape) {
    // =============================================================================
    // Static assertions (Phase 1 scope)
    // =============================================================================
    static_assert(
        pool_type == ckernel::PoolType::MAX || pool_type == ckernel::PoolType::MIN,
        "Phase 1 of issue #43736 only implements MAX and MIN.  SUM/AVG are reserved for follow-up "
        "phases (would also need an INT32 binary-add tile primitive for the cross-tile fold).");
    static_assert(
        reduce_dim == ckernel::ReduceDim::REDUCE_ROW || reduce_dim == ckernel::ReduceDim::REDUCE_COL,
        "reduce_sfpu only supports REDUCE_ROW (W axis) and REDUCE_COL (H axis); REDUCE_SCALAR is not "
        "needed for Phase 1 -- the W-then-H multi-core HW path goes through reduce_sfpu twice.");
    static_assert(
        format == DataFormat::Int32,
        "Phase 1 of issue #43736 only implements DataFormat::Int32.  UInt32/UInt16/Float32/Float16_b "
        "are reserved for follow-up phases (the SFPU LLK supports them but the binary fold and pad "
        "sentinel paths still need to be wired in).");
    // The LLK only implements REDUCE_ROW for SUM and MAX (see _calculate_reduce_max_min in
    // ckernel_sfpu_reduce.h).  INT32 MIN W-reduce must be rejected at the host level before we
    // reach this helper -- this static assert catches host bugs at kernel compile time.
    static_assert(
        !(pool_type == ckernel::PoolType::MIN && reduce_dim == ckernel::ReduceDim::REDUCE_ROW),
        "INT32 MIN with REDUCE_ROW (W axis) is not supported: sfpu_reduce<MIN, *, REDUCE_ROW> is "
        "not implemented in the LLK.  Host dispatch must gate this combination.");

    constexpr uint32_t onetile = 1;

    // Two DST registers used per output tile:
    //   acc_dst  - running max/min (initialised from the first input tile)
    //   work_dst - holds each subsequent input tile while binary_*_int32_tile
    //              folds it into acc_dst element-wise.
    constexpr uint32_t acc_dst = 0;
    constexpr uint32_t work_dst = 1;

    // ReduceInputBlockShape carries Ht / Wt / NC as runtime values, so the
    // "do we need a cross-tile binary fold?" check is also runtime here.  When
    // a future caller fixes Wt / Ht at compile time it can dispatch around the
    // fold via `if constexpr` from the call site.
    const uint32_t Ht = input_block_shape.rows;
    const uint32_t Wt = input_block_shape.cols;
    const uint32_t NC = input_block_shape.batches;

    const uint32_t tiles_per_output = (reduce_dim == ckernel::ReduceDim::REDUCE_ROW) ? Wt : Ht;
    const bool needs_cross_tile_fold = tiles_per_output > 1;

    // =============================================================================
    // SFPU pipeline init.  init_sfpu sets up unpacker + math for the SFPU
    // (datacopy + SFPU) path; copy_tile_to_dst_init_short configures the unpacker
    // for the per-tile copy_tile we use to seed/refill DST.
    // =============================================================================
    init_sfpu(input_cb_id, output_cb_id);
    copy_tile_to_dst_init_short(input_cb_id);

    // Drain the scaler tile that the reader prepares.  We don't actually use it
    // (sfpu_reduce takes no scaler) but the reader pushes it because the FPU
    // reduce path needs it; sharing the dataflow kernels keeps the host simple.
    cb_wait_front(scaler_cb_id, onetile);

    // Configure the packer reduce mask so non-result lanes are zeroed on every
    // pack_tile call below.  Held until the matching mask_clear at the end of
    // the helper.
    detail::sfpu_reduce_pack_mask_config<reduce_dim>();

    // =============================================================================
    // REDUCE_ROW: each row of Wt input tiles produces one output tile.
    //             Output count per batch = Ht.
    // REDUCE_COL: each column of Ht input tiles produces one output tile.
    //             Output count per batch = Wt.  Reader is configured with
    //             row_chunk=1 by the host so tiles arrive column-by-column.
    // =============================================================================
    const uint32_t outer_count = (reduce_dim == ckernel::ReduceDim::REDUCE_ROW) ? Ht : Wt;

    for (uint32_t nc = 0; nc < NC; ++nc) {
        for (uint32_t i = 0; i < outer_count; ++i) {
            tile_regs_acquire();

            // (a) Cross-tile fold along the reduce axis.  Re-issue the binary
            // init each output iteration because the trailing sfpu_reduce_init
            // from the previous iteration clobbered SFPCONFIG slots 4-5.
            if (needs_cross_tile_fold) {
                detail::sfpu_reduce_binary_fold_init<pool_type, format>();
            }

            // First reduce-axis tile seeds the accumulator.
            cb_wait_front(input_cb_id, onetile);
            copy_tile(input_cb_id, 0, acc_dst);
            cb_pop_front(input_cb_id, onetile);

            for (uint32_t k = 1; k < tiles_per_output; ++k) {
                cb_wait_front(input_cb_id, onetile);
                copy_tile(input_cb_id, 0, work_dst);
                detail::sfpu_reduce_binary_fold_tile<pool_type, format>(acc_dst, work_dst, acc_dst);
                cb_pop_front(input_cb_id, onetile);
            }

            // (b) Within-tile reduction: 32 cols/row -> col 0 (REDUCE_ROW), or
            // 32 rows/col -> row 0 (REDUCE_COL).  This re-init overwrites the
            // binary LOADMACRO templates with the reduce ones.
            sfpu_reduce_init<pool_type, format>();
            sfpu_reduce<pool_type, format, reduce_dim>(acc_dst, /*ct_dim=*/1, /*rt_dim=*/1);

            tile_regs_commit();

            cb_reserve_back(output_cb_id, onetile);
            tile_regs_wait();
            pack_tile(acc_dst, output_cb_id);
            tile_regs_release();
            cb_push_back(output_cb_id, onetile);
        }
    }

    detail::sfpu_reduce_pack_mask_clear();
}

}  // namespace compute_kernel_lib
