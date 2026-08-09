// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The online-softmax fold of a sealed partial into the live stream:
//
//   m   = max(shift, live_scores)
//   r   = exp(shift - m)
//   lw  = exp(live_scores - m)
//   out = (partial * r + prefix_sum * lw) / (mass * r + lw)
//
// Everything but the last line is per-token-row work on a `[.., N, 1]` tile,
// so the divide folds into the row scalars — `a = r/den`, `b = lw/den` — and
// the full-width work collapses to
//
//   out[h][w] = partial[h][w] * a[h] + prefix_sum[h][w] * b[h]
//
// which is two column-broadcast multiplies accumulating into one dst tile.
// `init_bcast` plus a MATH init with acc_to_dest=1 makes every
// `mul_tiles_bcast_cols` a MAC, so the two terms never round-trip a CB.
//
// The scalar chain runs in dst via the SFPU: its binaries are dst-to-dst, so
// deriving `a` and `b` costs one pack of two tiles and no CB traffic. It is
// per-row work amortized over the row's Wt output tiles, and a core's tile run
// is contiguous, so most cores derive one row's scalars and never revisit them.

#include "api/compute/bcast.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

using namespace ckernel;

namespace {

constexpr auto cb_wide = tt::CBIndex::c_0;     // [partial, prefix_sum] per output tile
constexpr auto cb_row = tt::CBIndex::c_1;      // [a, b], produced here rather than read
constexpr auto cb_scalars = tt::CBIndex::c_2;  // [shift, mass, live_scores] per token row
constexpr auto cb_out = tt::CBIndex::c_16;

constexpr uint32_t kScalars = 3;
constexpr uint32_t kRowWeights = 2;
constexpr uint32_t kOperands = 2;
constexpr uint32_t onetile = 1;

constexpr uint32_t dst_a = 0;
constexpr uint32_t dst_b = 1;
constexpr uint32_t dst_den = 2;
constexpr uint32_t dst_max = 3;
constexpr uint32_t dst_acc = 0;

// The scalars live in column 0 of their tile — the only column BroadcastType::COL
// reads — so the rest of the tile is padding and its exp/reciprocal results are
// never packed into a full-width operand.
void derive_row_weights(CircularBuffer& cb_row_obj) {
    unary_op_init_common(cb_scalars, cb_row);

    tile_regs_acquire();
    copy_tile(cb_scalars, 0, dst_a);
    copy_tile(cb_scalars, 1, dst_den);
    copy_tile(cb_scalars, 2, dst_b);

    binary_max_tile_init();
    binary_max_tile(dst_a, dst_b, dst_max);

    sub_binary_tile_init();
    sub_binary_tile(dst_a, dst_max, dst_a);
    sub_binary_tile(dst_b, dst_max, dst_b);

    exp_tile_init();
    exp_tile(dst_a);
    exp_tile(dst_b);

    mul_binary_tile_init();
    mul_binary_tile(dst_den, dst_a, dst_den);

    add_binary_tile_init();
    add_binary_tile(dst_den, dst_b, dst_den);

    recip_tile_init();
    recip_tile(dst_den);

    mul_binary_tile_init();
    mul_binary_tile(dst_a, dst_den, dst_a);
    mul_binary_tile(dst_b, dst_den, dst_b);
    tile_regs_commit();

    cb_row_obj.reserve_back(kRowWeights);
    tile_regs_wait();
    pack_tile(dst_a, cb_row);
    pack_tile(dst_b, cb_row);
    tile_regs_release();
    cb_row_obj.push_back(kRowWeights);

    cb_row_obj.wait_front(kRowWeights);

    init_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(cb_wide, cb_row, cb_out);
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL, MATH_FIDELITY>(
        cb_wide, cb_row, 1 /*acc_to_dest*/)));
    reconfig_data_format(cb_wide, cb_row);
    pack_reconfig_data_format(cb_out);
}

}  // namespace

void kernel_main() {
    // compile-time args
    constexpr uint32_t Wt = get_compile_time_arg_val(0);

    // runtime args
    const uint32_t num_output_tiles = get_arg_val<uint32_t>(0);
    const uint32_t start_id = get_arg_val<uint32_t>(1);

    CircularBuffer cb_wide_obj(cb_wide);
    CircularBuffer cb_row_obj(cb_row);
    CircularBuffer cb_scalars_obj(cb_scalars);
    CircularBuffer cb_out_obj(cb_out);

    // The reader turns the scalar set over on the same test — `i % Wt == 0` over
    // the global tile index — so the two stay in step without a semaphore.
    uint32_t width_index = start_id % Wt;
    cb_scalars_obj.wait_front(kScalars);
    derive_row_weights(cb_row_obj);

    for (uint32_t i = 0; i < num_output_tiles; ++i) {
        if (i != 0 && width_index == 0) {
            cb_scalars_obj.pop_front(kScalars);
            cb_row_obj.pop_front(kRowWeights);
            cb_scalars_obj.wait_front(kScalars);
            derive_row_weights(cb_row_obj);
        }

        cb_wide_obj.wait_front(kOperands);
        tile_regs_acquire();
        mul_tiles_bcast_cols(cb_wide, cb_row, 0, 0, dst_acc);
        mul_tiles_bcast_cols(cb_wide, cb_row, 1, 1, dst_acc);
        tile_regs_commit();
        cb_wide_obj.pop_front(kOperands);

        cb_out_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_tile(dst_acc, cb_out);
        tile_regs_release();
        cb_out_obj.push_back(onetile);

        ++width_index;
        if (width_index == Wt) {
            width_index = 0;
        }
    }

    // Leave the CBs balanced: the last row's sets were waited on but never turned over.
    cb_scalars_obj.pop_front(kScalars);
    cb_row_obj.pop_front(kRowWeights);
}
