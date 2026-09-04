// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// One site's whole read path: this rank's share of the scoring statistics, the
// cross-rank gather that completes them, and the online-softmax fold that turns
// them into `h`. As three separate dispatches — statistics, collective, fold —
// the walk pays roughly 380 us of host dispatch per site for about 26 ms of
// device work over the whole schedule. One program is a host-side change that
// happens to also save the statistics a round trip through DRAM.
//
// The seam is exact and is why this fuses at all. The statistics reduce emits
// per token row a sum of squares then a dot; the fold wants, per token row,
// `shift`, `mass`, then a (sum of squares, dot) pair per rank in rank order.
// So the gather is not a general collective over a tensor — it is a fixed
// 2-tile exchange between the cores that hold the same token rows on each rank.
//
// Two passes, and the grid is divided differently for each. Pass one reduces a token
// row along the whole of its width, so a row belongs entirely to one core and the
// pass cannot use more than Ht of them. Pass two has no such tie — every output tile
// is independent once the row weights exist — so it is split by tile across the rest
// of the grid, and a core's fold run is generally not the rows it reduced.
//
// The two are separated by a barrier and not merely by data: the exchange is funnelled
// through a single fabric-holding core that cannot send until every statistics core
// has parked every row, so no tile can fold before all of them have been produced.
// `running_sum` is therefore read twice, once per pass, over different rows.

#include "api/compute/bcast.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

using namespace ckernel;

namespace {

constexpr auto cb_partial = tt::CBIndex::c_0;      // fold operand, one row at a time
constexpr auto cb_scaler = tt::CBIndex::c_1;       // reduce scaler, resident
constexpr auto cb_q = tt::CBIndex::c_2;            // query row, resident
constexpr auto cb_prefix = tt::CBIndex::c_3;       // the live stream: reduce input, then fold operand
constexpr auto cb_row = tt::CBIndex::c_4;          // [a, b], produced here rather than read
constexpr auto cb_scalars = tt::CBIndex::c_5;      // [shift, mass, gathered statistics] per token row
constexpr auto cb_tmp = tt::CBIndex::c_6;          // transformed row, drained by the reduce
constexpr auto cb_local_stats = tt::CBIndex::c_7;  // this rank's pair, handed to the sender
constexpr auto cb_pending = tt::CBIndex::c_10;     // a residual write not yet in the stream
constexpr auto cb_total = tt::CBIndex::c_11;       // the settled stream, reduced here
constexpr auto cb_total_out = tt::CBIndex::c_12;   // the same row again, for the writer
constexpr auto cb_out = tt::CBIndex::c_16;

constexpr uint32_t Wt = get_compile_time_arg_val(0);
// Tensor-parallel ranks over the gather axis; also the number of statistic pairs
// `derive_row_weights` sums. fp32 bit patterns follow, which is what the SFPU
// scalar binops take.
constexpr uint32_t num_partials = get_compile_time_arg_val(1);
constexpr uint32_t inv_hidden_size = get_compile_time_arg_val(2);
constexpr uint32_t eps = get_compile_time_arg_val(3);
// Whether a deferred residual write is settled into the stream on the way through.
constexpr bool fuse_add = get_compile_time_arg_val(4) == 1;

// What the statistics are taken over. A sum of squares does not distribute across the
// two addends, so where there is a write to settle it has to be summed for real first;
// the fold in pass two has no such tie and distributes instead.
constexpr auto cb_stream = fuse_add ? cb_total : cb_prefix;

constexpr uint32_t kFixedScalars = 2;
constexpr uint32_t kStatsPerPartial = 2;
constexpr uint32_t kScalars = kFixedScalars + kStatsPerPartial * num_partials;
constexpr uint32_t kRowWeights = 2;
constexpr uint32_t onetile = 1;

constexpr uint32_t dst_a = 0;
constexpr uint32_t dst_b = 1;
constexpr uint32_t dst_den = 2;
constexpr uint32_t dst_max = 3;
constexpr uint32_t dst_acc = 0;

// The reduce drains `cb_tmp`, so the same buffer carries both transforms of the
// row: fill it, reduce it, fill it again. `init` re-establishes the unpack and
// pack configuration the previous reduce left changed.
template <typename Init, typename TransformOne>
ALWI void reduce_transformed_row(DataflowBuffer& tmp_buf, Init init, TransformOne transform_one) {
    init();
    tmp_buf.reserve_back(Wt);
    for (uint32_t wt = 0; wt < Wt; ++wt) {
        tile_regs_acquire();
        transform_one(wt);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_tmp, wt);
        tile_regs_release();
    }
    tmp_buf.push_back(Wt);

    compute_kernel_lib::reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        cb_tmp,
        cb_scaler,
        cb_local_stats,
        compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(compute_kernel_lib::ReduceInputBlockShape::row(Wt));
}

// Settle a deferred residual write into the row the statistics are about to be taken
// over. The row is packed twice out of one dest register: a circular buffer has a single
// consumer, and the sum is wanted both by the two reduces here and by the writer that
// parks it as the stream the caller carries forward.
ALWI void settle_row(
    DataflowBuffer& prefix_buf, DataflowBuffer& pending_buf, DataflowBuffer& total_buf, DataflowBuffer& total_out_buf) {
    prefix_buf.wait_front(Wt);
    pending_buf.wait_front(Wt);
    total_buf.reserve_back(Wt);
    total_out_buf.reserve_back(Wt);

    reconfig_data_format(cb_prefix, cb_pending);
    pack_reconfig_data_format(cb_total);
    add_init(cb_prefix, cb_pending);

    for (uint32_t wt = 0; wt < Wt; ++wt) {
        tile_regs_acquire();
        add_tiles(cb_prefix, cb_pending, wt, wt, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_total, wt);
        pack_tile(0, cb_total_out, wt);
        tile_regs_release();
    }

    total_buf.push_back(Wt);
    total_out_buf.push_back(Wt);
    prefix_buf.pop_front(Wt);
    pending_buf.pop_front(Wt);
    total_buf.wait_front(Wt);
}

// The scalars live in column 0 of their tile — the only column BroadcastType::COL
// reads — so the rest of the tile is padding and its exp/reciprocal results are
// never packed into a full-width operand.
//
// The gather leaves `cb_scalars` rank-major, which is the layout this reads it in;
// nothing is reordered on arrival.
void derive_row_weights(CircularBuffer& cb_row_obj) {
    unary_op_init_common(cb_scalars, cb_row);

    tile_regs_acquire();
    // The live score, from the statistics up:
    //
    //   live_scores = dots * rsqrt(sum_squares * inv_hidden_size + eps)
    //
    // summed across ranks first. `dst_a` is the only register free until the
    // sums collapse, so each rank's tiles land there one at a time.
    copy_tile(cb_scalars, kFixedScalars, dst_den);
    copy_tile(cb_scalars, kFixedScalars + 1, dst_b);

    if constexpr (num_partials > 1) {
        add_binary_tile_init();
        for (uint32_t p = 1; p < num_partials; ++p) {
            copy_tile(cb_scalars, kFixedScalars + kStatsPerPartial * p, dst_a);
            add_binary_tile(dst_den, dst_a, dst_den);
            copy_tile(cb_scalars, kFixedScalars + kStatsPerPartial * p + 1, dst_a);
            add_binary_tile(dst_b, dst_a, dst_b);
        }
    }

    binop_with_scalar_tile_init();
    mul_unary_tile(dst_den, inv_hidden_size);
    add_unary_tile(dst_den, eps);

    rsqrt_tile_init();
    rsqrt_tile(dst_den);

    mul_binary_tile_init();
    mul_binary_tile(dst_b, dst_den, dst_b);

    copy_tile(cb_scalars, 0, dst_a);
    copy_tile(cb_scalars, 1, dst_den);

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

    // `partial`, `running_sum` and any deferred write are separate CBs here rather than
    // one interleaved set, so the MAC alternates its srcA operand. That is only safe
    // because the device operation rejects a dtype mismatch among them — one
    // `bcast_init` configures the unpacker for all of them.
    bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(cb_partial, cb_row);
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL, MATH_FIDELITY>(
        cb_partial, cb_row, 1 /*acc_to_dest*/)));
    reconfig_data_format(cb_partial, cb_row);
    pack_reconfig_data_format(cb_out);
}

}  // namespace

void kernel_main() {
    // Statistics are split by token row and the fold by output tile, so a core's two
    // runs are unrelated. `num_stat_rows` is zero on a core that only folds.
    const uint32_t num_stat_rows = get_arg_val<uint32_t>(0);
    const uint32_t num_fold_tiles = get_arg_val<uint32_t>(1);
    const uint32_t fold_start_tile = get_arg_val<uint32_t>(2);

    CircularBuffer cb_partial_obj(cb_partial);
    CircularBuffer cb_prefix_obj(cb_prefix);
    CircularBuffer cb_pending_obj(cb_pending);
    CircularBuffer cb_row_obj(cb_row);
    CircularBuffer cb_scalars_obj(cb_scalars);
    CircularBuffer cb_out_obj(cb_out);

    DataflowBuffer prefix_buf(cb_prefix);
    DataflowBuffer pending_buf(cb_pending);
    DataflowBuffer total_buf(cb_total);
    DataflowBuffer total_out_buf(cb_total_out);
    DataflowBuffer q_buf(cb_q);
    DataflowBuffer tmp_buf(cb_tmp);
    DataflowBuffer scaler_buf(cb_scaler);

    compute_kernel_hw_startup(cb_prefix, cb_q, cb_tmp);

    // Pass one: this rank's statistics for every row this core carries, which is what
    // the writer parks and the gather core then puts on the wire. A core that only
    // folds skips the pass whole — the reader does not push q or the scaler to it.
    if (num_stat_rows > 0) {
        // q and the reduce scaler are the same for every row this core owns; the
        // reader pushes each once and nothing pops them until the pass is done.
        q_buf.wait_front(Wt);

        for (uint32_t row = 0; row < num_stat_rows; ++row) {
            if constexpr (fuse_add) {
                settle_row(prefix_buf, pending_buf, total_buf, total_out_buf);
            } else {
                prefix_buf.wait_front(Wt);
            }

            reduce_transformed_row(
                tmp_buf,
                [] {
                    reconfig_data_format(cb_stream, cb_stream);
                    pack_reconfig_data_format(cb_tmp);
                    mul_tiles_init(cb_stream, cb_stream);
                },
                [](uint32_t wt) { mul_tiles(cb_stream, cb_stream, wt, wt, 0); });

            reduce_transformed_row(
                tmp_buf,
                [] {
                    reconfig_data_format(cb_stream, cb_q);
                    pack_reconfig_data_format(cb_tmp);
                    mul_bcast_rows_init(cb_stream, cb_q);
                },
                [](uint32_t wt) { mul_tiles_bcast_rows(cb_stream, cb_q, wt, wt, 0); });

            if constexpr (fuse_add) {
                total_buf.pop_front(Wt);
            } else {
                prefix_buf.pop_front(Wt);
            }
        }

        q_buf.pop_front(Wt);
        scaler_buf.pop_front(1);
    }

    // Pass two: the fold, over a contiguous run of output tiles. The writer only
    // pushes scalar sets once every rank's slots have landed, so waiting on
    // `cb_scalars` is the whole of this kernel's participation in the collective.
    //
    // One derivation per token row the run touches, not per tile: the run is
    // contiguous, so a row's tiles are consecutive and the weights hold across them.
    bool row_weights_live = false;
    for (uint32_t i = fold_start_tile; i < fold_start_tile + num_fold_tiles; ++i) {
        if (!row_weights_live) {
            cb_scalars_obj.wait_front(kScalars);
            derive_row_weights(cb_row_obj);
            row_weights_live = true;
        }

        cb_partial_obj.wait_front(onetile);
        cb_prefix_obj.wait_front(onetile);
        if constexpr (fuse_add) {
            cb_pending_obj.wait_front(onetile);
        }
        tile_regs_acquire();
        mul_tiles_bcast_cols(cb_partial, cb_row, 0, 0, dst_acc);
        // The stream's weight is distributed over the two addends — `b*(prefix + pending)`
        // as `b*prefix + b*pending` — so the fold never waits on the settled row landing
        // in DRAM. It is one more MAC into the accumulator already open.
        mul_tiles_bcast_cols(cb_prefix, cb_row, 0, 1, dst_acc);
        if constexpr (fuse_add) {
            mul_tiles_bcast_cols(cb_pending, cb_row, 0, 1, dst_acc);
        }
        tile_regs_commit();

        cb_out_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_tile(dst_acc, cb_out);
        tile_regs_release();
        cb_out_obj.push_back(onetile);

        cb_partial_obj.pop_front(onetile);
        cb_prefix_obj.pop_front(onetile);
        if constexpr (fuse_add) {
            cb_pending_obj.pop_front(onetile);
        }

        if ((i + 1) % Wt == 0) {
            cb_scalars_obj.pop_front(kScalars);
            cb_row_obj.pop_front(kRowWeights);
            row_weights_live = false;
        }
    }
    if (row_weights_live) {
        cb_scalars_obj.pop_front(kScalars);
        cb_row_obj.pop_front(kRowWeights);
    }
}
