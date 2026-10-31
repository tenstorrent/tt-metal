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
// Two passes over one split: a core owns runs of token rows (see attn_res_core_runs.hpp),
// reduces them in pass one and folds the same runs in pass two. The stream it reduced
// stays resident between the passes, so it is read from DRAM once. A row's statistics
// reduce along its whole width, so where rows are split into chunks, each chunk's core
// reduces its own run and the row's chunk 0 core sums the row's partials before they
// cross the fabric.
//
// The two passes are separated by a barrier and not merely by data: the exchange is
// funnelled through a single fabric-holding core that cannot send until every row's
// statistics are parked, so no tile can fold before all of them have been produced.

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
#include "ttnn/operations/experimental/deepseek_prefill/attn_res_gather_softmax/device/kernels/attn_res_core_runs.hpp"

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
constexpr auto cb_chunk_stats = tt::CBIndex::c_14;  // a row leader's: its other chunks' pairs
constexpr auto cb_row_stats = tt::CBIndex::c_15;    // a row leader's: the row's summed pair
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
using Split = AttnResSplit<5>;
constexpr uint32_t chunk_tiles = Split::chunk_tiles;
constexpr uint32_t chunks_per_row = Split::chunks_per_row;

// What the statistics are taken over and the fold weights, resident from pass one to
// pass two: the live stream, or with a write to settle, the settled sum.
constexpr auto cb_stream = fuse_add ? cb_total : cb_prefix;

constexpr uint32_t kFixedScalars = 2;
constexpr uint32_t kStatsPerPartial = 2;
constexpr uint32_t kScalars = kFixedScalars + kStatsPerPartial * num_partials;
constexpr uint32_t kRowWeights = 2;
constexpr uint32_t kStatsPerRow = 2;
constexpr uint32_t onetile = 1;

constexpr uint32_t dst_a = 0;
constexpr uint32_t dst_b = 1;
constexpr uint32_t dst_den = 2;
constexpr uint32_t dst_max = 3;
constexpr uint32_t dst_acc = 0;

// The reduce drains `cb_tmp`, so the same buffer carries both transforms of the
// row: fill it, reduce it, fill it again. `init` re-establishes the unpack and
// pack configuration the previous reduce left changed.
template <typename DFB, typename Init, typename TransformOne>
ALWI void reduce_transformed_run(DFB& tmp_buf, Init init, TransformOne transform_one) {
    init();
    tmp_buf.reserve_back(chunk_tiles);
    for (uint32_t k = 0; k < chunk_tiles; ++k) {
        tile_regs_acquire();
        transform_one(k);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_tmp, k);
        tile_regs_release();
    }
    tmp_buf.push_back(chunk_tiles);

    compute_kernel_lib::reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        cb_tmp,
        cb_scaler,
        cb_local_stats,
        compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
        compute_kernel_lib::ReduceInputBlockShape::row(chunk_tiles));
}

// Settle a deferred residual write into the run the statistics are about to be taken
// over. The run is packed twice out of one dest register: a circular buffer has a single
// consumer, and the sum is wanted both here — by the two reduces and the fold, which keep
// it resident — and by the writer that parks it as the stream the caller carries forward.
template <typename PrefixDFB, typename PendingDFB, typename TotalDFB, typename TotalOutDFB>
ALWI void settle_run(PrefixDFB& prefix_buf, PendingDFB& pending_buf, TotalDFB& total_buf, TotalOutDFB& total_out_buf) {
    prefix_buf.wait_front(chunk_tiles);
    pending_buf.wait_front(chunk_tiles);
    total_buf.reserve_back(chunk_tiles);
    total_out_buf.reserve_back(chunk_tiles);

    reconfig_data_format(cb_prefix, cb_pending);
    pack_reconfig_data_format(cb_total);
    add_init(cb_prefix, cb_pending);

    for (uint32_t k = 0; k < chunk_tiles; ++k) {
        tile_regs_acquire();
        add_tiles(cb_prefix, cb_pending, k, k, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_total, k);
        pack_tile(0, cb_total_out, k);
        tile_regs_release();
    }

    total_buf.push_back(chunk_tiles);
    total_out_buf.push_back(chunk_tiles);
    prefix_buf.pop_front(chunk_tiles);
    pending_buf.pop_front(chunk_tiles);
}

// A row leader's: its own pair plus the row's other chunks', summed in fp32 dest.
void sum_row_stats() {
    CircularBuffer cb_local_obj(cb_local_stats);
    CircularBuffer cb_chunk_obj(cb_chunk_stats);
    CircularBuffer cb_row_obj(cb_row_stats);

    cb_local_obj.wait_front(kStatsPerRow);
    cb_chunk_obj.wait_front(kStatsPerRow * (chunks_per_row - 1));
    cb_row_obj.reserve_back(kStatsPerRow);
    copy_init(cb_local_stats);
    pack_reconfig_data_format(cb_row_stats);
    for (uint32_t s = 0; s < kStatsPerRow; ++s) {
        tile_regs_acquire();
        copy_tile(cb_local_stats, s, dst_a);
        add_binary_tile_init();
        for (uint32_t c = 1; c < chunks_per_row; ++c) {
            copy_tile(cb_chunk_stats, kStatsPerRow * (c - 1) + s, dst_b);
            add_binary_tile(dst_a, dst_b, dst_a);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst_a, cb_row_stats);
        tile_regs_release();
    }
    cb_row_obj.push_back(kStatsPerRow);
    cb_chunk_obj.pop_front(kStatsPerRow * (chunks_per_row - 1));
    cb_local_obj.pop_front(kStatsPerRow);
}

// The scalars live in column 0 of their tile — the only column BroadcastType::COL
// reads — so the rest of the tile is padding and its exp/reciprocal results are
// never packed into a full-width operand.
//
// The gather leaves `cb_scalars` rank-major, which is the layout this reads it in;
// nothing is reordered on arrival.
void derive_row_weights(CircularBuffer& cb_row_obj) {
    // Full re-init (hw_configure + pack_dest/math_pack_sync), matching the removed unary_op_init_common.
    // TODO(#52395): compute_kernel_hw_startup is a call-once API; this mid-kernel re-init (preserving the
    // pre-cleanup full-init behaviour) should become a targeted DST re-arm.
    compute_kernel_hw_startup(cb_scalars, cb_row);
    copy_init(cb_scalars);

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

    // `partial` and the stream are separate CBs here rather than one interleaved set, so
    // the MAC alternates its srcA operand. That is only safe because the device
    // operation rejects a dtype mismatch among them — one `bcast_init` configures the
    // unpacker for all of them.
    bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(cb_partial, cb_row);
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL, MATH_FIDELITY>(
        cb_partial, cb_row, 1 /*acc_to_dest*/)));
    reconfig_data_format(cb_partial, cb_row);
    pack_reconfig_data_format(cb_out);
}

}  // namespace

void kernel_main() {
    const auto [first_unit, num_units] = Split::core_units();

    CircularBuffer cb_partial_obj(cb_partial);
    CircularBuffer cb_stream_obj(cb_stream);
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

    // Pass one: this rank's statistics for every run this core carries, which the writer
    // parks — or hands to the row's leader — and the gather core then puts on the wire.
    // q and the reduce scaler are the same for every run; the reader pushes each once.
    q_buf.wait_front(chunk_tiles);
    for (uint32_t i = 0; i < num_units; ++i) {
        // The stream stays resident for the fold, so a run's tiles sit after the earlier
        // runs' in its buffer.
        const uint32_t base = i * chunk_tiles;
        if constexpr (fuse_add) {
            settle_run(prefix_buf, pending_buf, total_buf, total_out_buf);
        }
        cb_stream_obj.wait_front(base + chunk_tiles);

        reduce_transformed_run(
            tmp_buf,
            [] {
                reconfig_data_format(cb_stream, cb_stream);
                pack_reconfig_data_format(cb_tmp);
                mul_init(cb_stream, cb_stream);
            },
            [base](uint32_t k) { mul_tiles(cb_stream, cb_stream, base + k, base + k, 0); });

        reduce_transformed_run(
            tmp_buf,
            [] {
                reconfig_data_format(cb_stream, cb_q);
                pack_reconfig_data_format(cb_tmp);
                mul_bcast_rows_init(cb_stream, cb_q);
            },
            [base](uint32_t k) { mul_tiles_bcast_rows(cb_stream, cb_q, base + k, k, 0); });

        if constexpr (chunks_per_row > 1) {
            if (Split::unit(first_unit + i).chunk == 0) {
                sum_row_stats();
            }
        }
    }
    q_buf.pop_front(chunk_tiles);
    scaler_buf.pop_front(1);

    // Pass two: the fold, over the same runs. The writer only pushes a run's scalar set
    // once every rank's slots have landed, so waiting on `cb_scalars` is the whole of
    // this kernel's participation in the collective. A run lies within one token row, so
    // one derivation of the row weights serves it.
    cb_partial_obj.wait_front(num_units * chunk_tiles);
    for (uint32_t i = 0; i < num_units; ++i) {
        cb_scalars_obj.wait_front(kScalars);
        derive_row_weights(cb_row_obj);

        for (uint32_t k = i * chunk_tiles; k < (i + 1) * chunk_tiles; ++k) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(cb_partial, cb_row, k, 0, dst_acc);
            mul_tiles_bcast_cols(cb_stream, cb_row, k, 1, dst_acc);
            tile_regs_commit();

            cb_out_obj.reserve_back(onetile);
            tile_regs_wait();
            pack_tile(dst_acc, cb_out);
            tile_regs_release();
            cb_out_obj.push_back(onetile);
        }

        cb_scalars_obj.pop_front(kScalars);
        cb_row_obj.pop_front(kRowWeights);
    }
    cb_partial_obj.pop_front(num_units * chunk_tiles);
    cb_stream_obj.pop_front(num_units * chunk_tiles);
}
