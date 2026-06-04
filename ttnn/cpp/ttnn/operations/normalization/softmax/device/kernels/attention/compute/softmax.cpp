// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/softmax.h"
#include "api/compute/reduce.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"

// for scale+mask+softmax:
// bcast HW (mul by 1 tile)  example: (  [2,1,1024,64] * [1,1,32,32]  )
// bcast add H               example: ( [2,1,1024,64] + [2,1,32,64] ) (bcast W -> H)
// Note that the attention mask will not fit in L1 for the entire tensor
// The buffer for the att mask is currently sized as (1t,Wt) so we only reuse it for one HtWt-sized batch of x
// then read another Wt tiles of mask for the next batch

void calc_numeric_stable(
    uint32_t Wt, uint32_t ndst, uint32_t cb_in, uint32_t cb_max_scaler, uint32_t cb_max, uint32_t cb_out) {
    auto cb_in_obj = CircularBuffer(cb_in);
    auto cb_max_obj = CircularBuffer(cb_max);
    auto cb_out_obj = CircularBuffer(cb_out);

    // calculate max val per row
    compute_kernel_lib::reduce<
        PoolType::MAX,
        ReduceDim::REDUCE_ROW,
        compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT>(
        cb_in, cb_max_scaler, cb_max, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

    // calculate x-max(x)
    exp_tile_init<EXP_APPROX>();
    reconfig_data_format_srcb(cb_max);
    cb_max_obj.wait_front(1);
    sub_bcast_cols_init_short(cb_in, cb_max);
    for (uint32_t wt = 0; wt < Wt; wt += ndst) {
        tile_regs_acquire();
        for (uint32_t wt8 = 0; wt8 < ndst; wt8++) {
            sub_tiles_bcast_cols(cb_in, cb_max, wt + wt8, 0, wt8);
        }
        cb_out_obj.reserve_back(ndst);
        for (uint32_t wt8 = 0; wt8 < ndst; wt8++) {
            exp_tile<EXP_APPROX>(wt8);  // exp on DST[0]
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t wt8 = 0; wt8 < ndst; wt8++) {
            pack_tile(wt8, cb_out);     // reuse the exps buffer again, this time in a circular manner
        }
        tile_regs_release();
        cb_out_obj.push_back(ndst);
    }
    cb_in_obj.pop_front(Wt);
    cb_max_obj.pop_front(1);
    cb_out_obj.wait_front(Wt);
}

void kernel_main() {
    const uint32_t NCHt = get_arg_val<uint32_t>(0);
    const uint32_t Ht = get_arg_val<uint32_t>(1);
    const uint32_t Wt = get_arg_val<uint32_t>(2);
    const uint32_t ndst = get_arg_val<uint32_t>(3);
    const uint32_t start_ht = get_arg_val<uint32_t>(4);
    const uint32_t mask_padded_data = get_arg_val<uint32_t>(5);
    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_6);

    constexpr uint32_t onetile = 1;
    // reserve one tile for zeros on cb_in2
    // We only do the reserve for the intermediates once and use pack_tile
    // So effectively these are used as pre-allocated arrays
    // Note that the entire W dimension must fit in the intermed0 CB for this kernel to be correct
    constexpr auto cb_max_scaler = tt::CBIndex::c_2;
    constexpr auto cb_sum_scaler = tt::CBIndex::c_13;
    constexpr auto cb_fused_scale = tt::CBIndex::c_3;
    constexpr auto cb_fused_attn = tt::CBIndex::c_4;
    constexpr auto cb_mask_padded = tt::CBIndex::c_5;
    constexpr auto cb_exps = tt::CBIndex::c_6;
    constexpr auto cb_scale_mask = tt::CBIndex::c_9;
    constexpr auto cb_recipsumexps = tt::CBIndex::c_7;
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_out0 = tt::CBIndex::c_11;
    CircularBuffer cb_max_scaler_obj(cb_max_scaler);
    CircularBuffer cb_sum_scaler_obj(cb_sum_scaler);
    CircularBuffer cb_fused_scale_obj(cb_fused_scale);
    CircularBuffer cb_fused_attn_obj(cb_fused_attn);
    CircularBuffer cb_mask_padded_obj(cb_mask_padded);
    CircularBuffer cb_exps_obj(cb_exps);
    CircularBuffer cb_scale_mask_obj(cb_scale_mask);
    CircularBuffer cb_recipsumexps_obj(cb_recipsumexps);
    CircularBuffer cb_in0_obj(cb_in0);
    CircularBuffer cb_out0_obj(cb_out0);
#ifdef NUMERIC_STABLE
    constexpr auto cb_max = tt::CBIndex::c_8;
    constexpr auto cb_x = tt::CBIndex::c_10;
    CircularBuffer cb_max_obj(cb_max);
#else
    constexpr auto cb_x = cb_exps;
#endif
    CircularBuffer cb_x_obj(cb_x);

    cb_max_scaler_obj.wait_front(1);  // comes from the reader
    cb_sum_scaler_obj.wait_front(1);  // comes from the reader

#if FUSED_SCALE_MASK
    cb_fused_scale_obj.wait_front(1);
#endif

    constexpr int dst0 = 0;
    uint32_t ht = start_ht;
    bool wait_mask = true;
    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
#if FUSED_SCALE_MASK
        // apply fused scale [*= 1/sqrt(...)] — cb_in0 * cb_fused_scale (scalar
        // bcast) -> cb_scale_mask. Per-ndst DEST batching collapsed to per-tile
        // streaming via chain: cb_in0 InputLifecycle::Streaming + Scalar (wait+pop 1 per iter),
        // cb_fused_scale InputLifecycle::CallerManaged (held outside via line 112 wait_front(1)),
        // cb_scale_mask OutputLifecycle::Streaming (per-tile reserve+push).
        //
        // Reconfig: reconfig_data_format(cb_in0, cb_fused_scale) +
        // mul_tiles_bcast_scalar_init_short -> BinaryDataFormatReconfig::Input.
        // pack_reconfig_data_format(cb_scale_mask) -> PackTileReconfig::Output.
        compute_kernel_lib::mul<
            cb_in0,
            cb_fused_scale,
            cb_scale_mask,
            compute_kernel_lib::BroadcastDim::Scalar,
            compute_kernel_lib::BinaryDataFormatReconfig::Input,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::InputLifecycle::Streaming,
            compute_kernel_lib::InputLifecycle::CallerManaged>(Wt);
        reconfig_data_format(cb_scale_mask, cb_fused_attn);

#ifndef NUMERIC_STABLE
        exp_tile_init<EXP_APPROX>();
#endif

#ifdef CAUSAL_MASK
        add_tiles_init(cb_scale_mask, cb_fused_attn);
#else
        add_bcast_rows_init_short(cb_scale_mask, cb_fused_attn);
#endif
        for (uint32_t wt = 0; wt < Wt; wt += ndst) {
            tile_regs_acquire();
            cb_scale_mask_obj.wait_front(ndst);
#ifdef CAUSAL_MASK
            cb_fused_attn_obj.wait_front(wt + ndst);  // cumulative wait for up to Wt tiles
            for (uint32_t wt8 = 0; wt8 < ndst; wt8++) {
                add_tiles(cb_scale_mask, cb_fused_attn, wt8, wt + wt8, wt8);  // tile *= 1/(sum(exp(x)))
            }
#else
            if (wait_mask) {
                cb_fused_attn_obj.wait_front(wt + ndst);  // cumulative wait for up to Wt tiles, only at first ht
            }

            for (uint32_t wt8 = 0; wt8 < ndst; wt8++) {
                add_tiles_bcast_rows(cb_scale_mask, cb_fused_attn, wt8, wt + wt8, wt8);  // tile *= 1/(sum(exp(x)))
            }
#endif
            cb_scale_mask_obj.pop_front(ndst);
            cb_x_obj.reserve_back(ndst);
#ifndef NUMERIC_STABLE
            for (uint32_t wt8 = 0; wt8 < ndst; wt8++) {
                exp_tile<EXP_APPROX>(wt8);  // exp on DST[0]
            }
#endif
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t wt8 = 0; wt8 < ndst; wt8++) {
                pack_tile(wt8, cb_x);  // reuse the exps buffer again, this time in a circular manner
            }
            tile_regs_release();
            cb_x_obj.push_back(ndst);
        }

// add numeric_stable
// fuse exp with sub tiles
#ifdef NUMERIC_STABLE
        calc_numeric_stable(Wt, ndst, cb_x, cb_max_scaler, cb_max, cb_exps);
#endif

#ifdef CAUSAL_MASK
        cb_fused_attn_obj.pop_front(Wt);
#else
        if (wait_mask) {
            wait_mask = false;
        }
        ht++;
        if (ht == Ht) {
            cb_fused_attn_obj.pop_front(Wt);
            ht = 0;
            wait_mask = true;
        }
#endif  // CAUSAL_MASK

        reconfig_data_format(cb_exps, cb_sum_scaler);
#else
        reconfig_data_format(cb_in0, cb_in0);
        pack_reconfig_data_format(cb_exps);
        copy_tile_to_dst_init_short(cb_in0);  // need to copy from CB to DST to be able to run sfpu math
#ifndef NUMERIC_STABLE
        exp_tile_init<EXP_APPROX>();
#endif
        if (mask_padded_data) {
            // mask_padded path: Wt-1 plain copy(+exp) tiles + 1 last tile with
            // bcast-rows add of cb_mask_padded(+exp). Split into 2 chains:
            //   A: Wt-1 iters — CopyTile(cb_in0) + [Exp if !NUMERIC_STABLE] + PackTile(cb_x)
            //   B: 1 iter — BinaryFpu(cb_in0, cb_mask_padded, Add, Row) +
            //               [Exp if !NUMERIC_STABLE] + PackTile(cb_x)
            // cb_mask_padded held outside via wait_front(1); InputLifecycle::CallerManaged.
            // cb_in0 InputLifecycle::Streaming + Scalar (per-tile wait+pop, total Wt).
            // cb_x OutputLifecycle::Streaming.
            //
            // Reconfig: copy_tile_init / add_bcast_rows_init_short
            // reconfig srca/srcb -> Input.
            cb_mask_padded_obj.wait_front(1);
            compute_kernel_lib::eltwise_chain(
                Wt - 1,
                compute_kernel_lib::CopyTile<
                    cb_in0,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::CopyTileReconfig::Input>{},
#ifndef NUMERIC_STABLE
                compute_kernel_lib::Exp<
                    static_cast<compute_kernel_lib::Approx>(EXP_APPROX),
                    compute_kernel_lib::Approx::Exact,
                    compute_kernel_lib::Dst::D0>{},
#endif
                compute_kernel_lib::PackTile<
                    cb_x,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::None>{});

            compute_kernel_lib::eltwise_chain(
                1u,
                compute_kernel_lib::BinaryFpu<
                    cb_in0,
                    cb_mask_padded,
                    compute_kernel_lib::BinaryFpuOp::Add,
                    compute_kernel_lib::BroadcastDim::Row,
                    compute_kernel_lib::BinaryDataFormatReconfig::Input,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::InputLifecycle::CallerManaged,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OperandKind::Scalar>{},
#ifndef NUMERIC_STABLE
                compute_kernel_lib::Exp<
                    static_cast<compute_kernel_lib::Approx>(EXP_APPROX),
                    compute_kernel_lib::Approx::Exact,
                    compute_kernel_lib::Dst::D0>{},
#endif
                compute_kernel_lib::PackTile<
                    cb_x,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::None>{});

// add numeric_stable
// fuse exp with sub tiles
#ifdef NUMERIC_STABLE
            calc_numeric_stable(Wt, ndst, cb_x, cb_max_scaler, cb_max, cb_exps);
#endif

        } else {
// add numeric_stable
// fuse exp with sub tiles
#ifdef NUMERIC_STABLE
            calc_numeric_stable(Wt, ndst, cb_in0, cb_max_scaler, cb_max, cb_exps);
#else
            // Migrated: streaming copy + exp + pack via eltwise_chain.
            // Per-tile semantics (BlockSize=1) replaces original's per-ndst ACQ window —
            // semantically equivalent because reads/writes are sequential and the chain
            // re-acquires DEST per tile. ndst is a runtime arg so we can't use it as
            // BlockSize template; perf trade-off accepted (more ACQ/REL pairs).
            // Reconfig: copy_tile + exp_tile are SFPU/copy ops; no explicit
            // reconfig_data_format outside this block, so CopyTileReconfig::Input matches
            // copy_tile_init's reconfig. PackTileReconfig::None — pack format set by
            // binary_op_init_common at line 70 to cb_exps already.
            compute_kernel_lib::unary<
                compute_kernel_lib::Exp<
                    static_cast<compute_kernel_lib::Approx>(EXP_APPROX),
                    compute_kernel_lib::Approx::Exact,
                    compute_kernel_lib::Dst::D0>,
                cb_in0,
                cb_exps,
                compute_kernel_lib::CopyTileReconfig::Input,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::InputLifecycle::Streaming,
                compute_kernel_lib::OutputLifecycle::Streaming,
                compute_kernel_lib::PackTileReconfig::None>(Wt);
#endif
        }
#endif

        // SUM reduce with reciprocal post-processing (1/sum)
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_exps,
                cb_sum_scaler,
                cb_recipsumexps,
                compute_kernel_lib::ReduceInputBlockShape::row(Wt),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::NoAccumulation{},
                [](uint32_t) {
                    recip_tile_init();
                    recip_tile(0);
                });

        cb_recipsumexps_obj.wait_front(1);  // will reuse Wt times for bcast

        // multiply by 1/sum(exp(x)) — bcast COL on the held cb_recipsumexps.
        // Original cumulative-waited Wt tiles in cb_exps upfront + did 1 final
        // pop_front(Wt). Chain InputLifecycle::Streaming + Scalar emits per-tile wait_front(1)
        // + pop_front(1) — same net effect over Wt iters since reader pushed
        // all Wt tiles upfront.
        //
        // Reconfig: reconfig_data_format(cb_exps, cb_recipsumexps) +
        // mul_bcast_cols_init_short reconfig srca/srcb -> Input.
        // pack_reconfig_data_format(cb_out0) -> PackTileReconfig::Output.
        // cb_recipsumexps held outside (wait/pop bracket the chain) -> InputLifecycle::CallerManaged.
        compute_kernel_lib::mul<
            cb_exps,
            cb_recipsumexps,
            cb_out0,
            compute_kernel_lib::BroadcastDim::Col,
            compute_kernel_lib::BinaryDataFormatReconfig::Input,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::InputLifecycle::Streaming,
            compute_kernel_lib::InputLifecycle::CallerManaged>(Wt);
        cb_recipsumexps_obj.pop_front(1);
    }  // NCHt loop
    // cb_pop_front(cb_max_scaler, 1); // we don't actually have to do this
    // cb_pop_front(cb_fused_scale, 1); // we don't actually have to do this
}
