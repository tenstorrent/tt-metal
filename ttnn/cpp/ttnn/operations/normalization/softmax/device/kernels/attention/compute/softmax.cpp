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

// Templated on the CBs so the eltwise_chain below can take them as compile-time NTTPs.
// All call sites pass constexpr CBs.
template <uint32_t cb_in, uint32_t cb_max_scaler, uint32_t cb_max, uint32_t cb_out>
void calc_numeric_stable(uint32_t Wt, uint32_t ndst) {
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

    // x - max(x) then exp, fused into one chain — DEST-batched ndst tiles per acquire,
    // matching the original's `for (wt += ndst)` window. cb_in is fully resident (the
    // reduce above WaitUpfrontNoPop-waited all Wt tiles; popped below) -> CallerManaged +
    // Block, read by absolute index `wt_base + j`. cb_max held (wait/pop kept outside) ->
    // CallerManaged + Scalar. cb_out per-chunk reserve+push -> Chunked (reserve ndst /
    // push ndst per block, == original reserve_back(ndst)/push_back(ndst)).
    // EltwiseShape::tiles(Wt, ndst): ndst is the RUNTIME block_size; the chain clamps it to
    // the DEST/lane capacity automatically. sub_bcast_cols_init_short ->
    // BinaryDataFormatReconfig::Input; plain pack_tile (pack format already cb_out) ->
    // PackTileReconfig::None.
    cb_max_obj.wait_front(1);
    compute_kernel_lib::eltwise_chain(
        compute_kernel_lib::EltwiseShape::tiles(Wt, ndst),
        compute_kernel_lib::BinaryFpu<
            cb_in,
            cb_max,
            compute_kernel_lib::BinaryFpuOp::Sub,
            compute_kernel_lib::BroadcastDim::Col,
            compute_kernel_lib::InputLifecycle::CallerManaged,
            compute_kernel_lib::InputLifecycle::CallerManaged,
            compute_kernel_lib::BinaryDataFormatReconfig::Input,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::OperandKind::Block,
            compute_kernel_lib::OperandKind::Scalar>{},
        compute_kernel_lib::Exp<
            static_cast<compute_kernel_lib::Approx>(EXP_APPROX),
            compute_kernel_lib::Approx::Exact,
            compute_kernel_lib::Dst::D0>{},
        compute_kernel_lib::PackTile<
            cb_out,
            compute_kernel_lib::OutputLifecycle::Chunked,
            compute_kernel_lib::PackTileReconfig::None>{});
    cb_in_obj.pop_front(Wt);
    cb_max_obj.pop_front(1);
    cb_out_obj.wait_front(Wt);
}

void kernel_main() {
    const uint32_t NCHt = get_arg_val<uint32_t>(0);
    const uint32_t Ht = get_arg_val<uint32_t>(1);
    const uint32_t Wt = get_arg_val<uint32_t>(2);
    const uint32_t ndst = get_arg_val<uint32_t>(3);  // DEST-batch size (== host block_size)
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
            compute_kernel_lib::InputLifecycle::Streaming,
            compute_kernel_lib::InputLifecycle::CallerManaged>(Wt);
        // fused mask add (+exp) — DEST-batched ndst tiles per acquire, matching the original's
        // `for (wt += ndst)` window. Both inputs are fully resident before this chain runs, so
        // both walk by absolute index `wt_base + j`:
        //   cb_scale_mask: Site-1 (fused scale) above pushed all Wt tiles into it (cb_scale_mask
        //   holds Wt + block_size); the original waited/popped ndst per chunk reading relative
        //   wt8 (front pinned at 0 because it isn't popped between chunks) == chain Bulk + Block
        //   (wait Wt upfront, walk absolute, pop Wt at end).
        //   cb_fused_attn: absolute-indexed `wt + wt8`, held across Ht (waited upfront, popped
        //   below) -> CallerManaged + Block. Its cumulative wait collapses to one upfront
        //   wait_front(Wt) — its CB (cb_fused_attn = Wt-resident) holds all Wt tiles.
        // cb_x per-chunk reserve+push -> Chunked (cb_x = cb_exps/cb_x are Wt-resident).
        // reconfig_data_format + add init -> BinaryDataFormatReconfig::Input; plain pack_tile
        // (cb_x/cb_scale_mask share format) -> PackTileReconfig::None. Exp dropped when
        // NUMERIC_STABLE (done later in calc_numeric_stable). tiles(Wt, ndst): ndst is the
        // RUNTIME block_size, clamped to DEST capacity by the chain.
#ifdef CAUSAL_MASK
        cb_fused_attn_obj.wait_front(Wt);
#else
        if (wait_mask) {
            cb_fused_attn_obj.wait_front(Wt);
        }
#endif
        compute_kernel_lib::eltwise_chain(
            compute_kernel_lib::EltwiseShape::tiles(Wt, ndst),
            compute_kernel_lib::BinaryFpu<
                cb_scale_mask,
                cb_fused_attn,
                compute_kernel_lib::BinaryFpuOp::Add,
#ifdef CAUSAL_MASK
                compute_kernel_lib::BroadcastDim::None,
#else
                compute_kernel_lib::BroadcastDim::Row,
#endif
                compute_kernel_lib::InputLifecycle::Bulk,
                compute_kernel_lib::InputLifecycle::CallerManaged,
                compute_kernel_lib::BinaryDataFormatReconfig::Input,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OperandKind::Block,
                compute_kernel_lib::OperandKind::Block>{},
#ifndef NUMERIC_STABLE
            compute_kernel_lib::Exp<
                static_cast<compute_kernel_lib::Approx>(EXP_APPROX),
                compute_kernel_lib::Approx::Exact,
                compute_kernel_lib::Dst::D0>{},
#endif
            compute_kernel_lib::PackTile<
                cb_x,
                compute_kernel_lib::OutputLifecycle::Chunked,
                compute_kernel_lib::PackTileReconfig::None>{});

// add numeric_stable
// fuse exp with sub tiles
#ifdef NUMERIC_STABLE
        calc_numeric_stable<cb_x, cb_max_scaler, cb_max, cb_exps>(Wt, ndst);
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
                compute_kernel_lib::CopyTile<cb_in0>{},
#ifndef NUMERIC_STABLE
                compute_kernel_lib::Exp<
                    static_cast<compute_kernel_lib::Approx>(EXP_APPROX),
                    compute_kernel_lib::Approx::Exact,
                    compute_kernel_lib::Dst::D0>{},
#endif
                compute_kernel_lib::PackTile<
                    cb_x,
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::None>{});

            compute_kernel_lib::eltwise_chain(
                1u,
                compute_kernel_lib::BinaryFpu<
                    cb_in0,
                    cb_mask_padded,
                    compute_kernel_lib::BinaryFpuOp::Add,
                    compute_kernel_lib::BroadcastDim::Row,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::InputLifecycle::CallerManaged>{},
#ifndef NUMERIC_STABLE
                compute_kernel_lib::Exp<
                    static_cast<compute_kernel_lib::Approx>(EXP_APPROX),
                    compute_kernel_lib::Approx::Exact,
                    compute_kernel_lib::Dst::D0>{},
#endif
                compute_kernel_lib::PackTile<
                    cb_x,
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::None>{});

// add numeric_stable
// fuse exp with sub tiles
#ifdef NUMERIC_STABLE
            calc_numeric_stable<cb_x, cb_max_scaler, cb_max, cb_exps>(Wt, ndst);
#endif

        } else {
// add numeric_stable
// fuse exp with sub tiles
#ifdef NUMERIC_STABLE
            calc_numeric_stable<cb_in0, cb_max_scaler, cb_max, cb_exps>(Wt, ndst);
#else
            // Migrated: streaming copy + exp + pack via eltwise_chain. LEFT AT BlockSize=1
            // (NOT DEST-batched) on purpose: the original consumed cb_in0 per-ndst with a
            // cb_wait_front(ndst)/cb_pop_front(ndst) INSIDE the loop and read RELATIVE index
            // wt8 — i.e. cb_in0 is a streamed reader, NOT row-resident. Here cb_in0 = in0_t =
            // block_size*2 (the Wt-resident size only applies in the no-mask NUMERIC_STABLE
            // path; this is the non-NS branch), so it cannot hold the whole Wt row. A Block
            // walker would need an upfront wait_front(Wt) on a 2*block_size CB -> deadlock, so
            // the input stays InputLifecycle::Streaming (which forces the chain's block to 1).
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
                compute_kernel_lib::InputLifecycle::Streaming,
                compute_kernel_lib::OutputLifecycle::Streaming,
                compute_kernel_lib::CopyTileReconfig::Input,
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

        // multiply by 1/sum(exp(x)) — bcast COL on the held cb_recipsumexps — DEST-batched
        // ndst tiles per acquire, matching the original's `for (wt += ndst)` window. The SUM
        // reduce above (WaitUpfrontNoPop) already waited all Wt tiles of cb_exps, so they are
        // fully resident (cb_exps = Wt-resident): cb_exps walks by absolute index `wt_base + j`
        // -> CallerManaged + Block, with the single final cb_exps_obj.pop_front(Wt) kept
        // OUTSIDE the chain (== the original's lone pop_front(Wt) after the mul loop).
        // cb_recipsumexps held outside (wait/pop bracket the chain) -> CallerManaged + Scalar.
        // cb_out0 (= block_size*2, NOT row-resident) gets per-chunk reserve+push -> Chunked
        // (reserve ndst / push ndst per block, == original reserve_back(ndst)/push_back(ndst)).
        //
        // Reconfig: reconfig_data_format(cb_exps, cb_recipsumexps) +
        // mul_bcast_cols_init_short reconfig srca/srcb -> Input.
        // pack_reconfig_data_format(cb_out0) -> PackTileReconfig::Output.
        // tiles(Wt, ndst): ndst is the RUNTIME block_size, clamped to DEST capacity.
        compute_kernel_lib::mul<
            cb_exps,
            cb_recipsumexps,
            cb_out0,
            compute_kernel_lib::BroadcastDim::Col,
            compute_kernel_lib::InputLifecycle::CallerManaged,
            compute_kernel_lib::InputLifecycle::CallerManaged,
            compute_kernel_lib::OutputLifecycle::Chunked,
            compute_kernel_lib::BinaryDataFormatReconfig::Input,
            compute_kernel_lib::PackTileReconfig::Output,
            compute_kernel_lib::OperandKind::Block,
            compute_kernel_lib::OperandKind::Scalar>(compute_kernel_lib::EltwiseShape::tiles(Wt, ndst));
        cb_exps_obj.pop_front(Wt);
        cb_recipsumexps_obj.pop_front(1);
    }  // NCHt loop
    // cb_pop_front(cb_max_scaler, 1); // we don't actually have to do this
    // cb_pop_front(cb_fused_scale, 1); // we don't actually have to do this
}
