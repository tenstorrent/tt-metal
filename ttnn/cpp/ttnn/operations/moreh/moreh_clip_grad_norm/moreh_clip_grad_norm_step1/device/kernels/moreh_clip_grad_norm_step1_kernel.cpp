// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"  // PowerIterative, Recip, Log, Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"  // Mask, Abs
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/circular_buffer.h"

ALWI bool need_to_do_mask_h(uint32_t tile_idx, uint32_t ht, uint32_t wt) { return (((tile_idx / wt) + 1) % ht) == 0; }

void kernel_main() {
    int i{0};
    const auto num_tiles = get_arg_val<uint32_t>(i++);
    const auto p = get_arg_val<uint32_t>(i++);
    const bool p_is_negative = get_arg_val<uint32_t>(i++) == 1;
    const auto origin_h = get_arg_val<uint32_t>(i++);
    const auto origin_w = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_x = 0;         // input(==x)
    constexpr uint32_t cb_one = 1;       // one
    constexpr uint32_t cb_decimal = 2;   // decimal
    constexpr uint32_t cb_mask_h_w = 3;  // mask_h_w

    constexpr uint32_t cb_y = 16;  // output(==y)

    constexpr uint32_t cb_xabs = 24;          // |x|
    constexpr uint32_t cb_xpow = 25;          // |x|^p
    constexpr uint32_t cb_xpowadd = 26;       // Add[|x|^p * exp(log(|x|) * decimal)]
    constexpr uint32_t cb_logx = 27;          // log(|x|)
    constexpr uint32_t cb_exp_lxmd = 28;      // exp(log(|x|) * decimal)
    constexpr uint32_t cb_correct_xpow = 29;  // |x|^p * exp(log(|x|) * decimal)

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;

    const bool do_mask_h = (origin_h % TILE_H) != 0;
    const bool do_mask_w = (origin_w % TILE_W) != 0;

    const auto ht = (origin_h + TILE_H - 1) / TILE_H;
    const auto wt = (origin_w + TILE_W - 1) / TILE_W;

    binary_op_init_common(cb_logx, cb_decimal, cb_y);

    cb_wait_front(cb_decimal, onetile);  // comes from the reader
    cb_wait_front(cb_one, onetile);      // comes from the reader

    if (do_mask_h || do_mask_w) {
        cb_wait_front(cb_mask_h_w, 2);  // comes from the reader
    }

    // Compute cb_xpowadd
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        // abs(x) with optional H/W masking.
        // 4-branch runtime dispatch on (mh, mw): copy_x -> [opt mask(0)] ->
        // [opt mask(1)] -> abs -> pack to cb_xabs.
        //
        // Reconfig audit: copy_tile_init reconfigs srca per copy ->
        //   CopyTileReconfig::Input on each copy. pack_tile (no reconfig) ->
        //   PackTileReconfig::None.
        // Lifecycles: cb_x InputLifecycle::Streaming (chain owns wait+pop); cb_mask_h_w
        //   InputLifecycle::CallerManaged + Scalar (waited once outside loop at line 53;
        //   chain reads at index 0 / index 1 via compute_kernel_lib::TileOffset::Set);
        //   cb_xabs OutputLifecycle::Streaming.
        const bool mh = do_mask_h && need_to_do_mask_h(tile_idx, ht, wt);
        const bool mw = do_mask_w && ((tile_idx + 1) % wt) == 0;
        if (mh && mw) {
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::CopyTile<
                    cb_x,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::CopyTileReconfig::Input>{},
                compute_kernel_lib::CopyTile<
                    cb_mask_h_w,
                    compute_kernel_lib::Dst::D1,
                    compute_kernel_lib::InputLifecycle::CallerManaged,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::CopyTileReconfig::Input>{},
                compute_kernel_lib::Mask<DataFormat::Float16_b, compute_kernel_lib::Dst::D0>{},
                compute_kernel_lib::CopyTile<
                    cb_mask_h_w,
                    compute_kernel_lib::Dst::D1,
                    compute_kernel_lib::InputLifecycle::CallerManaged,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::CopyTileReconfig::Input,
                    compute_kernel_lib::TileOffset::Set>{},
                compute_kernel_lib::Mask<DataFormat::Float16_b, compute_kernel_lib::Dst::D0>{},
                compute_kernel_lib::Abs<compute_kernel_lib::Dst::D0>{},
                compute_kernel_lib::PackTile<
                    cb_xabs,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::None>{});
        } else if (mh) {
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::CopyTile<
                    cb_x,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::CopyTileReconfig::Input>{},
                compute_kernel_lib::CopyTile<
                    cb_mask_h_w,
                    compute_kernel_lib::Dst::D1,
                    compute_kernel_lib::InputLifecycle::CallerManaged,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::CopyTileReconfig::Input>{},
                compute_kernel_lib::Mask<DataFormat::Float16_b, compute_kernel_lib::Dst::D0>{},
                compute_kernel_lib::Abs<compute_kernel_lib::Dst::D0>{},
                compute_kernel_lib::PackTile<
                    cb_xabs,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::None>{});
        } else if (mw) {
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::CopyTile<
                    cb_x,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::CopyTileReconfig::Input>{},
                compute_kernel_lib::CopyTile<
                    cb_mask_h_w,
                    compute_kernel_lib::Dst::D1,
                    compute_kernel_lib::InputLifecycle::CallerManaged,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::CopyTileReconfig::Input,
                    compute_kernel_lib::TileOffset::Set>{},
                compute_kernel_lib::Mask<DataFormat::Float16_b, compute_kernel_lib::Dst::D0>{},
                compute_kernel_lib::Abs<compute_kernel_lib::Dst::D0>{},
                compute_kernel_lib::PackTile<
                    cb_xabs,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::None>{});
        } else {
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::CopyTile<
                    cb_x,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::CopyTileReconfig::Input>{},
                compute_kernel_lib::Abs<compute_kernel_lib::Dst::D0>{},
                compute_kernel_lib::PackTile<
                    cb_xabs,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::None>{});
        }

        // |x + decimal|^p
        // Inline power_tile_to_cb body as 4 eltwise_chain stages
        // (see moreh_clip_grad_norm_step2 51cffeb6f03 for the same pattern).
        //   A: |x|^p  (InputLifecycle::HeldStream cb_xabs + PowerIterative(p) + [Recip] + PackTile<cb_xpow>)
        //   B: log(|x|) (InputLifecycle::NoWaitPop cb_xabs + Log + PackTile<cb_logx>)
        //   C: exp(log*decimal) (BinaryFpu Mul + Exp + PackTile<cb_exp_lxmd>)
        //   D: xpow * exp_lxmd -> cb_correct_xpow
        // cb_decimal InputLifecycle::CallerManaged (pre-waited at top of kernel).
        if (p_is_negative) {
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::CopyTile<
                    cb_xabs,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::InputLifecycle::HeldStream,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::CopyTileReconfig::Input>{},
                compute_kernel_lib::PowerIterative<compute_kernel_lib::Dst::D0>{p},
                compute_kernel_lib::Recip<compute_kernel_lib::Dst::D0>{},
                compute_kernel_lib::PackTile<
                    cb_xpow,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::Output>{});
        } else {
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::CopyTile<
                    cb_xabs,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::InputLifecycle::HeldStream,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::CopyTileReconfig::Input>{},
                compute_kernel_lib::PowerIterative<compute_kernel_lib::Dst::D0>{p},
                compute_kernel_lib::PackTile<
                    cb_xpow,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::Output>{});
        }
        compute_kernel_lib::eltwise_chain(
            onetile,
            compute_kernel_lib::CopyTile<
                cb_xabs,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::InputLifecycle::NoWaitPop,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::CopyTileReconfig::Input>{},
            compute_kernel_lib::Log<compute_kernel_lib::Approx::Exact, compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::PackTile<
                cb_logx,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OutputLifecycle::Streaming,
                compute_kernel_lib::PackTileReconfig::Output>{});
        compute_kernel_lib::eltwise_chain(
            onetile,
            compute_kernel_lib::BinaryFpu<
                cb_logx,
                cb_decimal,
                compute_kernel_lib::BinaryFpuOp::Mul,
                compute_kernel_lib::BroadcastDim::None,
                compute_kernel_lib::BinaryDataFormatReconfig::Input,
                compute_kernel_lib::InputLifecycle::Streaming,
                compute_kernel_lib::InputLifecycle::CallerManaged,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OperandKind::Scalar>{},
            compute_kernel_lib::Exp<
                compute_kernel_lib::Approx::Exact,
                compute_kernel_lib::Approx::Exact,
                compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::PackTile<
                cb_exp_lxmd,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OutputLifecycle::Streaming,
                compute_kernel_lib::PackTileReconfig::Output>{});
        compute_kernel_lib::eltwise_chain(
            onetile,
            compute_kernel_lib::BinaryFpu<
                cb_xpow,
                cb_exp_lxmd,
                compute_kernel_lib::BinaryFpuOp::Mul,
                compute_kernel_lib::BroadcastDim::None,
                compute_kernel_lib::BinaryDataFormatReconfig::Input,
                compute_kernel_lib::InputLifecycle::Streaming,
                compute_kernel_lib::InputLifecycle::Streaming,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OperandKind::Scalar>{},
            compute_kernel_lib::PackTile<
                cb_correct_xpow,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OutputLifecycle::Streaming,
                compute_kernel_lib::PackTileReconfig::Output>{});

        if (tile_idx == 0) {
            // Seed cb_xpowadd with first cb_correct_xpow tile.
            // Original: copy_tile_init(cb_correct_xpow) reconfigs srca; pack_tile has no
            // pack reconfig (pack is set to cb_y at startup).
            compute_kernel_lib::copy<
                cb_correct_xpow,
                cb_xpowadd,
                compute_kernel_lib::CopyTileReconfig::Input,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::InputLifecycle::Streaming,
                compute_kernel_lib::OutputLifecycle::Streaming,
                compute_kernel_lib::PackTileReconfig::None>(onetile);
        } else {
            // cb_xpowadd = cb_correct_xpow + cb_xpowadd (in-place accumulator).
            // Original: add_tiles_init reconfigs srca/srcb; pack_tile no reconfig.
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::BinaryFpu<
                    cb_correct_xpow,
                    cb_xpowadd,
                    compute_kernel_lib::BinaryFpuOp::Add,
                    compute_kernel_lib::BroadcastDim::None,
                    compute_kernel_lib::BinaryDataFormatReconfig::Input,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OperandKind::Scalar>{},
                compute_kernel_lib::PackTile<
                    cb_xpowadd,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::None>{});
        }
    }

    // Compute cb_y - reduce single pre-accumulated tile to scalar
    compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM>(
        cb_xpowadd, cb_one, cb_y, compute_kernel_lib::ReduceInputBlockShape::single());

    cb_pop_front(cb_decimal, onetile);
    cb_pop_front(cb_one, onetile);
    if (do_mask_h || do_mask_w) {
        cb_pop_front(cb_mask_h_w, 2);
    }
}
