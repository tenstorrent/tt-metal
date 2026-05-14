// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"

ALWI bool need_to_do_mask_h(uint32_t tile_idx, uint32_t ht, uint32_t wt) { return (((tile_idx / wt) + 1) % ht) == 0; }

void kernel_main() {
    int i{0};
    const auto num_tiles = get_arg_val<uint32_t>(i++);
    const auto p = get_arg_val<uint32_t>(i++);
    const bool p_is_negative = get_arg_val<uint32_t>(i++) == 1;
    const auto origin_h = get_arg_val<uint32_t>(i++);
    const auto origin_w = get_arg_val<uint32_t>(i++);

    constexpr std::uint32_t cb_x = 0;         // input(==x)
    constexpr std::uint32_t cb_one = 1;       // one
    constexpr std::uint32_t cb_decimal = 2;   // decimal
    constexpr std::uint32_t cb_mask_h_w = 3;  // mask_h_w

    constexpr std::uint32_t cb_y = 16;  // output(==y)

    constexpr std::uint32_t cb_xabs = 24;          // |x|
    constexpr std::uint32_t cb_xpow = 25;          // |x|^p
    constexpr std::uint32_t cb_xpowadd = 26;       // Add[|x|^p * exp(log(|x|) * decimal)]
    constexpr std::uint32_t cb_logx = 27;          // log(|x|)
    constexpr std::uint32_t cb_exp_lxmd = 28;      // exp(log(|x|) * decimal)
    constexpr std::uint32_t cb_correct_xpow = 29;  // |x|^p * exp(log(|x|) * decimal)

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
        // PARTIAL migration: abs+mask+pack block via eltwise_chain. Four runtime
        // branches cover (mask_h?, mask_w?) ∈ {(0,0),(1,0),(0,1),(1,1)}. The mask
        // CB has 2 pre-waited tiles (tile 0 = mask_h, tile 1 = mask_w); we read
        // them via NoWaitNoPop CopyTile elements pinned to fixed indices.
        {
            using namespace compute_kernel_lib;
            const bool mh = do_mask_h && need_to_do_mask_h(tile_idx, ht, wt);
            const bool mw = do_mask_w && ((tile_idx + 1) % wt) == 0;
            if (mh && mw) {
                eltwise_chain(
                    onetile,
                    CopyTile<cb_x, Dst::D0, CopyTilePolicy::WaitAndPop>{},
                    CopyTile<cb_mask_h_w, Dst::D1, CopyTilePolicy::NoWaitNoPop, CbIndexMode::FirstTile>{},
                    Mask<DataFormat::Float16_b, Dst::D0>{},
                    CopyTile<cb_mask_h_w, Dst::D1, CopyTilePolicy::NoWaitNoPop, CbIndexMode::Pinned>{1u},
                    Mask<DataFormat::Float16_b, Dst::D0>{},
                    Abs<Dst::D0>{},
                    PackTile<cb_xabs, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
            } else if (mh) {
                eltwise_chain(
                    onetile,
                    CopyTile<cb_x, Dst::D0, CopyTilePolicy::WaitAndPop>{},
                    CopyTile<cb_mask_h_w, Dst::D1, CopyTilePolicy::NoWaitNoPop, CbIndexMode::FirstTile>{},
                    Mask<DataFormat::Float16_b, Dst::D0>{},
                    Abs<Dst::D0>{},
                    PackTile<cb_xabs, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
            } else if (mw) {
                eltwise_chain(
                    onetile,
                    CopyTile<cb_x, Dst::D0, CopyTilePolicy::WaitAndPop>{},
                    CopyTile<cb_mask_h_w, Dst::D1, CopyTilePolicy::NoWaitNoPop, CbIndexMode::Pinned>{1u},
                    Mask<DataFormat::Float16_b, Dst::D0>{},
                    Abs<Dst::D0>{},
                    PackTile<cb_xabs, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
            } else {
                eltwise_chain(
                    onetile,
                    CopyTile<cb_x, Dst::D0, CopyTilePolicy::WaitAndPop>{},
                    Abs<Dst::D0>{},
                    PackTile<cb_xabs, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
            }
        }

        // PARTIAL migration: inline power_tile_to_cb body as 4 eltwise_chain stages.
        //   Block A: |x|^p          (CopyTile<cb_xabs, WaitNoPop> + PowerIterative + [Recip] + PackTile<cb_xpow>)
        //   Block B: log(|x|)       (CopyTile<cb_xabs, NoWaitPop> + Log + PackTile<cb_logx>)
        //   Block C: exp(log * d)   (BinaryFpu<cb_logx, cb_decimal, cb_exp_lxmd, Mul> + Exp + PackTile<cb_exp_lxmd>)
        //   Block D: xpow*exp       (BinaryFpu<cb_xpow, cb_exp_lxmd, cb_correct_xpow, Mul> + PackTile<cb_correct_xpow>)
        {
            using namespace compute_kernel_lib;
            if (p_is_negative) {
                eltwise_chain(
                    onetile,
                    CopyTile<
                        cb_xabs,
                        Dst::D0,
                        CopyTilePolicy::WaitNoPop,
                        CbIndexMode::FirstTile,
                        CopyTileReconfig::Input>{},
                    PowerIterative<Dst::D0>{p},
                    Recip<Dst::D0>{},
                    PackTile<
                        cb_xpow,
                        Dst::D0,
                        PackTilePolicy::PerTileReserveAndPush,
                        PackTileIndexMode::FirstTile,
                        PackTileReconfig::Output>{});
            } else {
                eltwise_chain(
                    onetile,
                    CopyTile<
                        cb_xabs,
                        Dst::D0,
                        CopyTilePolicy::WaitNoPop,
                        CbIndexMode::FirstTile,
                        CopyTileReconfig::Input>{},
                    PowerIterative<Dst::D0>{p},
                    PackTile<
                        cb_xpow,
                        Dst::D0,
                        PackTilePolicy::PerTileReserveAndPush,
                        PackTileIndexMode::FirstTile,
                        PackTileReconfig::Output>{});
            }

            eltwise_chain(
                onetile,
                CopyTile<
                    cb_xabs,
                    Dst::D0,
                    CopyTilePolicy::NoWaitPop,
                    CbIndexMode::FirstTile,
                    CopyTileReconfig::Input>{},
                Log<>{},
                PackTile<
                    cb_logx,
                    Dst::D0,
                    PackTilePolicy::PerTileReserveAndPush,
                    PackTileIndexMode::FirstTile,
                    PackTileReconfig::Output>{});

            eltwise_chain(
                onetile,
                BinaryFpu<
                    cb_logx,
                    cb_decimal,
                    cb_exp_lxmd,
                    BinaryFpuOp::Mul,
                    BroadcastDim::None,
                    BinaryDataFormatReconfig::Input,
                    CopyTilePolicy::WaitAndPop,
                    CopyTilePolicy::NoWaitNoPop,
                    CbIndexMode::FirstTile,
                    Dst::D0>{},
                Exp<>{},
                PackTile<
                    cb_exp_lxmd,
                    Dst::D0,
                    PackTilePolicy::PerTileReserveAndPush,
                    PackTileIndexMode::FirstTile,
                    PackTileReconfig::Output>{});

            eltwise_chain(
                onetile,
                BinaryFpu<
                    cb_xpow,
                    cb_exp_lxmd,
                    cb_correct_xpow,
                    BinaryFpuOp::Mul,
                    BroadcastDim::None,
                    BinaryDataFormatReconfig::Input,
                    CopyTilePolicy::WaitAndPop,
                    CopyTilePolicy::WaitAndPop,
                    CbIndexMode::FirstTile,
                    Dst::D0>{},
                PackTile<
                    cb_correct_xpow,
                    Dst::D0,
                    PackTilePolicy::PerTileReserveAndPush,
                    PackTileIndexMode::FirstTile,
                    PackTileReconfig::Output>{});
        }

        if (tile_idx == 0) {
            // Seed cb_xpowadd with first cb_correct_xpow tile.
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::CopyTile<
                    cb_correct_xpow,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::CopyTilePolicy::WaitAndPop>{},
                compute_kernel_lib::PackTile<
                    cb_xpowadd,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::PackTilePolicy::PerTileReserveAndPush>{});
        } else {
            // cb_xpowadd = cb_correct_xpow + cb_xpowadd (in-place accumulator)
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::BinaryFpu<
                    cb_correct_xpow,
                    cb_xpowadd,
                    cb_xpowadd,
                    compute_kernel_lib::BinaryFpuOp::Add,
                    compute_kernel_lib::BroadcastDim::None,
                    compute_kernel_lib::BinaryDataFormatReconfig::InputAndOutput,
                    compute_kernel_lib::CopyTilePolicy::WaitAndPop,
                    compute_kernel_lib::CopyTilePolicy::WaitAndPop,
                    compute_kernel_lib::CbIndexMode::FirstTile,
                    compute_kernel_lib::Dst::D0>{},
                compute_kernel_lib::PackTile<
                    cb_xpowadd,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::PackTilePolicy::PerTileReserveAndPush>{});
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
