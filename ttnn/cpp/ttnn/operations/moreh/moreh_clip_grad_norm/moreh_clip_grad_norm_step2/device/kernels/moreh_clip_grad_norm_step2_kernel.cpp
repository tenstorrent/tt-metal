// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"

void kernel_main() {
    int i{0};
    const auto num_tiles = get_arg_val<uint32_t>(i++);
    const auto p = get_arg_val<uint32_t>(i++);
    const bool p_is_negative = get_arg_val<uint32_t>(i++) == 1;

    constexpr uint32_t cb_input = 0;    // input(==tmp_pow_sum)
    constexpr uint32_t cb_decimal = 1;  // decimal

    // x^p * exp(log(x) * decimal)
    constexpr uint32_t cb_y = 16;  // output(==total_norm)

    constexpr uint32_t cb_x = 24;         // Sum[tmp_pow_sum](==x)
    constexpr uint32_t cb_xpow = 25;      // x^p
    constexpr uint32_t cb_logx = 26;      // log(x)
    constexpr uint32_t cb_exp_lxmd = 27;  // exp(log(x) * decimal)

    constexpr uint32_t onetile = 1;

    if (num_tiles > 1) {
        binary_op_init_common(cb_input, cb_x, cb_y);
    } else {
        binary_op_init_common(cb_logx, cb_decimal, cb_y);
    }

    cb_wait_front(cb_decimal, onetile);  // comes from the reader

    // Compute cb_x — migrated to eltwise_chain.
    //   tile_idx == 0: copy cb_input -> cb_x (seed accumulator).
    //   tile_idx  > 0: cb_x = cb_input + cb_x (in-place accumulator).
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        if (tile_idx == 0) {
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::CopyTile<
                    cb_input,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::CopyTilePolicy::WaitAndPop,
                    compute_kernel_lib::CbIndexMode::FirstTile,
                    compute_kernel_lib::CopyTileReconfig::None>{},
                compute_kernel_lib::PackTile<
                    cb_x,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::PackTilePolicy::PerTileReserveAndPush,
                    compute_kernel_lib::PackTileIndexMode::FirstTile,
                    compute_kernel_lib::PackTileReconfig::None>{});
        } else {
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::BinaryFpu<
                    cb_input,
                    cb_x,
                    compute_kernel_lib::BinaryFpuOp::Add,
                    compute_kernel_lib::BroadcastDim::None,
                    compute_kernel_lib::BinaryDataFormatReconfig::Input,
                    compute_kernel_lib::CopyTilePolicy::WaitAndPop,
                    compute_kernel_lib::CopyTilePolicy::WaitAndPop,
                    compute_kernel_lib::CbIndexMode::FirstTile,
                    compute_kernel_lib::Dst::D0>{},
                compute_kernel_lib::PackTile<
                    cb_x,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::PackTilePolicy::PerTileReserveAndPush,
                    compute_kernel_lib::PackTileIndexMode::FirstTile,
                    compute_kernel_lib::PackTileReconfig::Output>{});
        }
    }
    // PARTIAL migration: inline power_tile_to_cb body as 4 eltwise_chain stages.
    //   Block A: x^p           (CopyTile<cb_x, WaitNoPop, CopyTilePolicy::WaitAndPop, CbIndexMode::FirstTile,
    //   CopyTileReconfig::None> + PowerIterative + [Recip] + PackTile<cb_xpow, Dst::D0,
    //   PackTilePolicy::PerTileReserveAndPush, PackTileIndexMode::FirstTile, PackTileReconfig::None>) Block B: log(x)
    //   (CopyTile<cb_x, NoWaitPop, CopyTilePolicy::WaitAndPop, CbIndexMode::FirstTile, CopyTileReconfig::None> + Log +
    //   PackTile<cb_logx, Dst::D0, PackTilePolicy::PerTileReserveAndPush, PackTileIndexMode::FirstTile,
    //   PackTileReconfig::None>) Block C: exp(log(x)*d) (BinaryFpu<cb_logx, cb_decimal, Mul> + Exp +
    //   PackTile<cb_exp_lxmd, Dst::D0, PackTilePolicy::PerTileReserveAndPush, PackTileIndexMode::FirstTile,
    //   PackTileReconfig::None>) Block D: xpow * exp(.) (BinaryFpu<cb_xpow, cb_exp_lxmd, Mul> + PackTile<cb_y, Dst::D0,
    //   PackTilePolicy::PerTileReserveAndPush, PackTileIndexMode::FirstTile, PackTileReconfig::None>)
    {
        using namespace compute_kernel_lib;

        // Block A: x^p (optionally recip). Hold cb_x for Block B.
        if (p_is_negative) {
            eltwise_chain(
                onetile,
                CopyTile<cb_x, Dst::D0, CopyTilePolicy::WaitNoPop, CbIndexMode::FirstTile, CopyTileReconfig::Input>{},
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
                CopyTile<cb_x, Dst::D0, CopyTilePolicy::WaitNoPop, CbIndexMode::FirstTile, CopyTileReconfig::Input>{},
                PowerIterative<Dst::D0>{p},
                PackTile<
                    cb_xpow,
                    Dst::D0,
                    PackTilePolicy::PerTileReserveAndPush,
                    PackTileIndexMode::FirstTile,
                    PackTileReconfig::Output>{});
        }

        // Block B: log(x). cb_x already waited in Block A; pop here.
        eltwise_chain(
            onetile,
            CopyTile<cb_x, Dst::D0, CopyTilePolicy::NoWaitPop, CbIndexMode::FirstTile, CopyTileReconfig::Input>{},
            Log<>{},
            PackTile<
                cb_logx,
                Dst::D0,
                PackTilePolicy::PerTileReserveAndPush,
                PackTileIndexMode::FirstTile,
                PackTileReconfig::Output>{});

        // Block C: exp(log(x) * decimal). cb_decimal pre-waited at top of kernel — NoWaitNoPop.
        eltwise_chain(
            onetile,
            BinaryFpu<
                cb_logx,
                cb_decimal,
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

        // Block D: cb_xpow * cb_exp_lxmd -> cb_y.
        eltwise_chain(
            onetile,
            BinaryFpu<
                cb_xpow,
                cb_exp_lxmd,
                BinaryFpuOp::Mul,
                BroadcastDim::None,
                BinaryDataFormatReconfig::Input,
                CopyTilePolicy::WaitAndPop,
                CopyTilePolicy::WaitAndPop,
                CbIndexMode::FirstTile,
                Dst::D0>{},
            PackTile<
                cb_y,
                Dst::D0,
                PackTilePolicy::PerTileReserveAndPush,
                PackTileIndexMode::FirstTile,
                PackTileReconfig::Output>{});
    }
}
