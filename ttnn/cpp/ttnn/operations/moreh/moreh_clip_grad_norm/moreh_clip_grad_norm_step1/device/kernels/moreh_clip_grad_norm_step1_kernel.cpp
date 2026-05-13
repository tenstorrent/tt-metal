// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
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
        // Comput cb_xabs and mask(optional)
        // |x|
        tile_regs_acquire();
        cb_wait_front(cb_x, onetile);  // comes from the reader
        cb_reserve_back(cb_xabs, onetile);

        copy_tile_init(cb_x);
        copy_tile(cb_x, 0, dst0);

        if (do_mask_h && need_to_do_mask_h(tile_idx, ht, wt)) {
            copy_tile_init(cb_mask_h_w);
            copy_tile(cb_mask_h_w, 0, dst1);

            mask_tile_init();
            mask_tile(dst0, dst1);
        }

        if (do_mask_w && ((tile_idx + 1) % wt) == 0) {
            copy_tile_init(cb_mask_h_w);
            copy_tile(cb_mask_h_w, 1, dst1);

            mask_tile_init();
            mask_tile(dst0, dst1);
        }

        abs_tile_init();
        abs_tile(dst0);
        cb_pop_front(cb_x, onetile);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(dst0, cb_xabs);
        cb_push_back(cb_xabs, onetile);
        tile_regs_release();

        // |x + decimal|^p
        power_tile_to_cb(cb_xabs, cb_xpow, cb_logx, cb_decimal, cb_exp_lxmd, cb_correct_xpow, p, p_is_negative);

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
