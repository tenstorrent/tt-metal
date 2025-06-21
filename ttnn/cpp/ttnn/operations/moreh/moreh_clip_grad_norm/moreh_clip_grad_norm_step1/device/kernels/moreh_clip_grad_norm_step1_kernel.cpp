// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

ALWI bool need_to_do_mask_h(uint32_t tile_idx, uint32_t ht, uint32_t wt) { return (((tile_idx / wt) + 1) % ht) == 0; }

namespace NAMESPACE {
void MAIN {
    int i{0};
    const auto num_tiles = get_arg_val<uint32_t>(i++);
    const auto p = get_arg_val<uint32_t>(i++);
    const bool p_is_negative = get_arg_val<uint32_t>(i++) == 1;
    const auto origin_h = get_arg_val<uint32_t>(i++);
    const auto origin_w = get_arg_val<uint32_t>(i++);

    std::uint8_t input_id{0};
    const auto cb_x = input_id++;         // input(==x)
    const auto cb_one = input_id++;       // one
    const auto cb_decimal = input_id++;   // decimal
    const auto cb_mask_h_w = input_id++;  // mask_h_w

    std::uint8_t output_id{16};
    const auto cb_y = output_id++;  // output(==y)

    std::uint8_t intermed_id{24};
    const auto cb_xabs = intermed_id++;          // |x|
    const auto cb_xpow = intermed_id++;          // |x|^p
    const auto cb_xpowadd = intermed_id++;       // Add[|x|^p * exp(log(|x|) * decimal)]
    const auto cb_logx = intermed_id++;          // log(|x|)
    const auto cb_exp_lxmd = intermed_id++;      // exp(log(|x|) * decimal)
    const auto cb_correct_xpow = intermed_id++;  // |x|^p * exp(log(|x|) * decimal)

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
            tile_regs_acquire();
            cb_wait_front(cb_correct_xpow, onetile);
            cb_reserve_back(cb_xpowadd, onetile);

            copy_tile_init(cb_correct_xpow);
            copy_tile(cb_correct_xpow, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(dst0, cb_xpowadd);

            cb_pop_front(cb_correct_xpow, onetile);
            cb_push_back(cb_xpowadd, onetile);
            tile_regs_release();
        } else {
            tile_regs_acquire();
            cb_wait_front(cb_correct_xpow, onetile);
            cb_wait_front(cb_xpowadd, onetile);
            cb_reserve_back(cb_xpowadd, onetile);

            add_tiles_init(cb_correct_xpow, cb_xpowadd);
            add_tiles(cb_correct_xpow, cb_xpowadd, 0, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(dst0, cb_xpowadd);

            cb_pop_front(cb_correct_xpow, onetile);
            cb_pop_front(cb_xpowadd, onetile);
            cb_push_back(cb_xpowadd, onetile);
            tile_regs_release();
        }
    }

    // Compute cb_y
    tile_regs_acquire();
    cb_wait_front(cb_xpowadd, onetile);
    cb_reserve_back(cb_y, onetile);

    reduce_init(cb_xpowadd, cb_one, cb_y);
    reduce_tile(cb_xpowadd, cb_one, 0, 0, dst0);
    reduce_uninit();
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(dst0, cb_y);

    cb_pop_front(cb_xpowadd, onetile);
    cb_push_back(cb_y, onetile);
    tile_regs_release();

    cb_pop_front(cb_decimal, onetile);
    cb_pop_front(cb_one, onetile);
    if (do_mask_h || do_mask_w) {
        cb_pop_front(cb_mask_h_w, 2);
    }

}  // void MAIN
}  // namespace NAMESPACE
