// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "experimental/circular_buffer.h"

ALWI bool need_to_do_mask_h(uint32_t tile_idx, uint32_t ht, uint32_t wt) { return (((tile_idx / wt) + 1) % ht) == 0; }

void kernel_main() {
    int i{0};
    const auto num_tiles = get_arg_val<uint32_t>(i++);
    const auto p = get_arg_val<uint32_t>(i++);
    const bool p_is_negative = get_arg_val<uint32_t>(i++) == 1;
    const auto origin_h = get_arg_val<uint32_t>(i++);
    const auto origin_w = get_arg_val<uint32_t>(i++);

    std::uint8_t input_id{0};
    const auto cb_x = input_id++;
    experimental::CircularBuffer cb_x_obj(cb_x);  // input(==x)
    const auto cb_one = input_id++;
    experimental::CircularBuffer cb_one_obj(cb_one);  // one
    const auto cb_decimal = input_id++;
    experimental::CircularBuffer cb_decimal_obj(cb_decimal);  // decimal
    const auto cb_mask_h_w = input_id++;
    experimental::CircularBuffer cb_mask_h_w_obj(cb_mask_h_w);  // mask_h_w

    std::uint8_t output_id{16};
    const auto cb_y = output_id++;  // output(==y)

    std::uint8_t intermed_id{24};
    const auto cb_xabs = intermed_id++;
    experimental::CircularBuffer cb_xabs_obj(cb_xabs);  // |x|
    const auto cb_xpow = intermed_id++;          // |x|^p
    const auto cb_xpowadd = intermed_id++;
    experimental::CircularBuffer cb_xpowadd_obj(cb_xpowadd);  // Add[|x|^p * exp(log(|x|) * decimal)]
    const auto cb_logx = intermed_id++;          // log(|x|)
    const auto cb_exp_lxmd = intermed_id++;      // exp(log(|x|) * decimal)
    const auto cb_correct_xpow = intermed_id++;
    experimental::CircularBuffer cb_correct_xpow_obj(cb_correct_xpow);  // |x|^p * exp(log(|x|) * decimal)

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

    cb_decimal_obj.wait_front(onetile);  // comes from the reader
    cb_one_obj.wait_front(onetile);      // comes from the reader

    if (do_mask_h || do_mask_w) {
        cb_mask_h_w_obj.wait_front(2);  // comes from the reader
    }

    // Compute cb_xpowadd
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        // Comput cb_xabs and mask(optional)
        // |x|
        tile_regs_acquire();
        cb_x_obj.wait_front(onetile);  // comes from the reader
        cb_xabs_obj.reserve_back(onetile);

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
        cb_x_obj.pop_front(onetile);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(dst0, cb_xabs);
        cb_xabs_obj.push_back(onetile);
        tile_regs_release();

        // |x + decimal|^p
        power_tile_to_cb(cb_xabs, cb_xpow, cb_logx, cb_decimal, cb_exp_lxmd, cb_correct_xpow, p, p_is_negative);

        if (tile_idx == 0) {
            tile_regs_acquire();
            cb_correct_xpow_obj.wait_front(onetile);
            cb_xpowadd_obj.reserve_back(onetile);

            copy_tile_init(cb_correct_xpow);
            copy_tile(cb_correct_xpow, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(dst0, cb_xpowadd);

            cb_correct_xpow_obj.pop_front(onetile);
            cb_xpowadd_obj.push_back(onetile);
            tile_regs_release();
        } else {
            tile_regs_acquire();
            cb_correct_xpow_obj.wait_front(onetile);
            cb_xpowadd_obj.wait_front(onetile);
            cb_xpowadd_obj.reserve_back(onetile);

            add_tiles_init(cb_correct_xpow, cb_xpowadd);
            add_tiles(cb_correct_xpow, cb_xpowadd, 0, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(dst0, cb_xpowadd);

            cb_correct_xpow_obj.pop_front(onetile);
            cb_xpowadd_obj.pop_front(onetile);
            cb_xpowadd_obj.push_back(onetile);
            tile_regs_release();
        }
    }

    // Compute cb_y - reduce single pre-accumulated tile to scalar
    compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM>(
        cb_xpowadd, cb_one, cb_y, compute_kernel_lib::ReduceInputBlockShape::single());

    cb_decimal_obj.pop_front(onetile);
    cb_one_obj.pop_front(onetile);
    if (do_mask_h || do_mask_w) {
        cb_mask_h_w_obj.pop_front(2);
    }
}
