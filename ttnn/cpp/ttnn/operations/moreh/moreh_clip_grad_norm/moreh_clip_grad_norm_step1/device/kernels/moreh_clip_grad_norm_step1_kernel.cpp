// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"

ALWI bool need_to_do_mask_h(uint32_t tile_idx, uint32_t ht, uint32_t wt) { return (((tile_idx / wt) + 1) % ht) == 0; }

void kernel_main() {
    int i{0};
    const auto num_tiles = get_arg_val<uint32_t>(i++);
    const auto p = get_arg_val<uint32_t>(i++);
    const bool p_is_negative = get_arg_val<uint32_t>(i++) == 1;
    const auto origin_h = get_arg_val<uint32_t>(i++);
    const auto origin_w = get_arg_val<uint32_t>(i++);

    constexpr std::uint8_t input_id = 0;
    constexpr auto cb_x = input_id + 0;
    DataflowBuffer dfb_x_obj(cb_x);  // input(==x)
    constexpr auto cb_one = input_id + 1;
    DataflowBuffer dfb_one_obj(cb_one);  // one
    constexpr auto cb_decimal = input_id + 2;
    DataflowBuffer dfb_decimal_obj(cb_decimal);  // decimal
    constexpr auto cb_mask_h_w = input_id + 3;
    DataflowBuffer dfb_mask_h_w_obj(cb_mask_h_w);  // mask_h_w

    constexpr std::uint8_t output_id = 16;
    constexpr auto cb_y = output_id;  // output(==y)

    constexpr std::uint8_t intermed_id = 24;
    constexpr auto cb_xabs = intermed_id + 0;
    DataflowBuffer dfb_xabs_obj(cb_xabs);        // |x|
    constexpr auto cb_xpow = intermed_id + 1;    // |x|^p
    DataflowBuffer dfb_xpow_obj(cb_xpow);
    constexpr auto cb_xpowadd = intermed_id + 2;
    DataflowBuffer dfb_xpowadd_obj(cb_xpowadd);  // Add[|x|^p * exp(log(|x|) * decimal)]
    constexpr auto cb_logx = intermed_id + 3;    // log(|x|)
    DataflowBuffer dfb_logx_obj(cb_logx);
    constexpr auto cb_exp_lxmd = intermed_id + 4;  // exp(log(|x|) * decimal)
    DataflowBuffer dfb_exp_lxmd_obj(cb_exp_lxmd);
    constexpr auto cb_correct_xpow = intermed_id + 5;
    DataflowBuffer dfb_correct_xpow_obj(cb_correct_xpow);  // |x|^p * exp(log(|x|) * decimal)

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;

    const bool do_mask_h = (origin_h % TILE_H) != 0;
    const bool do_mask_w = (origin_w % TILE_W) != 0;

    const auto ht = (origin_h + TILE_H - 1) / TILE_H;
    const auto wt = (origin_w + TILE_W - 1) / TILE_W;

    compute_kernel_hw_startup(cb_logx, cb_decimal, cb_y);

    dfb_decimal_obj.wait_front(onetile);  // comes from the reader
    dfb_one_obj.wait_front(onetile);      // comes from the reader

    if (do_mask_h || do_mask_w) {
        dfb_mask_h_w_obj.wait_front(2);  // comes from the reader
    }

    // Compute cb_xpowadd
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        // Comput cb_xabs and mask(optional)
        // |x|
        tile_regs_acquire();
        dfb_x_obj.wait_front(onetile);  // comes from the reader
        dfb_xabs_obj.reserve_back(onetile);

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
        dfb_x_obj.pop_front(onetile);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(dst0, cb_xabs);
        dfb_xabs_obj.push_back(onetile);
        tile_regs_release();

        // |x + decimal|^p
        power_tile_to_cb(
            dfb_xabs_obj,
            dfb_xpow_obj,
            dfb_logx_obj,
            dfb_decimal_obj,
            dfb_exp_lxmd_obj,
            dfb_correct_xpow_obj,
            p,
            p_is_negative);

        if (tile_idx == 0) {
            tile_regs_acquire();
            dfb_correct_xpow_obj.wait_front(onetile);
            dfb_xpowadd_obj.reserve_back(onetile);

            copy_tile_init(cb_correct_xpow);
            copy_tile(cb_correct_xpow, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(dst0, cb_xpowadd);

            dfb_correct_xpow_obj.pop_front(onetile);
            dfb_xpowadd_obj.push_back(onetile);
            tile_regs_release();
        } else {
            tile_regs_acquire();
            dfb_correct_xpow_obj.wait_front(onetile);
            dfb_xpowadd_obj.wait_front(onetile);
            dfb_xpowadd_obj.reserve_back(onetile);

            add_init(cb_correct_xpow, cb_xpowadd);
            add_tiles(cb_correct_xpow, cb_xpowadd, 0, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(dst0, cb_xpowadd);

            dfb_correct_xpow_obj.pop_front(onetile);
            dfb_xpowadd_obj.pop_front(onetile);
            dfb_xpowadd_obj.push_back(onetile);
            tile_regs_release();
        }
    }

    // Compute cb_y - reduce single pre-accumulated tile to scalar
    compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, cb_xpowadd, cb_one, cb_y>(
        compute_kernel_lib::ReduceInputBlockShape::single());

    dfb_decimal_obj.pop_front(onetile);
    dfb_one_obj.pop_front(onetile);
    if (do_mask_h || do_mask_w) {
        dfb_mask_h_w_obj.pop_front(2);
    }
}
