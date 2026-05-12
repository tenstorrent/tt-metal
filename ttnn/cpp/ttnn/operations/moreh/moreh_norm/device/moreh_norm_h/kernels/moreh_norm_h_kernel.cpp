// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "experimental/circular_buffer.h"

void kernel_main() {
    int i{0};
    const auto num_cols_per_core = get_arg_val<uint32_t>(i++);
    const auto Ht = get_arg_val<uint32_t>(i++);
    const auto origin_h = get_arg_val<uint32_t>(i++);
    const auto p = get_arg_val<uint32_t>(i++);
    const bool p_is_negative = get_arg_val<uint32_t>(i++) == 1;
    const auto recip_p = get_arg_val<uint32_t>(i++);
    const bool recip_p_is_negative = get_arg_val<uint32_t>(i++) == 1;

    std::uint8_t input_id{tt::CBIndex::c_0};
    const auto cb_x = input_id++;
    experimental::CircularBuffer cb_x_obj(cb_x);  // input
    const auto cb_one = input_id++;
    experimental::CircularBuffer cb_one_obj(cb_one);  // one
    const auto cb_decimal = input_id++;
    experimental::CircularBuffer cb_decimal_obj(cb_decimal);  // decimal
    const auto cb_recip_p_decimal = input_id++;
    experimental::CircularBuffer cb_recip_p_decimal_obj(cb_recip_p_decimal);  // recip_p_decimal
    const auto cb_mask_h = input_id++;
    experimental::CircularBuffer cb_mask_h_obj(cb_mask_h);  // mask_h

    std::uint8_t output_id{tt::CBIndex::c_16};
    const auto cb_y = output_id++;  // output

    std::uint8_t intermed_id{tt::CBIndex::c_24};
    const auto cb_tmp0 = intermed_id++;
    const auto cb_tmp1 = intermed_id++;
    const auto cb_tmp2 = intermed_id++;
    const auto cb_tmp3 = intermed_id++;
    const auto cb_tmp4 = intermed_id++;
    const auto cb_tmp5 = intermed_id++;
    const auto cb_tmp6 = intermed_id++;

    const auto cb_xabs = cb_tmp0;
    experimental::CircularBuffer cb_xabs_obj(cb_xabs);  // |x|
    const auto cb_xpow = cb_tmp1;          // |x|^p
    const auto cb_logx = cb_tmp2;          // log(|x|)
    const auto cb_exp_lxmd = cb_tmp3;      // exp(log(|x|) * decimal)
    const auto cb_correct_xpow = cb_tmp4;
    experimental::CircularBuffer cb_correct_xpow_obj(
        cb_correct_xpow);  // |x|^p * exp(log(|x|) * decimal)(==|x + decimal|^p)
    const auto cb_xpowadd = cb_tmp5;
    experimental::CircularBuffer cb_xpowadd_obj(cb_xpowadd);  // Add(|x + decimal|^p)
    const auto cb_xpowsum = cb_tmp6;       // Sum(|x + decimal|^p)

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_0, tt::CBIndex::c_16);

    cb_one_obj.wait_front(onetile);              // comes from the reader
    cb_decimal_obj.wait_front(onetile);          // comes from the reader
    cb_recip_p_decimal_obj.wait_front(onetile);  // comes from the reader

    constexpr uint32_t TILE_H = 32;
    const bool do_mask_h = (origin_h % TILE_H) != 0;
    const auto mask_h = do_mask_h ? (origin_h % TILE_H) : TILE_H;

    if (do_mask_h) {
        cb_mask_h_obj.wait_front(onetile);  // comes from the reader
    }

    for (uint32_t col_idx = 0; col_idx < num_cols_per_core; ++col_idx) {
        for (uint32_t row_idx = 0; row_idx < Ht; ++row_idx) {
            // |x|
            tile_regs_acquire();
            cb_x_obj.wait_front(onetile);  // comes from the reader
            cb_xabs_obj.reserve_back(onetile);

            copy_tile_init_with_dt(cb_x);
            copy_tile(cb_x, 0, dst0);

            if (do_mask_h && (row_idx == Ht - 1)) {
                copy_tile_init_with_dt(cb_mask_h);
                copy_tile(cb_mask_h, 0, dst1);

                mask_tile_init();
                mask_tile(dst0, dst1);
            }

            abs_tile_init();
            abs_tile(dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_xabs);
            tile_regs_release();

            cb_x_obj.pop_front(onetile);
            cb_xabs_obj.push_back(onetile);

            power_tile_to_cb(cb_xabs, cb_xpow, cb_logx, cb_decimal, cb_exp_lxmd, cb_correct_xpow, p, p_is_negative);

            // Add(|x|^p)
            if (row_idx == 0) {
                tile_regs_acquire();
                cb_correct_xpow_obj.wait_front(onetile);
                cb_xpowadd_obj.reserve_back(onetile);

                copy_tile_init_with_dt(cb_correct_xpow);
                copy_tile(cb_correct_xpow, 0, dst0);
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, cb_xpowadd);
                tile_regs_release();

                cb_correct_xpow_obj.pop_front(onetile);
                cb_xpowadd_obj.push_back(onetile);
            } else {
                tile_regs_acquire();
                cb_correct_xpow_obj.wait_front(onetile);
                cb_xpowadd_obj.wait_front(onetile);
                cb_xpowadd_obj.reserve_back(onetile);

                add_tiles_init_with_dt(cb_correct_xpow, cb_xpowadd);
                add_tiles(cb_correct_xpow, cb_xpowadd, 0, 0, dst0);
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, cb_xpowadd);
                tile_regs_release();

                cb_correct_xpow_obj.pop_front(onetile);
                cb_xpowadd_obj.pop_front(onetile);
                cb_xpowadd_obj.push_back(onetile);
            }
        }
        // Sum(|x|^p) - reduce single pre-accumulated tile
        compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM>(
            cb_xpowadd, cb_one, cb_xpowsum, compute_kernel_lib::ReduceInputBlockShape::single());

        power_tile_to_cb(cb_xpowsum, cb_tmp0, cb_tmp1, cb_recip_p_decimal, cb_tmp2, cb_y, recip_p, recip_p_is_negative);
    }

    cb_one_obj.pop_front(onetile);
    cb_decimal_obj.pop_front(onetile);
    cb_recip_p_decimal_obj.pop_front(onetile);
    if (do_mask_h) {
        cb_mask_h_obj.pop_front(onetile);
    }
}
