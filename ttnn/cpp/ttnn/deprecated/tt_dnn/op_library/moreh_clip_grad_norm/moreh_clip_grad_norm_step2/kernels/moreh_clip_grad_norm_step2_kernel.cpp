// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

namespace NAMESPACE {
void MAIN {
    int i{0};
    const auto num_tiles = get_arg_val<uint32_t>(i++);
    const auto p = get_arg_val<uint32_t>(i++);
    const bool p_is_negative = get_arg_val<uint32_t>(i++) == 1;
    const auto norm_type = get_arg_val<uint32_t>(i++);

    std::uint8_t input_id{0};
    const auto cb_input = input_id++;    // input(==tmp_pow_sum)
    const auto cb_decimal = input_id++;  // decimal

    std::uint8_t output_id{16};
    // x^p * exp(log(x) * decimal)
    const auto cb_y = output_id++;  // output(==total_norm)

    std::uint8_t intermed_id{24};
    const auto cb_x = intermed_id++;         // Sum[tmp_pow_sum](==x)
    const auto cb_xpow = intermed_id++;      // x^p
    const auto cb_logx = intermed_id++;      // log(x)
    const auto cb_exp_lxmd = intermed_id++;  // exp(log(x) * decimal)

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    if (num_tiles > 1) {
        binary_op_init_common(cb_input, cb_x);
    } else {
        binary_op_init_common(cb_logx, cb_decimal);
    }

    cb_wait_front(cb_decimal, onetile);  // comes from the reader

    // Compute cb_x
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        if (tile_idx == 0) {
            ACQ();
            cb_wait_front(cb_input, onetile);  // comes from the reader
            cb_reserve_back(cb_x, onetile);

            copy_tile_init();
            copy_tile(cb_input, 0, dst0);

            pack_tile(dst0, cb_x);

            cb_pop_front(cb_input, onetile);
            cb_push_back(cb_x, onetile);
            REL();
        } else {
            ACQ();
            cb_wait_front(cb_input, onetile);  // comes from the reader
            cb_wait_front(cb_x, onetile);
            cb_reserve_back(cb_x, onetile);

            add_tiles_init();
            add_tiles(cb_input, cb_x, 0, 0, dst0);

            pack_tile(dst0, cb_x);

            cb_pop_front(cb_input, onetile);
            cb_pop_front(cb_x, onetile);
            cb_push_back(cb_x, onetile);
            REL();
        }
    }

    // x^p
    power_tile_to_cb(cb_x, cb_xpow, cb_logx, cb_decimal, cb_exp_lxmd, cb_y, p, p_is_negative);

}  // void MAIN
}  // namespace NAMESPACE
