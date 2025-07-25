// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"

#include "debug/dprint.h"

void kernel_main() {
    uint32_t num_tiles_per_core = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    const uint32_t coeff0 = get_compile_time_arg_val(1);
    const uint32_t coeff1 = get_compile_time_arg_val(2);
    const uint32_t coeff2 = get_compile_time_arg_val(3);
    const uint32_t coeff3 = get_compile_time_arg_val(4);
    const uint32_t coeff4 = get_compile_time_arg_val(5);
    const uint32_t coeff5 = get_compile_time_arg_val(6);
    const uint32_t coeff6 = get_compile_time_arg_val(7);
    const uint32_t coeff7 = get_compile_time_arg_val(8);
    const uint32_t coeff8 = get_compile_time_arg_val(9);
    const uint32_t coeff9 = get_compile_time_arg_val(10);
    const uint32_t coeff10 = get_compile_time_arg_val(11);
    const uint32_t one_scalar = get_compile_time_arg_val(12);

    constexpr auto cb_coeff0 = tt::CBIndex::c_3;
    constexpr auto cb_coeff1 = tt::CBIndex::c_4;
    constexpr auto cb_coeff2 = tt::CBIndex::c_5;
    constexpr auto cb_coeff3 = tt::CBIndex::c_6;
    constexpr auto cb_coeff4 = tt::CBIndex::c_7;
    constexpr auto cb_coeff5 = tt::CBIndex::c_8;
    constexpr auto cb_coeff6 = tt::CBIndex::c_9;
    constexpr auto cb_coeff7 = tt::CBIndex::c_10;
    constexpr auto cb_coeff8 = tt::CBIndex::c_11;
    constexpr auto cb_coeff9 = tt::CBIndex::c_12;
    constexpr auto cb_coeff10 = tt::CBIndex::c_13;
    constexpr auto cb_one = tt::CBIndex::c_14;

    cb_push_back(cb_id_in0, num_tiles_per_core);

    constexpr uint32_t onetile = 1;

    cb_reserve_back(cb_coeff0, onetile);
    fill_with_val_bfloat16(cb_coeff0, coeff0);
    cb_push_back(cb_coeff0, onetile);
    // coeff1
    cb_reserve_back(cb_coeff1, onetile);
    fill_with_val_bfloat16(cb_coeff1, coeff1);
    cb_push_back(cb_coeff1, onetile);
    // coeff2
    cb_reserve_back(cb_coeff2, onetile);
    fill_with_val_bfloat16(cb_coeff2, coeff2);
    cb_push_back(cb_coeff2, onetile);
    // coeff3
    cb_reserve_back(cb_coeff3, onetile);
    fill_with_val_bfloat16(cb_coeff3, coeff3);
    cb_push_back(cb_coeff3, onetile);
    // coeff4
    cb_reserve_back(cb_coeff4, onetile);
    fill_with_val_bfloat16(cb_coeff4, coeff4);
    cb_push_back(cb_coeff4, onetile);
    // coeff5
    cb_reserve_back(cb_coeff5, onetile);
    fill_with_val_bfloat16(cb_coeff5, coeff5);
    cb_push_back(cb_coeff5, onetile);
    // coeff6
    cb_reserve_back(cb_coeff6, onetile);
    fill_with_val_bfloat16(cb_coeff6, coeff6);
    cb_push_back(cb_coeff6, onetile);
    // coeff7
    cb_reserve_back(cb_coeff7, onetile);
    fill_with_val_bfloat16(cb_coeff7, coeff7);
    cb_push_back(cb_coeff7, onetile);
    // coeff8
    cb_reserve_back(cb_coeff8, onetile);
    fill_with_val_bfloat16(cb_coeff8, coeff8);
    cb_push_back(cb_coeff8, onetile);
    // coeff9
    cb_reserve_back(cb_coeff9, onetile);
    fill_with_val_bfloat16(cb_coeff9, coeff9);
    cb_push_back(cb_coeff9, onetile);
    // coeff10
    cb_reserve_back(cb_coeff10, onetile);
    fill_with_val_bfloat16(cb_coeff10, coeff10);
    cb_push_back(cb_coeff10, onetile);
    // one scalar
    cb_reserve_back(cb_one, onetile);
    fill_with_val_bfloat16(cb_one, one_scalar);
    cb_push_back(cb_one, onetile);
}
