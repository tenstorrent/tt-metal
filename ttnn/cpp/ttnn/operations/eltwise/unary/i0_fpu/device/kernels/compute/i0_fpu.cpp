// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    constexpr auto cb_input = tt::CBIndex::c_0;       // input
    constexpr auto cb_in_squared = tt::CBIndex::c_1;  // input_squared
    constexpr auto cb_output = tt::CBIndex::c_2;      // output
    constexpr auto cb_coeff0 = tt::CBIndex::c_3;      // coeff0
    constexpr auto cb_coeff1 = tt::CBIndex::c_4;      // coeff1
    constexpr auto cb_coeff2 = tt::CBIndex::c_5;      // coeff2
    constexpr auto cb_coeff3 = tt::CBIndex::c_6;      // coeff3
    constexpr auto cb_coeff4 = tt::CBIndex::c_7;      // coeff4
    constexpr auto cb_coeff5 = tt::CBIndex::c_8;      // coeff5
    constexpr auto cb_coeff6 = tt::CBIndex::c_9;      // coeff6
    constexpr auto cb_coeff7 = tt::CBIndex::c_10;     // coeff7
    constexpr auto cb_coeff8 = tt::CBIndex::c_11;     // coeff8
    constexpr auto cb_coeff9 = tt::CBIndex::c_12;     // coeff9
    constexpr auto cb_coeff10 = tt::CBIndex::c_13;    // coeff10
    constexpr auto cb_one = tt::CBIndex::c_14;        // one

    constexpr uint32_t one = 0x3f800000u;  //  1.0f

    init_sfpu(cb_input, cb_output);
    // binop_with_scalar_tile_init();

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(cb_output, per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            // input_squared = input * input
            cb_reserve_back(cb_in_squared, 1);
            cb_wait_front(cb_input, 1);

            // DPRINT << "input " << TSLICE(tt::CBIndex::c_0, 0, SliceRange::h0_w0_32()) << ENDL();
            tile_regs_acquire();

            mul_tiles_init(cb_input, cb_input);
            mul_tiles(cb_input, cb_input, 0, 0, 0);

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_in_squared);

            tile_regs_release();

            cb_push_back(cb_in_squared, 1);
            cb_pop_front(cb_input, 1);  // pop input tile

            // coef10 * t4
            cb_wait_front(cb_coeff10, 1);
            cb_wait_front(cb_in_squared, 1);

            // DPRINT << "cb_in_squared " << TSLICE(tt::CBIndex::c_1, 0, SliceRange::h0_w0_32()) << ENDL();
            // DPRINT << "cb_coeff10 " << TSLICE(tt::CBIndex::c_13, 0, SliceRange::h0_w0_32()) << ENDL();

            tile_regs_acquire();
            mul_tiles_init(cb_in_squared, cb_coeff10);
            mul_tiles(cb_in_squared, cb_coeff10, 0, 0, 0);

            // (coef9 + coef10 * t4)
            cb_wait_front(cb_coeff9, 1);
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_coeff9);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_coeff9, 0, 0);

            // (coef9 + coef10 * t4) * t4
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_in_squared);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_in_squared, 0, 0);
            // (coef8 + (coef9 + coef10 * t4) * t4)
            cb_wait_front(cb_coeff8, 1);
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_coeff8);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_coeff8, 0, 0);

            // (coef8 + (coef9 + coef10 * t4) * t4) * t4
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_in_squared);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_in_squared, 0, 0);

            // (coef7 + (coef8 + (coef9 + coef10 * t4) * t4) * t4)
            cb_wait_front(cb_coeff7, 1);
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_coeff7);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_coeff7, 0, 0);

            // (coef7 + (coef8 + (coef9 + coef10 * t4) * t4) * t4) * t4)
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_in_squared);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_in_squared, 0, 0);

            // (coef6 + (coef7 + (coef8 + (coef9 + coef10 * t4) * t4) * t4) * t4)
            cb_wait_front(cb_coeff6, 1);
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_coeff6);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_coeff6, 0, 0);

            // (coef6 + (coef7 + (coef8 + (coef9 + coef10 * t4) * t4) * t4) * t4) * t4)
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_in_squared);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_in_squared, 0, 0);

            // (coef5 + (coef6 + (coef7 + (coef8 + (coef9 + coef10 * t4) * t4) * t4) * t4) * t4)
            cb_wait_front(cb_coeff5, 1);
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_coeff5);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_coeff5, 0, 0);

            // (coef5 + (coef6 + (coef7 + (coef8 + (coef9 + coef10 * t4) * t4) * t4) * t4) * t4) * t4)
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_in_squared);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_in_squared, 0, 0);

            // (coef4 + (coef5 + (coef6 + (coef7 + (coef8 + (coef9 + coef10 * t4) * t4) * t4) * t4) * t4) * t4)
            cb_wait_front(cb_coeff4, 1);
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_coeff4);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_coeff4, 0, 0);

            // (coef4 + (coef5 + (coef6 + (coef7 + (coef8 + (coef9 + coef10 * t4) * t4) * t4) * t4) * t4) * t4) * t4)
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_in_squared);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_in_squared, 0, 0);

            // (coef3 + (coef4 + (coef5 + (coef6 + (coef7 + (coef8 + (coef9 + coef10 * t4) * t4) * t4) * t4) * t4) * t4)
            // * t4)
            cb_wait_front(cb_coeff3, 1);
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_coeff3);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_coeff3, 0, 0);

            // (coef3 +  (coef4 + (coef5 + (coef6 + (coef7 + (coef8 + (coef9 + coef10 * t4) * t4) * t4) * t4) * t4) *
            // t4) * t4) * t4) *
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_in_squared);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_in_squared, 0, 0);

            // (coef2 +   (coef3 +  (coef4 + (coef5 + (coef6 + (coef7 + (coef8 + (coef9 + coef10 * t4) * t4) * t4) * t4)
            // * t4) * t4) * t4) * t4) *
            cb_wait_front(cb_coeff2, 1);
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_coeff2);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_coeff2, 0, 0);

            // (coef2 +   (coef3 +  (coef4 + (coef5 + (coef6 + (coef7 + (coef8 + (coef9 + coef10 * t4) * t4) * t4) * t4)
            // * t4) * t4) * t4) * t4) * t4) *
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_in_squared);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_in_squared, 0, 0);

            // (coef1 +  (coef2 +   (coef3 +  (coef4 + (coef5 + (coef6 + (coef7 + (coef8 + (coef9 + coef10 * t4) * t4) *
            // t4) * t4) * t4) * t4) * t4) * t4) * t4) *
            cb_wait_front(cb_coeff1, 1);
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_coeff1);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_coeff1, 0, 0);

            // (coef1 +  (coef2 +   (coef3 +  (coef4 + (coef5 + (coef6 + (coef7 + (coef8 + (coef9 + coef10 * t4) * t4) *
            // t4) * t4) * t4) * t4) * t4) * t4) * t4) * t4) *
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_in_squared);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_in_squared, 0, 0);

            // (coef0 +  (coef1 +  (coef2 +   (coef3 +  (coef4 + (coef5 + (coef6 + (coef7 + (coef8 + (coef9 + coef10 *
            // t4) * t4) * t4) * t4) * t4) * t4) * t4) * t4) * t4) * t4) *
            cb_wait_front(cb_coeff0, 1);
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_coeff0);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_coeff0, 0, 0);

            // ((coef0 +  (coef1 +  (coef2 +   (coef3 +  (coef4 + (coef5 + (coef6 + (coef7 + (coef8 + (coef9 + coef10 *
            // t4) * t4) * t4) * t4) * t4) * t4) * t4) * t4) * t4) * t4) * t4)
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_in_squared);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_in_squared, 0, 0);

            // result + 1.0f
            cb_wait_front(cb_one, 1);
            // DPRINT << "cb_one " << TSLICE(tt::CBIndex::c_14, 0, SliceRange::h0_w0_32()) << ENDL();
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_one);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_one, 0, 0);

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_output);

            // DPRINT << "out" << TSLICE(tt::CBIndex::c_2, 0, SliceRange::h0_w0_32()) << ENDL();

            tile_regs_release();

            cb_pop_front(cb_in_squared, 1);
        }
        cb_push_back(cb_output, per_core_block_dim);
    }
}
}  // namespace NAMESPACE
