// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Implemented based on bmm.cpp
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/transpose_wh.h"

namespace NAMESPACE {

////////////////////
// global variables
////////////////////
constexpr int32_t MAX_NUM_DIMENSIONS = 8;
constexpr uint32_t onetile = 1;
constexpr uint32_t num_mask_tiles = 3;
constexpr uint32_t MASK_TILE_H_IDX = 0;
constexpr uint32_t MASK_TILE_W_IDX = 1;
constexpr uint32_t MASK_TILE_HW_IDX = 2;
constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
constexpr uint32_t cb_in2 = tt::CBIndex::c_2;
constexpr uint32_t cb_in3 = tt::CBIndex::c_3;
constexpr uint32_t bias_cb_id = tt::CBIndex::c_4;
constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
constexpr uint32_t cb_intermed0 = tt::CBIndex::c_24;
constexpr uint32_t cb_intermed1 = tt::CBIndex::c_25;
constexpr uint32_t cb_intermed2 = tt::CBIndex::c_26;

////////////////////
// inline functions
////////////////////
FORCE_INLINE void unravel_output_tidx(uint32_t output_tidx, uint32_t* output_idxes, uint32_t* output_stride) {
    for (int32_t i = MAX_NUM_DIMENSIONS - 1; i >= 0; --i) {
        uint32_t dim = output_tidx / output_stride[i];
        output_idxes[i] = dim;
        output_tidx -= (output_idxes[i] * output_stride[i]);
    }
}

// TODO: move it to moreh_common.hpp if more use cases.
FORCE_INLINE void transpose_wh_tile_to_cb(uint32_t icb, uint32_t ocb, uint32_t itile = 0, uint32_t idst = 0) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format_srca(icb);
#endif
    transpose_wh_init_short(icb);
    tile_regs_acquire();
    transpose_wh_tile(icb, itile, idst);
    tile_regs_commit();
    cb_reserve_back(ocb, onetile);
    tile_regs_wait();
#if defined FP32_DEST_ACC_EN
    pack_reconfig_data_format(ocb);
#endif
    pack_tile(idst, ocb);
    tile_regs_release();
    cb_push_back(ocb, onetile);
}

FORCE_INLINE void transpose_tile(uint32_t& mm_src, bool transpose, bool need_mask, bool is_input) {
    if (!transpose) {
        return;
    }

    if (need_mask) {
        cb_wait_front(mm_src, onetile);
        transpose_wh_tile_to_cb(mm_src, mm_src);
        cb_pop_front(mm_src, onetile);
    } else {
        uint32_t trans_src = (is_input) ? (cb_in0) : (cb_in1);
        mm_src = (is_input) ? (cb_intermed1) : (cb_intermed2);
        transpose_wh_tile_to_cb(trans_src, mm_src);
    }
}

FORCE_INLINE void pack_onetile_to_cb(uint32_t ocb = 16, uint32_t idst = 0) {
    cb_reserve_back(ocb, onetile);
    tile_regs_wait();
#if defined FP32_DEST_ACC_EN
    pack_reconfig_data_format(ocb);
#endif
    pack_tile(idst, ocb);
    tile_regs_release();
    cb_push_back(ocb, onetile);
}

FORCE_INLINE void mask_tile_to_cb(
    uint32_t& mm_src,
    bool& need_mask,
    bool need_mask_h,
    bool need_mask_w,
    bool last_out,
    bool last_line,
    bool transpose,
    bool is_input) {
    bool need_mask_last_line_and_out = (last_line && last_out);
    bool need_mask_last_line = false;
    bool need_mask_last_out = false;

    if (!(need_mask_w || need_mask_h)) {
        return;
    }

    if (is_input) {
        need_mask_last_line = last_line && ((transpose) ? (need_mask_w) : (need_mask_h));
        need_mask_last_out = last_out && ((transpose) ? (need_mask_h) : (need_mask_w));
    } else {
        need_mask_last_line = last_line && ((transpose) ? (need_mask_h) : (need_mask_w));
        need_mask_last_out = last_out && ((transpose) ? (need_mask_w) : (need_mask_h));
    }

    if (need_mask_last_line_and_out || need_mask_last_line || need_mask_last_out) {
        uint32_t cb_in = (is_input) ? (cb_in0) : (cb_in1);
        uint32_t cb_mask = (is_input) ? (cb_in2) : (cb_in3);
        uint32_t cb_intermed = (is_input) ? (cb_intermed1) : (cb_intermed2);
        uint32_t mask_tidx = MASK_TILE_H_IDX;
        if (need_mask_last_line_and_out) {
            mask_tidx = MASK_TILE_HW_IDX;
        } else if (need_mask_last_line) {
            if (is_input) {
                mask_tidx = (transpose) ? (MASK_TILE_W_IDX) : (MASK_TILE_H_IDX);
            } else {
                mask_tidx = (transpose) ? (MASK_TILE_H_IDX) : (MASK_TILE_W_IDX);
            }
        } else {
            if (is_input) {
                mask_tidx = (transpose) ? (MASK_TILE_H_IDX) : (MASK_TILE_W_IDX);
            } else {
                mask_tidx = (transpose) ? (MASK_TILE_W_IDX) : (MASK_TILE_H_IDX);
            }
        }

        // mul input tile with mask tile
        tile_regs_acquire();
#if defined FP32_DEST_ACC_EN
        reconfig_data_format(cb_in0, cb_mask);
#endif
        mul_tiles_init(cb_in, cb_mask);
        mul_tiles(cb_in, cb_mask, 0, mask_tidx, 0);
        tile_regs_commit();

        pack_onetile_to_cb(cb_intermed);
        mm_src = cb_intermed;
        need_mask = true;
    }
}

#ifdef FUSE_BIAS
FORCE_INLINE void bias_add(bool is_scalar_bias) {
    static bool scalar_bias_loaded = false;
    pack_onetile_to_cb(cb_intermed0);
    cb_wait_front(cb_intermed0, onetile);

    if (is_scalar_bias && !scalar_bias_loaded) {
        cb_wait_front(bias_cb_id, onetile);
        scalar_bias_loaded = true;
    } else {
        cb_wait_front(bias_cb_id, onetile);
    }

    tile_regs_acquire();
    if (is_scalar_bias) {
#if defined FP32_DEST_ACC_EN
        reconfig_data_format(cb_intermed0, bias_cb_id);
#endif
        add_bcast_scalar_init_short(cb_intermed0, bias_cb_id);
        add_tiles_bcast_scalar(cb_intermed0, bias_cb_id, 0, 0, 0);
    } else {
#if defined FP32_DEST_ACC_EN
        reconfig_data_format(cb_intermed0, bias_cb_id);
#endif
        add_bcast_rows_init_short(cb_intermed0, bias_cb_id);
        add_tiles_bcast_rows(cb_intermed0, bias_cb_id, 0, 0, 0);
    }
    tile_regs_commit();

    cb_pop_front(cb_intermed0, onetile);
    if (!is_scalar_bias) {
        cb_pop_front(bias_cb_id, onetile);
    }
}
#endif

FORCE_INLINE void matmul_with_transpose_and_mask(
    uint32_t output_tidx,
    uint32_t num_output_tiles,
    uint32_t Kt,
    bool transpose_input,
    bool transpose_other,
    bool need_input_mask_h,
    bool need_input_mask_w,
    uint32_t* output_stride,
    uint32_t Mt,
    uint32_t Nt,
    bool need_other_mask_h,
    bool need_other_mask_w,
    bool is_scalar_bias) {
    // TODO: checking required when the input cb format and intermediate cb format are different.
    mm_init(cb_in0, cb_in1, cb_out0);
    if (transpose_input || transpose_other) {
        transpose_wh_init(cb_in0, cb_out0);
    }

    if (need_input_mask_h || need_input_mask_w) {
        cb_wait_front(cb_in2, num_mask_tiles);
    }

    if (need_other_mask_h || need_other_mask_w) {
        cb_wait_front(cb_in3, num_mask_tiles);
    }

#pragma GCC unroll 0
    for (uint32_t i = 0; i < num_output_tiles; ++i) {
        bool spill = Kt > 1;
        bool enable_reload = false;

        // get row and column positions of input and other based on output tile indexes.
        uint32_t output_idxes[MAX_NUM_DIMENSIONS];
        unravel_output_tidx(output_tidx, output_idxes, output_stride);
        bool input_last_row = (output_idxes[1] == Mt - 1) ? (true) : (false);
        bool other_last_col = (output_idxes[0] == Nt - 1) ? (true) : (false);

#pragma GCC unroll 0
        for (uint32_t kt = 0; kt < Kt; kt++) {
            bool last_out = kt == (Kt - 1);
            bool need_input_mask = false;
            bool need_other_mask = false;

            uint32_t mm_src0 = cb_in0;
            uint32_t mm_src1 = cb_in0;

            cb_wait_front(cb_in0, onetile);
            cb_wait_front(cb_in1, onetile);

            mm_src0 = cb_in0;
            mm_src1 = cb_in1;

            ////////////////////
            // mask: the first two arguments (mm_src0, need_input_mask) are passed by reference.
            // transpose: the first argument (mm_src0) is passed by reference.
            ////////////////////
            mask_tile_to_cb(
                mm_src0,
                need_input_mask,
                need_input_mask_h,
                need_input_mask_w,
                last_out,
                input_last_row,
                transpose_input,
                true);
            transpose_tile(mm_src0, transpose_input, need_input_mask, true);

            mask_tile_to_cb(
                mm_src1,
                need_other_mask,
                need_other_mask_h,
                need_other_mask_w,
                last_out,
                other_last_col,
                transpose_other,
                false);
            transpose_tile(mm_src1, transpose_other, need_other_mask, false);

            ////////////////////
            // matmul
            ////////////////////
            tile_regs_acquire();
            if (enable_reload) {
                cb_wait_front(cb_intermed0, onetile);
#if defined FP32_DEST_ACC_EN
                reconfig_data_format_srca(cb_intermed0);
#endif
                copy_tile_to_dst_init_short(cb_intermed0);
                copy_tile(cb_intermed0, 0, 0);
                cb_pop_front(cb_intermed0, onetile);
            }

            if (transpose_input || need_input_mask) {
                cb_wait_front(mm_src0, onetile);
            }

            if (transpose_other || need_other_mask) {
                cb_wait_front(mm_src1, onetile);
            }

            mm_init_short(mm_src0, mm_src1);
#if defined FP32_DEST_ACC_EN
            reconfig_data_format(mm_src0, mm_src1);
#endif
            matmul_tiles(mm_src0, mm_src1, 0, 0, 0, false);
            tile_regs_commit();

            cb_pop_front(cb_in0, onetile);
            cb_pop_front(cb_in1, onetile);

            if (transpose_input || need_input_mask) {
                cb_pop_front(mm_src0, onetile);
            }
            if (transpose_other || need_other_mask) {
                cb_pop_front(mm_src1, onetile);
            }

            if (last_out) {
////////////////////
// bias add
////////////////////
#ifdef FUSE_BIAS
                bias_add(is_scalar_bias);
#endif
                pack_onetile_to_cb(cb_out0);
            } else {
                pack_onetile_to_cb(cb_intermed0);
            }

            if (spill) {
                enable_reload = true;
            }
        }
        output_tidx++;
    }
}

FORCE_INLINE void matmul(uint32_t num_output_tiles, uint32_t Kt) {
    mm_init(cb_in0, cb_in1, cb_out0);
    for (uint32_t i = 0; i < num_output_tiles; ++i) {
        tile_regs_acquire();
        for (uint32_t kt = 0; kt < Kt; kt++) {
            cb_wait_front(cb_in0, onetile);
            cb_wait_front(cb_in1, onetile);
            matmul_tiles(cb_in0, cb_in1, 0, 0, 0, false);
            cb_pop_front(cb_in0, onetile);
            cb_pop_front(cb_in1, onetile);
        }
        tile_regs_commit();
        pack_onetile_to_cb(cb_out0);
    }
}

void MAIN {
    // compile-time args
    constexpr uint32_t num_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t Mt = get_compile_time_arg_val(1);
    constexpr uint32_t Nt = get_compile_time_arg_val(2);
    constexpr uint32_t Kt = get_compile_time_arg_val(3);
    constexpr bool transpose_input = (get_compile_time_arg_val(4) == 1);
    constexpr bool transpose_other = (get_compile_time_arg_val(5) == 1);
    constexpr uint32_t input_mask_h = get_compile_time_arg_val(6);
    constexpr uint32_t input_mask_w = get_compile_time_arg_val(7);
    constexpr uint32_t other_mask_h = get_compile_time_arg_val(8);
    constexpr uint32_t other_mask_w = get_compile_time_arg_val(9);
#ifdef FUSE_BIAS
    constexpr bool is_scalar_bias = (get_compile_time_arg_val(10) == 1);
    constexpr bool need_bias_add = true;
#else
    constexpr bool is_scalar_bias = false;
    constexpr bool need_bias_add = false;
#endif
    constexpr bool need_input_mask_h = (input_mask_h != 32);
    constexpr bool need_input_mask_w = (input_mask_w != 32);
    constexpr bool need_other_mask_h = (other_mask_h != 32);
    constexpr bool need_other_mask_w = (other_mask_w != 32);
    constexpr bool need_mask = (need_input_mask_h || need_input_mask_w || need_other_mask_h || need_other_mask_w);
    constexpr bool need_transpose = (transpose_input || transpose_other);

    // runtime args
    ArgFetcher arg_fetcher;
    uint32_t output_tile_start_idx = arg_fetcher.get_next_arg_val<uint32_t>();
    uint32_t output_stride[MAX_NUM_DIMENSIONS];
    for (int32_t i = 0; i < MAX_NUM_DIMENSIONS; ++i) {
        output_stride[i] = arg_fetcher.get_next_arg_val<uint32_t>();
    }

    if (need_transpose || need_mask || need_bias_add) {
        matmul_with_transpose_and_mask(
            output_tile_start_idx,
            num_output_tiles,
            Kt,
            transpose_input,
            transpose_other,
            need_input_mask_h,
            need_input_mask_w,
            output_stride,
            Mt,
            Nt,
            need_other_mask_h,
            need_other_mask_w,
            is_scalar_bias);
    } else {
        matmul(num_output_tiles, Kt);
    }
}
}  // namespace NAMESPACE
