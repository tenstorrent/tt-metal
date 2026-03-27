// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/reduce.h"

#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/tile_move_copy.h"
#include "experimental/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp"

#include "llk_math_eltwise_binary.h"

void kernel_main() {
    uint32_t Ht = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);
    uint32_t NC = get_compile_time_arg_val(2);

    // Circular buffers:
    constexpr uint32_t cb_input = tt::CBIndex::c_0;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;
    constexpr uint32_t cb_output = tt::CBIndex::c_3;
    constexpr uint32_t cb_acc = tt::CBIndex::c_4;
    constexpr uint32_t cb_ineg = tt::CBIndex::c_5;

    experimental::CircularBuffer cb_input_obj(cb_input);
    experimental::CircularBuffer cb_scaler_obj(cb_scaler);
    experimental::CircularBuffer cb_output_obj(cb_output);
    experimental::CircularBuffer cb_acc_obj(cb_acc);
    experimental::CircularBuffer cb_ineg_obj(cb_ineg);

    compute_kernel_hw_startup(cb_input, cb_scaler, cb_output);

    cb_scaler_obj.wait_front(1);  // scaler tile from the reader
    for (uint32_t nc = 0; nc < NC; nc++) {
        constexpr int onetile = 1;
        int dst_idx = 0;
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            // tiles are expected to be coming in in NCHW order (W-contiguous)
            // reducing in W means out[h][0] = sum(w=0..W-1, in[h][w])
            // in this case we just sequentially add to accumulator all the W-tiles in a row
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                // Negate input tile: cb_input -> -x -> cb_ineg
                // OUTPUT reconfig needed: packer was configured for cb_output/cb_acc by startup/reduce
                compute_kernel_lib::sfpu_op<
                    cb_input,
                    compute_kernel_lib::SfpuInputPolicy::WaitAndPopPerTile,
                    compute_kernel_lib::SfpuOutputPolicy::PerTile,
                    compute_kernel_lib::SfpuDataFormatReconfig::OUTPUT>(cb_ineg, 1, compute_kernel_lib::Neg<>{});

                tile_regs_acquire();
                if (wt > 0) {
                    cb_acc_obj.wait_front(onetile);
                    copy_tile_init(cb_acc);
                    copy_tile(cb_acc, 0, dst_idx);
                }

                cb_ineg_obj.wait_front(onetile);
                reduce_init<REDUCE_OP, REDUCE_DIM>(cb_ineg, cb_scaler, cb_acc);
                reduce_tile<REDUCE_OP, REDUCE_DIM>(cb_ineg, cb_scaler, 0, 0, dst_idx);
                reduce_uninit();
                tile_regs_wait();
                cb_ineg_obj.pop_front(onetile);
                if (wt > 0) {
                    cb_acc_obj.pop_front(onetile);
                }
                cb_acc_obj.reserve_back(onetile);
                tile_regs_commit();
                pack_tile(dst_idx, cb_acc);
                tile_regs_release();
                cb_acc_obj.push_back(onetile);
            }  // wt

            // Negate accumulated result: cb_acc -> -acc -> cb_output
            // INPUT_AND_OUTPUT reconfig: reduce left unpacker on cb_ineg/cb_scaler, packer on cb_acc
            compute_kernel_lib::sfpu_op<cb_acc>(cb_output, 1, compute_kernel_lib::Neg<>{});
        }  // ht
    }  // nc
}
