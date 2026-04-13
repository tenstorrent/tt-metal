// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/reduce.h"

#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/tile_move_copy.h"
#ifdef ARCH_QUASAR
#include "experimental/dataflow_buffer.h"
#else
#include "experimental/circular_buffer.h"
#endif

#include "llk_math_eltwise_binary.h"

void kernel_main() {
    uint32_t Ht = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);
    uint32_t NC = get_compile_time_arg_val(2);

    constexpr int onetile = 1;

#ifdef ARCH_QUASAR
    constexpr uint32_t cb_input = 0;
    constexpr uint32_t cb_scaler = 1;
    constexpr uint32_t cb_output = 2;
    constexpr uint32_t cb_acc = 3;
    constexpr uint32_t cb_ineg = 4;

    experimental::DataflowBuffer dfb_input(cb_input);
    experimental::DataflowBuffer dfb_scaler(cb_scaler);
    experimental::DataflowBuffer dfb_output(cb_output);
    experimental::DataflowBuffer dfb_acc(cb_acc);
    experimental::DataflowBuffer dfb_ineg(cb_ineg);
#else
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
#endif

    compute_kernel_hw_startup(cb_input, cb_scaler, cb_output);

#ifdef ARCH_QUASAR
    dfb_scaler.wait_front(onetile);
#else
    cb_scaler_obj.wait_front(1);  // scaler tile from the reader
#endif
    for (uint32_t nc = 0; nc < NC; nc++) {
        int dst_idx = 0;
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            // tiles are expected to be coming in in NCHW order (W-contiguous)
            // reducing in W means out[h][0] = sum(w=0..W-1, in[h][w])
            // in this case we just sequentially add to accumulator all the W-tiles in a row
            for (uint32_t wt = 0; wt < Wt; ++wt) {
#ifdef ARCH_QUASAR
                dfb_input.wait_front(onetile);
#else
                cb_input_obj.wait_front(onetile);
#endif
                tile_regs_acquire();
                copy_tile_init(cb_input);
                copy_tile(cb_input, 0, dst_idx);
                negative_tile_init();
                negative_tile(dst_idx);
                tile_regs_wait();
#ifdef ARCH_QUASAR
                dfb_input.pop_front(onetile);
                dfb_ineg.reserve_back(onetile);
#else
                cb_input_obj.pop_front(onetile);
                cb_ineg_obj.reserve_back(onetile);
#endif
                tile_regs_commit();
                pack_tile(dst_idx, cb_ineg);
                tile_regs_release();
#ifdef ARCH_QUASAR
                dfb_ineg.push_back(onetile);
#else
                cb_ineg_obj.push_back(onetile);
#endif

                tile_regs_acquire();
                if (wt > 0) {
#ifdef ARCH_QUASAR
                    dfb_acc.wait_front(onetile);
#else
                    cb_acc_obj.wait_front(onetile);
#endif
                    copy_tile_init(cb_acc);
                    copy_tile(cb_acc, 0, dst_idx);
                }

#ifdef ARCH_QUASAR
                dfb_ineg.wait_front(onetile);
#else
                cb_ineg_obj.wait_front(onetile);
#endif
                reduce_init(cb_ineg, cb_scaler, cb_acc);
                reduce_tile(cb_ineg, cb_scaler, 0, 0, dst_idx);
                reduce_uninit();
                tile_regs_wait();
#ifdef ARCH_QUASAR
                dfb_ineg.pop_front(onetile);
                if (wt > 0) {
                    dfb_acc.pop_front(onetile);
                }
                dfb_acc.reserve_back(onetile);
#else
                cb_ineg_obj.pop_front(onetile);
                if (wt > 0) {
                    cb_acc_obj.pop_front(onetile);
                }
                cb_acc_obj.reserve_back(onetile);
#endif
                tile_regs_commit();
                pack_tile(dst_idx, cb_acc);
                tile_regs_release();
#ifdef ARCH_QUASAR
                dfb_acc.push_back(onetile);
#else
                cb_acc_obj.push_back(onetile);
#endif
            }  // wt

#ifdef ARCH_QUASAR
            dfb_acc.wait_front(onetile);
#else
            cb_acc_obj.wait_front(onetile);
#endif
            tile_regs_acquire();
            copy_tile_init(cb_acc);
            copy_tile(cb_acc, 0, dst_idx);
            negative_tile_init();
            negative_tile(dst_idx);
            tile_regs_wait();
#ifdef ARCH_QUASAR
            dfb_acc.pop_front(onetile);
            dfb_output.reserve_back(onetile);
#else
            cb_acc_obj.pop_front(onetile);
            cb_output_obj.reserve_back(onetile);
#endif
            tile_regs_commit();
            pack_tile(dst_idx, cb_output);
            tile_regs_release();
#ifdef ARCH_QUASAR
            dfb_output.push_back(onetile);
#else
            cb_output_obj.push_back(onetile);
#endif
        }  // ht
    }  // nc
}
