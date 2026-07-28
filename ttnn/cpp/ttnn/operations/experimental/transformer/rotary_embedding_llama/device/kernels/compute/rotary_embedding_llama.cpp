// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

ALWI void ACQ() {
    tile_regs_acquire();
    tile_regs_wait();
}
ALWI void REL() {
    tile_regs_commit();
    tile_regs_release();
}

void kernel_main() {
    auto batch_start = get_arg(args::batch_start);
    auto batch_end = get_arg(args::batch_end);
    auto seq_t_start = get_arg(args::seq_t_start);
    auto seq_t_end = get_arg(args::seq_t_end);

    constexpr uint32_t onetile = 1;
    // Magic CB indices are gone: each buffer is a named DFB binding. The local
    // aliases keep the LLK/FIFO call sites readable; each is the dfb:: handle,
    // which converts implicitly to a CB id at the LLK call sites.
    constexpr auto in_cb = dfb::input;
    constexpr auto cos_cb = dfb::cos;
    constexpr auto sin_cb = dfb::sin;
    constexpr auto trans_mat_cb = dfb::trans_mat;

    constexpr auto rotated_in_interm_cb = dfb::rotated_interm;
    constexpr auto cos_interm_cb = dfb::cos_interm;
    constexpr auto sin_interm_cb = dfb::sin_interm;
    constexpr auto out_cb = dfb::out;
    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto n_heads = get_arg(args::n_heads);
    constexpr auto rotary_Ht = get_arg(args::rotary_Ht);

    DataflowBuffer in_cb_obj(in_cb);
    DataflowBuffer cos_cb_obj(cos_cb);
    DataflowBuffer sin_cb_obj(sin_cb);
    DataflowBuffer trans_mat_cb_obj(trans_mat_cb);
    DataflowBuffer rotated_in_interm_cb_obj(rotated_in_interm_cb);
    DataflowBuffer cos_interm_cb_obj(cos_interm_cb);
    DataflowBuffer sin_interm_cb_obj(sin_interm_cb);
    DataflowBuffer out_cb_obj(out_cb);

    const uint32_t rotary_seq_t_end = seq_t_end < rotary_Ht ? seq_t_end : rotary_Ht;
    const uint32_t my_rotary_seq_tiles = seq_t_start < rotary_seq_t_end ? rotary_seq_t_end - seq_t_start : 0;
    const uint32_t my_cos_sin_tiles = my_rotary_seq_tiles * Wt;

    compute_kernel_hw_startup<SrcOrder::Reverse>(in_cb, trans_mat_cb, out_cb);
    // Start from the state at the end of each iteration so same-format reconfigurations compile out.
    binary_op_init_common(cos_interm_cb, sin_interm_cb, out_cb);

    // Get the trans_mat
    trans_mat_cb_obj.wait_front(onetile);

    uint32_t in0_index = 0;
    uint32_t in1_index = 0;
    uint32_t interm_index = 0;

    for (uint32_t batch_id = batch_start; batch_id < batch_end; ++batch_id) {
#if RELOAD_IMPL == 0
        if (my_cos_sin_tiles > 0) {
            sin_cb_obj.wait_front(my_cos_sin_tiles);
            cos_cb_obj.wait_front(my_cos_sin_tiles);
        }
#endif
        for (uint32_t head_num = 0; head_num < n_heads; ++head_num) {
            uint32_t sin_cos_row_cnt = 0;
            for (uint32_t seq_tile = seq_t_start; seq_tile < rotary_seq_t_end; ++seq_tile) {
                // input cb wait and reserve
                in_cb_obj.wait_front(Wt);
#if RELOAD_IMPL == 1
                sin_cb_obj.wait_front(Wt);
                cos_cb_obj.wait_front(Wt);
#endif

                rotated_in_interm_cb_obj.reserve_back(Wt);
                sin_interm_cb_obj.reserve_back(Wt);
                cos_interm_cb_obj.reserve_back(Wt);
                out_cb_obj.reserve_back(Wt);

                // // rotated = x @ trans_mat
                // Matmul uses SrcOrder::Reverse: trans_mat is SrcA and input is SrcB.
                reconfig_data_format(cos_interm_cb, trans_mat_cb, sin_interm_cb, in_cb);
                pack_reconfig_data_format(out_cb, rotated_in_interm_cb);
                matmul_init(in_cb, trans_mat_cb);
                ACQ();
                for (uint32_t j = 0; j < Wt; ++j) {
                    matmul_tiles(in_cb, trans_mat_cb, j, in1_index, j);
                    pack_tile(j, rotated_in_interm_cb, j);
                }
                REL();
                rotated_in_interm_cb_obj.push_back(Wt);
                rotated_in_interm_cb_obj.wait_front(Wt);

                reconfig_data_format(trans_mat_cb, rotated_in_interm_cb, in_cb, sin_cb);
                pack_reconfig_data_format(rotated_in_interm_cb, sin_interm_cb);
                mul_tiles_init(rotated_in_interm_cb, sin_cb);
                ACQ();
                for (uint32_t j = 0; j < Wt; ++j) {
                    // sin_interim = rotated * sin
                    mul_tiles(rotated_in_interm_cb, sin_cb, j, j + (sin_cos_row_cnt * Wt), j);
                    pack_tile(j, sin_interm_cb, j);
                }
                REL();
                sin_interm_cb_obj.push_back(Wt);
                rotated_in_interm_cb_obj.pop_front(Wt);

                reconfig_data_format(rotated_in_interm_cb, in_cb, sin_cb, cos_cb);
                pack_reconfig_data_format(sin_interm_cb, cos_interm_cb);
                ACQ();
                for (uint32_t j = 0; j < Wt; ++j) {
                    // cos_interim = x * cos
                    mul_tiles(in_cb, cos_cb, j, j + (sin_cos_row_cnt * Wt), j);
                    pack_tile(j, cos_interm_cb, j);
                }
                REL();
                cos_interm_cb_obj.push_back(Wt);
                in_cb_obj.pop_front(Wt);  // Done with input
#if RELOAD_IMPL == 1
                sin_cb_obj.pop_front(Wt);
                cos_cb_obj.pop_front(Wt);
#endif

                sin_interm_cb_obj.wait_front(Wt);
                cos_interm_cb_obj.wait_front(Wt);
                reconfig_data_format(in_cb, cos_interm_cb, cos_cb, sin_interm_cb);
                pack_reconfig_data_format(cos_interm_cb, out_cb);
                add_tiles_init(cos_interm_cb, sin_interm_cb);
                ACQ();
                for (uint32_t j = 0; j < Wt; ++j) {
                    // out = cos_interim + sin_interim
                    add_tiles(cos_interm_cb, sin_interm_cb, j, j, j);
                    pack_tile(j, out_cb, j);
                }
                REL();
                out_cb_obj.push_back(Wt);
                sin_interm_cb_obj.pop_front(Wt);
                cos_interm_cb_obj.pop_front(Wt);

#if RELOAD_IMPL == 0
                // no-reload needs to increment this counter
                // Used a sin/cos row
                sin_cos_row_cnt++;
#endif
            }
        }

#if RELOAD_IMPL == 0
        if (my_cos_sin_tiles > 0) {
            sin_cb_obj.pop_front(my_cos_sin_tiles);
            cos_cb_obj.pop_front(my_cos_sin_tiles);
        }
#endif
    }

    // Done with the transformation matrix, so remove from CB
    trans_mat_cb_obj.pop_front(onetile);
}
