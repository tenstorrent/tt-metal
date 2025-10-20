// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/ema.h"

namespace NAMESPACE {
void MAIN {
    // Compile time args
    // -----------------
    constexpr auto total_batches_per_core = get_compile_time_arg_val(0);
    constexpr auto tiles_per_channel = get_compile_time_arg_val(1);
    constexpr auto alpha_bits = get_compile_time_arg_val(2);
    constexpr auto beta_bits = get_compile_time_arg_val(3);

    // CB indices
    // ----------
    constexpr auto src_cb = tt::CBIndex::c_0;
    constexpr auto dst_cb = tt::CBIndex::c_1;
    constexpr auto trp_cb = tt::CBIndex::c_2;

    // DST indices
    // -----------
    constexpr auto inp_dst_index = 0;
    constexpr auto output_dst_index = inp_dst_index + 1;

    //-------------------------------------------------------------------------
    // Main loop - compute ema for each batch
    ema_init();
    transpose_wh_init(src_cb, dst_cb);

    for (uint32_t batch_id = 0; batch_id < total_batches_per_core; ++batch_id) {
        for (uint32_t tile_id = 0; tile_id < tiles_per_channel; ++tile_id) {
            // Read input, transpose and compute ema
            cb_wait_front(src_cb, 1);
            tile_regs_acquire();
            transpose_wh_tile(src_cb, 0, inp_dst_index);
            ema_tile<inp_dst_index>((tile_id == 0));
            tile_regs_commit();
            cb_pop_front(src_cb, 1);

            cb_reserve_back(trp_cb, 1);
            tile_regs_wait();
            pack_tile(output_dst_index, trp_cb);
            tile_regs_release();
            cb_push_back(trp_cb, 1);

            // Transpose back and write to output
            cb_wait_front(trp_cb, 1);
            tile_regs_acquire();
            transpose_wh_tile(trp_cb, 0, output_dst_index);
            tile_regs_commit();
            cb_pop_front(trp_cb, 1);

            cb_reserve_back(dst_cb, 1);
            tile_regs_wait();
            pack_tile(output_dst_index, dst_cb);
            tile_regs_release();
            cb_push_back(dst_cb, 1);
        }
    }
}
}  // namespace NAMESPACE
