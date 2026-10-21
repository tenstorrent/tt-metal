// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for TILE reshape with W change.
//
// Per chunk: untilize in_tile_rows × Wt_in tiles (CB_in → CB_mid),
//            tilize out_tile_rows × Wt_out tiles (CB_mid → CB_out).
// The intermediate RM buffer holds the same flat data, just reinterpreted.
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/untilize.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t compute_num_blocks_per_column(uint32_t tile_cnt, uint32_t max_bct) {
    for (uint32_t bct = max_bct; bct >= 1; --bct) {
        if (tile_cnt % bct == 0) {
            return tile_cnt / bct;
        }
    }
    return 1;
}

void kernel_main() {
    constexpr uint32_t cb_in = get_named_compile_time_arg_val("cb_in");
    constexpr uint32_t cb_mid = get_named_compile_time_arg_val("cb_mid");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t Wt_in = get_named_compile_time_arg_val("wt_in");
    constexpr uint32_t Wt_out = get_named_compile_time_arg_val("wt_out");
    constexpr uint32_t in_tile_rows = get_named_compile_time_arg_val("in_tile_rows");
    constexpr uint32_t out_tile_rows = get_named_compile_time_arg_val("out_tile_rows");
    constexpr uint32_t max_bct = get_named_compile_time_arg_val("max_bct");

    uint32_t num_chunks = get_arg_val<uint32_t>(0);

    // Untilize blocking params
    constexpr uint32_t num_blocks_per_col = compute_num_blocks_per_column(Wt_in, max_bct);
    constexpr uint32_t block_ct_dim = Wt_in / num_blocks_per_col;
    constexpr uint32_t full_ct_dim = Wt_in;

    CircularBuffer in_buffer(cb_in);
    CircularBuffer mid_buffer(cb_mid);
    CircularBuffer out_buffer(cb_out);

    compute_kernel_hw_startup(cb_in, cb_mid);

    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        // Phase 1: untilize input tiles → RM sticks in cb_mid
        pack_untilize_init<block_ct_dim, full_ct_dim>(cb_in, cb_mid);

        for (uint32_t tr = 0; tr < in_tile_rows; ++tr) {
            mid_buffer.reserve_back(full_ct_dim);
            for (uint32_t b = 0; b < num_blocks_per_col; ++b) {
                in_buffer.wait_front(block_ct_dim);
                pack_untilize_block<block_ct_dim, full_ct_dim>(cb_in, 1, cb_mid, b);
                in_buffer.pop_front(block_ct_dim);
            }
            mid_buffer.push_back(full_ct_dim);
        }
        pack_untilize_uninit(cb_mid);

        // Phase 2: tilize RM sticks → output tiles in cb_out
        reconfig_data_format_srca(cb_in, cb_mid);
        pack_reconfig_data_format(cb_mid, cb_out);
        tilize_init(cb_mid, Wt_out, cb_out);

        for (uint32_t tr = 0; tr < out_tile_rows; ++tr) {
            mid_buffer.wait_front(Wt_out);
            out_buffer.reserve_back(Wt_out);
            tilize_block(cb_mid, Wt_out, cb_out);
            out_buffer.push_back(Wt_out);
            mid_buffer.pop_front(Wt_out);
        }
        tilize_uninit(cb_mid, cb_out);

        // Restore the phase-1 formats for the next chunk's untilize.
        reconfig_data_format_srca(cb_mid, cb_in);
        pack_reconfig_data_format(cb_out, cb_mid);
    }
}
