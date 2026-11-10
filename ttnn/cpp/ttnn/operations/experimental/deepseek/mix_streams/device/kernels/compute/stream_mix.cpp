// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"

// Per work unit (one [32, 32] tile position shared by all streams), n_out output tiles:
//   out[i] = sum_j diag_w(i, j) @ stream[j]  (+ diag_post(i) @ sublayer_out)
// accumulated in a single DST tile, so each output tile costs hc (+1) matmuls and one pack.
void kernel_main() {
    const uint32_t start_unit = get_arg_val<uint32_t>(0);
    const uint32_t num_units = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t hc = get_compile_time_arg_val(3);
    constexpr uint32_t n_out = get_compile_time_arg_val(4);
    constexpr bool has_post = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t d_tiles = get_compile_time_arg_val(6);

    constexpr uint32_t n_in = hc + (has_post ? 1u : 0u);
    constexpr uint32_t n_w = n_in * n_out;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_w, cb_in, cb_out);
    matmul_init(cb_w, cb_in);

    const uint32_t end_unit = start_unit + num_units;
    uint32_t unit = start_unit;
    while (unit < end_unit) {
        const uint32_t r = unit / d_tiles;
        const uint32_t group_end = (end_unit < (r + 1) * d_tiles) ? end_unit : (r + 1) * d_tiles;

        cb_wait_front(cb_w, n_w);
        for (uint32_t u = unit; u < group_end; ++u) {
            cb_wait_front(cb_in, n_in);
            for (uint32_t i = 0; i < n_out; ++i) {
                tile_regs_acquire();
                for (uint32_t j = 0; j < hc; ++j) {
                    matmul_tiles(cb_w, cb_in, i * hc + j, j, 0);
                }
                if constexpr (has_post) {
                    matmul_tiles(cb_w, cb_in, hc * n_out + i, hc, 0);
                }
                tile_regs_commit();

                cb_reserve_back(cb_out, 1);
                tile_regs_wait();
                pack_reconfig_data_format(cb_out);
                pack_tile(0, cb_out);
                tile_regs_release();
                cb_push_back(cb_out, 1);
            }
            cb_pop_front(cb_in, n_in);
        }
        cb_pop_front(cb_w, n_w);
        unit = group_end;
    }
}
