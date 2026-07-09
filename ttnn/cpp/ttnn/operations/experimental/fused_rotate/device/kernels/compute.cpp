// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Fused per-edge sparse Wigner rotation. For each tile-row (32 edges) and each output block i,
// accumulate the fan-in  out[i] = sum_{(i,j,k)} coef_tile[k] * x_tile[j]  entirely in the dest
// registers, then pack once. All `nnz` multiply-accumulates run in a single kernel launch with
// one DRAM read of x and one write of out (vs `nnz` separate ttnn.addcmul dispatches).

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_coef = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t n_in_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t coef_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t n_out_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t n_out = get_compile_time_arg_val(6);
    constexpr uint32_t Wt = get_compile_time_arg_val(7);

    uint32_t arg = 0;
    const uint32_t num_rows = get_arg_val<uint32_t>(arg++);
    // deg[0..n_out-1], then ks[0..nnz-1], then js[0..nnz-1]
    const uint32_t deg_base = arg;
    const uint32_t ks_base = deg_base + n_out;
    const uint32_t nnz = coef_tiles;
    const uint32_t js_base = ks_base + nnz;

    binary_op_init_common(cb_coef, cb_x, cb_out);

    for (uint32_t r = 0; r < num_rows; r++) {
        cb_wait_front(cb_x, n_in_tiles);
        cb_wait_front(cb_coef, coef_tiles);
        cb_reserve_back(cb_out, n_out_tiles);

        uint32_t off = 0;  // running offset into ks/js
        for (uint32_t i = 0; i < n_out; i++) {
            const uint32_t d = get_arg_val<uint32_t>(deg_base + i);
            for (uint32_t wt = 0; wt < Wt; wt++) {
                tile_regs_acquire();
                // Fan-in: compute all d products into dst[0..d-1] (FPU), then sum them into
                // dst[0] with the SFPU dst-to-dst adder. Doing all muls first then all adds
                // (rather than alternating FPU/SFPU) is required for correctness. Needs d dst
                // slots -> the program uses dst_full_sync_en (8 fp32 slots; d<=5 for uma-s lmax=2).
                mul_tiles_init(cb_coef, cb_x);
                for (uint32_t m = 0; m < d; m++) {
                    const uint32_t k = get_arg_val<uint32_t>(ks_base + off + m);
                    const uint32_t j = get_arg_val<uint32_t>(js_base + off + m);
                    mul_tiles(cb_coef, cb_x, k, j * Wt + wt, m);
                }
                if (d > 1) {
                    add_binary_tile_init();
                    for (uint32_t m = 1; m < d; m++) {
                        add_binary_tile(0, m, 0);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_out, i * Wt + wt);
                tile_regs_release();
            }
            off += d;
        }

        cb_push_back(cb_out, n_out_tiles);
        cb_pop_front(cb_x, n_in_tiles);
        cb_pop_front(cb_coef, coef_tiles);
    }
}
