// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Fused per-edge coefficient adjoint (the rotate_bw dE/dcoef). For each tile-row (32 edges) and
// each structural nonzero k=(i,j):
//     gc[e, k] = sum_w  gout[e, i*W + w] * xin[e, j*W + w]
// i.e. the per-edge dot product over the W channels of output block i of `gout` with input block j
// of `xin`. Done WITHOUT materialising the [E, nnz*W] product concat that the ttnn path builds in
// DRAM: the products stay L1-resident and one accumulating matmul against a column-selector tile
// (`sel[c]` has column c all-ones) does BOTH the W-reduction AND the placement into output column
// c in a single op. Reads gout+xin once, writes the compact gc[E, nnz] once.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_gout = get_compile_time_arg_val(0);
    constexpr uint32_t cb_xin = get_compile_time_arg_val(1);
    constexpr uint32_t cb_sel = get_compile_time_arg_val(2);
    constexpr uint32_t cb_prod = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);
    constexpr uint32_t n_out_tiles = get_compile_time_arg_val(5);  // n_out*Wt (gout blocks)
    constexpr uint32_t n_in_tiles = get_compile_time_arg_val(6);   // n_in*Wt  (xin blocks)
    constexpr uint32_t Wt = get_compile_time_arg_val(7);
    constexpr uint32_t nnz = get_compile_time_arg_val(8);
    constexpr uint32_t out_tiles = get_compile_time_arg_val(9);    // ceil(nnz/32)

    uint32_t arg = 0;
    const uint32_t num_rows = get_arg_val<uint32_t>(arg++);
    const uint32_t is_base = arg;              // is[0..nnz-1]: gout block per nonzero
    const uint32_t js_base = is_base + nnz;    // js[0..nnz-1]: xin block per nonzero

    binary_op_init_common(cb_gout, cb_xin, cb_out);

    // the 32 column-selector tiles are pos-independent -> loaded once, resident for the kernel.
    cb_wait_front(cb_sel, 32);

    for (uint32_t r = 0; r < num_rows; r++) {
        cb_wait_front(cb_gout, n_out_tiles);
        cb_wait_front(cb_xin, n_in_tiles);

        for (uint32_t ot = 0; ot < out_tiles; ot++) {
            uint32_t d = nnz - ot * 32;
            if (d > 32) {
                d = 32;
            }

            // Phase A: products gout_i * xin_j for the d nonzeros of this output tile -> cb_prod
            // (d*Wt tiles, packed at explicit slots c*Wt+wt).
            mul_tiles_init(cb_gout, cb_xin);
            cb_reserve_back(cb_prod, d * Wt);
            for (uint32_t c = 0; c < d; c++) {
                const uint32_t k = ot * 32 + c;
                const uint32_t i = get_arg_val<uint32_t>(is_base + k);
                const uint32_t j = get_arg_val<uint32_t>(js_base + k);
                for (uint32_t wt = 0; wt < Wt; wt++) {
                    tile_regs_acquire();
                    mul_tiles(cb_gout, cb_xin, i * Wt + wt, j * Wt + wt, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(0, cb_prod, c * Wt + wt);
                    tile_regs_release();
                }
            }
            cb_push_back(cb_prod, d * Wt);

            // Phase B: one accumulating matmul per (c,wt). matmul_tiles does dst += prod @ sel[c];
            // sel[c] has column c all-ones so prod@sel[c] = rowsum(prod) placed in column c. Summing
            // over wt gives the full W-reduction; over c fills the distinct output columns.
            cb_wait_front(cb_prod, d * Wt);
            mm_init(cb_prod, cb_sel, cb_out);
            cb_reserve_back(cb_out, 1);
            tile_regs_acquire();
            uint32_t pt = 0;
            for (uint32_t c = 0; c < d; c++) {
                for (uint32_t wt = 0; wt < Wt; wt++) {
                    matmul_tiles(cb_prod, cb_sel, pt, c, 0);
                    pt++;
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out, ot);
            tile_regs_release();
            cb_pop_front(cb_prod, d * Wt);
            cb_push_back(cb_out, 1);
        }

        cb_pop_front(cb_gout, n_out_tiles);
        cb_pop_front(cb_xin, n_in_tiles);
    }
}
