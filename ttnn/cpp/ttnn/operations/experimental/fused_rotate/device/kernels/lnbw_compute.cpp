// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Fused LayerNorm backward (grad wrt the LN input, affine scale folded into gy on host).
// Drop-in for tt_atom's hand-written _ln_bw. For each tile-row (32 edges) over W channels:
//   mean_x = mean_w(x);  xc = x - mean_x;  rstd = rsqrt(mean_w(xc^2) + eps);  xhat = xc*rstd
//   m1 = mean_w(gy);  m2 = mean_w(gy*xhat)
//   dx = rstd * (gy - m1 - xhat*m2)
// where gy = g_out * gamma is precomputed on host. Reductions are one accumulating matmul against
// a [32,32] tile whose column 0 is 1/W (rowsum-to-col0, then broadcast back with bcast_cols) --
// the same matmul-reduce trick as the gc kernel. All W stays L1-resident across the chain; one
// DRAM read of gy+x, one write of dx (vs ~15 ttnn ops).

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"

void kernel_main() {
    constexpr uint32_t cb_gout = get_compile_time_arg_val(0);   // [E, Wt] g_out (matmul, pre-silu-bw)
    constexpr uint32_t cb_x = get_compile_time_arg_val(1);      // [E, Wt] cached LN input
    constexpr uint32_t cb_red = get_compile_time_arg_val(2);    // [32,32] col0 = 1/W (resident)
    constexpr uint32_t cb_xc = get_compile_time_arg_val(3);     // scratch: x - mean_x
    constexpr uint32_t cb_xhat = get_compile_time_arg_val(4);   // scratch: xc * rstd
    constexpr uint32_t cb_prod = get_compile_time_arg_val(5);   // scratch: 1-tile products for reduce
    constexpr uint32_t cb_s = get_compile_time_arg_val(6);      // scratch: 1-tile scalars (mean/rstd)
    constexpr uint32_t cb_rstd = get_compile_time_arg_val(7);   // rstd (col0) held across the row
    constexpr uint32_t cb_dx = get_compile_time_arg_val(8);     // [E, Wt] output
    constexpr uint32_t Wt = get_compile_time_arg_val(9);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(10);
    constexpr uint32_t cb_n = get_compile_time_arg_val(11);     // [E, Wt] pre-silu activation (LN output)
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(12); // [1, Wt] LN affine scale (row bcast, resident)
    constexpr uint32_t cb_gy = get_compile_time_arg_val(13);    // internal: gy = g_out*silu'(n)*gamma
    constexpr uint32_t cb_g1 = get_compile_time_arg_val(14);    // internal preamble scratch

    uint32_t arg = 0;
    const uint32_t num_rows = get_arg_val<uint32_t>(arg++);

    binary_op_init_common(cb_x, cb_red, cb_dx);
    cb_wait_front(cb_gamma, Wt);   // resident LN scale, loaded once

    // Accumulating matmul reduce of `cb_in` (Wt tiles) into col0 of dst0 -> pack to `cb_scalar`.
    // cb_red col0 = 1/W so this is the row-wise mean over W. Caller must have cb_in filled.
    auto reduce_mean = [&](uint32_t cb_in, uint32_t cb_scalar) {
        mm_init(cb_in, cb_red, cb_scalar);
        tile_regs_acquire();
        for (uint32_t wt = 0; wt < Wt; wt++) {
            matmul_tiles(cb_in, cb_red, wt, 0, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_scalar, 1);
        pack_tile(0, cb_scalar);
        cb_push_back(cb_scalar, 1);
        tile_regs_release();
    };

    for (uint32_t r = 0; r < num_rows; r++) {
        // ===== silu+gamma fold: build gy = g_out * silu'(n) * gamma, one op per tile_regs_acquire.
        // silu'(n) = s + p - p*s  with s = sigmoid(n), p = n*s (= silu(n)). =====
        cb_wait_front(cb_gout, Wt);
        cb_wait_front(cb_n, Wt);

        // s = sigmoid(n) -> cb_xc
        cb_reserve_back(cb_xc, Wt);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(cb_n);
            copy_tile(cb_n, wt, 0);
            sigmoid_tile_init();
            sigmoid_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_xc, wt);
            tile_regs_release();
        }
        cb_push_back(cb_xc, Wt);
        cb_wait_front(cb_xc, Wt);   // s

        // p = n * s -> cb_xhat
        cb_reserve_back(cb_xhat, Wt);
        mul_tiles_init(cb_n, cb_xc);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            tile_regs_acquire();
            mul_tiles(cb_n, cb_xc, wt, wt, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_xhat, wt);
            tile_regs_release();
        }
        cb_push_back(cb_xhat, Wt);
        cb_wait_front(cb_xhat, Wt);  // p = silu(n)
        cb_pop_front(cb_n, Wt);

        // r = p * s -> cb_prod
        cb_reserve_back(cb_prod, Wt);
        mul_tiles_init(cb_xhat, cb_xc);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            tile_regs_acquire();
            mul_tiles(cb_xhat, cb_xc, wt, wt, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_prod, wt);
            tile_regs_release();
        }
        cb_push_back(cb_prod, Wt);
        cb_wait_front(cb_prod, Wt);  // r = p*s

        // tmp = s + p -> cb_g1
        cb_reserve_back(cb_g1, Wt);
        add_tiles_init(cb_xc, cb_xhat);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            tile_regs_acquire();
            add_tiles(cb_xc, cb_xhat, wt, wt, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_g1, wt);
            tile_regs_release();
        }
        cb_push_back(cb_g1, Wt);
        cb_wait_front(cb_g1, Wt);   // tmp = s+p
        cb_pop_front(cb_xc, Wt);
        cb_pop_front(cb_xhat, Wt);

        // silup = tmp - r -> cb_xc
        cb_reserve_back(cb_xc, Wt);
        sub_tiles_init(cb_g1, cb_prod);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            tile_regs_acquire();
            sub_tiles(cb_g1, cb_prod, wt, wt, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_xc, wt);
            tile_regs_release();
        }
        cb_push_back(cb_xc, Wt);
        cb_wait_front(cb_xc, Wt);   // silup = silu'(n)
        cb_pop_front(cb_g1, Wt);
        cb_pop_front(cb_prod, Wt);

        // g1 = g_out * silup -> cb_xhat
        cb_reserve_back(cb_xhat, Wt);
        mul_tiles_init(cb_gout, cb_xc);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            tile_regs_acquire();
            mul_tiles(cb_gout, cb_xc, wt, wt, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_xhat, wt);
            tile_regs_release();
        }
        cb_push_back(cb_xhat, Wt);
        cb_wait_front(cb_xhat, Wt);
        cb_pop_front(cb_xc, Wt);
        cb_pop_front(cb_gout, Wt);

        // gy = g1 * gamma (bcast gamma row across the 32 edges) -> cb_gy
        cb_reserve_back(cb_gy, Wt);
        mul_bcast_rows_init_short(cb_xhat, cb_gamma);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            tile_regs_acquire();
            mul_tiles_bcast_rows(cb_xhat, cb_gamma, wt, wt, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_gy, wt);
            tile_regs_release();
        }
        cb_push_back(cb_gy, Wt);
        cb_pop_front(cb_xhat, Wt);

        cb_wait_front(cb_gy, Wt);
        cb_wait_front(cb_x, Wt);

        // --- mean_x = mean_w(x) into cb_s ---
        reduce_mean(cb_x, cb_s);

        // --- xc = x - mean_x (bcast col0 of cb_s across cols) -> cb_xc ---
        cb_wait_front(cb_s, 1);
        cb_reserve_back(cb_xc, Wt);
        sub_bcast_cols_init_short(cb_x, cb_s);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            tile_regs_acquire();
            sub_tiles_bcast_cols(cb_x, cb_s, wt, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_xc, wt);
            tile_regs_release();
        }
        cb_push_back(cb_xc, Wt);
        cb_pop_front(cb_s, 1);

        // --- var = mean_w(xc^2): build xc^2 into cb_prod (Wt tiles), reduce -> cb_s ; rstd = rsqrt(var+eps) ---
        cb_wait_front(cb_xc, Wt);
        cb_reserve_back(cb_prod, Wt);
        mul_tiles_init(cb_xc, cb_xc);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            tile_regs_acquire();
            mul_tiles(cb_xc, cb_xc, wt, wt, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_prod, wt);
            tile_regs_release();
        }
        cb_push_back(cb_prod, Wt);
        cb_wait_front(cb_prod, Wt);
        // reduce cb_prod -> var(col0) in dst0, then add eps + rsqrt in-place -> cb_rstd
        mm_init(cb_prod, cb_red, cb_rstd);
        tile_regs_acquire();
        for (uint32_t wt = 0; wt < Wt; wt++) {
            matmul_tiles(cb_prod, cb_red, wt, 0, 0);
        }
        binop_with_scalar_tile_init();
        add_unary_tile(0, eps_bits);
        rsqrt_tile_init();
        rsqrt_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_rstd, 1);
        pack_tile(0, cb_rstd);
        cb_push_back(cb_rstd, 1);
        tile_regs_release();
        cb_pop_front(cb_prod, Wt);

        // --- xhat = xc * rstd (bcast) -> cb_xhat ---
        cb_wait_front(cb_rstd, 1);
        cb_reserve_back(cb_xhat, Wt);
        mul_bcast_cols_init_short(cb_xc, cb_rstd);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(cb_xc, cb_rstd, wt, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_xhat, wt);
            tile_regs_release();
        }
        cb_push_back(cb_xhat, Wt);
        cb_pop_front(cb_xc, Wt);

        // The dx assembly keeps only ONE scalar live in cb_s at a time (sharing cb_s for m1 AND m2
        // simultaneously corrupts the 2nd entry on multi-row cores). Each tile_regs_acquire runs a
        // SINGLE bcast/eltwise op then packs (multi-op-per-acquire with init switches also corrupts).
        // Scratch: cb_prod (a / gy*xhat product), cb_xc (b), cb_xhat reused for e.

        // --- m2 = mean_w(gy*xhat) -> cb_s ; b = xhat * m2 -> cb_xc ---
        cb_wait_front(cb_xhat, Wt);
        cb_reserve_back(cb_prod, Wt);
        mul_tiles_init(cb_gy, cb_xhat);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            tile_regs_acquire();
            mul_tiles(cb_gy, cb_xhat, wt, wt, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_prod, wt);
            tile_regs_release();
        }
        cb_push_back(cb_prod, Wt);
        cb_wait_front(cb_prod, Wt);
        reduce_mean(cb_prod, cb_s);        // m2 -> cb_s (front, sole entry)
        cb_pop_front(cb_prod, Wt);
        cb_wait_front(cb_s, 1);
        cb_reserve_back(cb_xc, Wt);
        mul_bcast_cols_init_short(cb_xhat, cb_s);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(cb_xhat, cb_s, wt, 0, 0);   // b = xhat * m2
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_xc, wt);
            tile_regs_release();
        }
        cb_push_back(cb_xc, Wt);
        cb_pop_front(cb_s, 1);
        cb_pop_front(cb_xhat, Wt);         // xhat done; reuse cb_xhat for e below

        // --- m1 = mean_w(gy) -> cb_s ; a = gy - m1 -> cb_prod ---
        reduce_mean(cb_gy, cb_s);
        cb_wait_front(cb_s, 1);
        cb_reserve_back(cb_prod, Wt);
        sub_bcast_cols_init_short(cb_gy, cb_s);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            tile_regs_acquire();
            sub_tiles_bcast_cols(cb_gy, cb_s, wt, 0, 0);     // a = gy - m1
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_prod, wt);
            tile_regs_release();
        }
        cb_push_back(cb_prod, Wt);
        cb_pop_front(cb_s, 1);

        // --- e = a - b -> cb_xhat (reused) ---
        cb_wait_front(cb_prod, Wt);
        cb_wait_front(cb_xc, Wt);
        cb_reserve_back(cb_xhat, Wt);
        sub_tiles_init(cb_prod, cb_xc);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            tile_regs_acquire();
            sub_tiles(cb_prod, cb_xc, wt, wt, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_xhat, wt);
            tile_regs_release();
        }
        cb_push_back(cb_xhat, Wt);
        cb_pop_front(cb_prod, Wt);
        cb_pop_front(cb_xc, Wt);

        // --- dx = e * rstd -> cb_dx ---
        cb_wait_front(cb_xhat, Wt);
        cb_reserve_back(cb_dx, Wt);
        mul_bcast_cols_init_short(cb_xhat, cb_rstd);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(cb_xhat, cb_rstd, wt, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_dx, wt);
            tile_regs_release();
        }
        cb_push_back(cb_dx, Wt);
        cb_pop_front(cb_xhat, Wt);

        cb_pop_front(cb_rstd, 1);
        cb_pop_front(cb_gy, Wt);
        cb_pop_front(cb_x, Wt);
    }
}
