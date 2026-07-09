// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Fused SO(3) gate activation (column-split elementwise, NO reduction). Per edge tile-row:
//   scalar tiles [0, Ht):   mode 0 (fwd): out = silu(a)
//                           mode 1 (bw):  out = a * silu'(b)   (silu_bw, b = x)
//   vector tiles [Ht, Wt):  out = a * gate[t-Ht]               (both modes)
// silu'(b) = s + p - p*s  with s = sigmoid(b), p = silu(b) = b*s. One op per tile_regs_acquire.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gate = get_compile_time_arg_val(1);
    constexpr uint32_t cb_b = get_compile_time_arg_val(2);
    constexpr uint32_t cb_sp = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);
    constexpr uint32_t Wt = get_compile_time_arg_val(5);
    constexpr uint32_t Gt = get_compile_time_arg_val(6);
    constexpr uint32_t Ht = get_compile_time_arg_val(7);
    constexpr uint32_t mode = get_compile_time_arg_val(8);
    constexpr uint32_t cb_s = get_compile_time_arg_val(9);
    constexpr uint32_t cb_p = get_compile_time_arg_val(10);
    constexpr uint32_t cb_r = get_compile_time_arg_val(11);
    constexpr uint32_t cb_tmp = get_compile_time_arg_val(12);

    uint32_t arg = 0;
    const uint32_t num_rows = get_arg_val<uint32_t>(arg++);

    binary_op_init_common(cb_a, cb_gate, cb_out);

    for (uint32_t r = 0; r < num_rows; r++) {
        cb_wait_front(cb_a, Wt);
        cb_wait_front(cb_gate, Gt);

        cb_reserve_back(cb_out, Wt);

        // ---- scalar (l=0) tiles [0, Ht) ----
        if (mode == 0) {
            // out = silu(a)
            for (uint32_t t = 0; t < Ht; t++) {
                tile_regs_acquire();
                copy_tile_to_dst_init_short(cb_a);
                copy_tile(cb_a, t, 0);
                silu_tile_init();
                silu_tile(0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_out, t);
                tile_regs_release();
            }
        } else {
            // silu'(b) into cb_sp, then out = a * silu'(b)
            cb_wait_front(cb_b, Ht);
            // s = sigmoid(b) -> cb_s
            cb_reserve_back(cb_s, Ht);
            for (uint32_t t = 0; t < Ht; t++) {
                tile_regs_acquire();
                copy_tile_to_dst_init_short(cb_b);
                copy_tile(cb_b, t, 0);
                sigmoid_tile_init();
                sigmoid_tile(0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_s, t);
                tile_regs_release();
            }
            cb_push_back(cb_s, Ht);
            cb_wait_front(cb_s, Ht);
            // p = b * s -> cb_p
            cb_reserve_back(cb_p, Ht);
            mul_tiles_init(cb_b, cb_s);
            for (uint32_t t = 0; t < Ht; t++) {
                tile_regs_acquire();
                mul_tiles(cb_b, cb_s, t, t, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_p, t);
                tile_regs_release();
            }
            cb_push_back(cb_p, Ht);
            cb_wait_front(cb_p, Ht);
            cb_pop_front(cb_b, Ht);
            // r = p * s -> cb_r
            cb_reserve_back(cb_r, Ht);
            mul_tiles_init(cb_p, cb_s);
            for (uint32_t t = 0; t < Ht; t++) {
                tile_regs_acquire();
                mul_tiles(cb_p, cb_s, t, t, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_r, t);
                tile_regs_release();
            }
            cb_push_back(cb_r, Ht);
            cb_wait_front(cb_r, Ht);
            // tmp = s + p -> cb_tmp
            cb_reserve_back(cb_tmp, Ht);
            add_tiles_init(cb_s, cb_p);
            for (uint32_t t = 0; t < Ht; t++) {
                tile_regs_acquire();
                add_tiles(cb_s, cb_p, t, t, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_tmp, t);
                tile_regs_release();
            }
            cb_push_back(cb_tmp, Ht);
            cb_wait_front(cb_tmp, Ht);
            cb_pop_front(cb_s, Ht);
            cb_pop_front(cb_p, Ht);
            // silup = tmp - r -> cb_sp
            cb_reserve_back(cb_sp, Ht);
            sub_tiles_init(cb_tmp, cb_r);
            for (uint32_t t = 0; t < Ht; t++) {
                tile_regs_acquire();
                sub_tiles(cb_tmp, cb_r, t, t, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_sp, t);
                tile_regs_release();
            }
            cb_push_back(cb_sp, Ht);
            cb_wait_front(cb_sp, Ht);
            cb_pop_front(cb_tmp, Ht);
            cb_pop_front(cb_r, Ht);
            // out = a * silup -> cb_out[0:Ht)
            mul_tiles_init(cb_a, cb_sp);
            for (uint32_t t = 0; t < Ht; t++) {
                tile_regs_acquire();
                mul_tiles(cb_a, cb_sp, t, t, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_out, t);
                tile_regs_release();
            }
            cb_pop_front(cb_sp, Ht);
        }

        // ---- vector tiles [Ht, Wt): out = a * gate[t-Ht] ----
        mul_tiles_init(cb_a, cb_gate);
        for (uint32_t t = Ht; t < Wt; t++) {
            tile_regs_acquire();
            mul_tiles(cb_a, cb_gate, t, t - Ht, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out, t);
            tile_regs_release();
        }

        cb_push_back(cb_out, Wt);
        cb_pop_front(cb_a, Wt);
        cb_pop_front(cb_gate, Gt);
    }
}
