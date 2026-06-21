// SPDX-License-Identifier: Apache-2.0
//
// Chunked-parallel gated delta-rule prefill (compute). One core per v-head, loops
// n_chunks chunks of C=32 tokens. Decay scalings are folded into the keys/queries on
// host (Kdec=(beta*d)k, KiT=transpose((1/d)k), Qd=d*q) so this kernel is matmul-heavy
// and precision-safe. Per chunk (entering state S0=[Dk,Dv]):
//   kS0 = k@S0 ; qS0 = q@S0
//   A   = trils * (Kdec @ KiT)                       [C,C] strict-lower
//   inv = (I+A)^-1 via Neumann (I-A)(I+A^2)(I+A^4)(I+A^8)(I+A^16)
//   rhs = betacol * (v - dcol*kS0)                    [C,Dv] (elementwise)
//   U   = inv @ rhs
//   M   = trili * (Qd @ KiT)                          [C,C] incl-lower
//   Oraw= dcol*qS0 + M@U                              [C,Dv]
//   Snew= dlast * (S0 + KiT@U)                        [Dk,Dv] (dlast scalar-bcast)
//   out = gated_rmsnorm(Oraw, z, norm_w)              [C,Dv] (per-row REDUCE_ROW)
// Verified math: chunk_delta_ref.py / chunked_prefill_pipeline_test.py (PCC≈1.0).

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"

void kernel_main() {
    constexpr uint32_t cb_k       = get_compile_time_arg_val(0);   // [C,Dk]
    constexpr uint32_t cb_q       = get_compile_time_arg_val(1);   // [C,Dk]
    constexpr uint32_t cb_v       = get_compile_time_arg_val(2);   // [C,Dv]
    constexpr uint32_t cb_z       = get_compile_time_arg_val(3);   // [C,Dv]
    constexpr uint32_t cb_Kdec    = get_compile_time_arg_val(4);   // [C,Dk]
    constexpr uint32_t cb_KiT     = get_compile_time_arg_val(5);   // [Dk,C]
    constexpr uint32_t cb_Qd      = get_compile_time_arg_val(6);   // [C,Dk]
    constexpr uint32_t cb_dcol    = get_compile_time_arg_val(7);   // [C,Dv]
    constexpr uint32_t cb_betacol = get_compile_time_arg_val(8);   // [C,Dv]
    constexpr uint32_t cb_dlast   = get_compile_time_arg_val(9);   // [.,.] scalar at [0,0]
    constexpr uint32_t cb_state_init = get_compile_time_arg_val(10);  // reader-seeded S0
    constexpr uint32_t cb_ident   = get_compile_time_arg_val(11);  // [C,C]
    constexpr uint32_t cb_trils   = get_compile_time_arg_val(12);  // [C,C] strict lower
    constexpr uint32_t cb_trili   = get_compile_time_arg_val(13);  // [C,C] incl lower
    constexpr uint32_t cb_normw   = get_compile_time_arg_val(14);  // [1,Dv]
    constexpr uint32_t cb_scaler  = get_compile_time_arg_val(15);  // 1/Dv
    constexpr uint32_t cb_eps     = get_compile_time_arg_val(16);
    constexpr uint32_t cb_output  = get_compile_time_arg_val(17);  // [C,Dv]
    // scratch
    constexpr uint32_t cb_kS0 = get_compile_time_arg_val(18);  // [C,Dv]
    constexpr uint32_t cb_qS0 = get_compile_time_arg_val(19);
    constexpr uint32_t cb_rhs = get_compile_time_arg_val(20);
    constexpr uint32_t cb_U   = get_compile_time_arg_val(21);
    constexpr uint32_t cb_p   = get_compile_time_arg_val(22);  // [C,C] Neumann P
    constexpr uint32_t cb_p2  = get_compile_time_arg_val(23);  // [C,C] P@P
    constexpr uint32_t cb_inv = get_compile_time_arg_val(24);  // [C,C] running inverse
    constexpr uint32_t cb_ipp = get_compile_time_arg_val(25);  // [C,C] I+P
    constexpr uint32_t cb_sn  = get_compile_time_arg_val(26);  // [Dk,Dv]
    constexpr uint32_t cb_t0  = get_compile_time_arg_val(27);  // scratch
    constexpr uint32_t cb_t1  = get_compile_time_arg_val(28);

    constexpr uint32_t cb_state_A = get_compile_time_arg_val(29);
    constexpr uint32_t cb_state_B = get_compile_time_arg_val(30);
    constexpr uint32_t Dk_tiles    = get_compile_time_arg_val(31);
    constexpr uint32_t Dv_tiles    = get_compile_time_arg_val(32);
    constexpr uint32_t n_chunks    = get_compile_time_arg_val(33);
    constexpr uint32_t state_tiles = Dk_tiles * Dv_tiles;

    binary_op_init_common(cb_k, cb_state_init, cb_kS0);

    for (uint32_t c = 0; c < n_chunks; c++) {
        // state CBs: single-producer/consumer per the proven prefill pattern.
        // chunk 0 reads reader-seeded init; thereafter ping-pong A/B (compute-only).
        uint32_t cur_state = (c == 0) ? cb_state_init : ((c % 2 == 1) ? cb_state_A : cb_state_B);
        uint32_t new_state = (c % 2 == 0) ? cb_state_A : cb_state_B;
        cb_wait_front(cur_state, state_tiles);
        cb_wait_front(cb_k, Dk_tiles);
        cb_wait_front(cb_q, Dk_tiles);
        cb_wait_front(cb_v, Dv_tiles);
        cb_wait_front(cb_Kdec, Dk_tiles);
        cb_wait_front(cb_KiT, Dk_tiles);
        cb_wait_front(cb_Qd, Dk_tiles);
        cb_wait_front(cb_dcol, Dv_tiles);
        cb_wait_front(cb_betacol, Dv_tiles);

        // kS0 = k@S0, qS0 = q@S0   [C,Dv]
        mm_init(cb_k, cur_state, cb_kS0);
        cb_reserve_back(cb_kS0, Dv_tiles);
        for (uint32_t jv = 0; jv < Dv_tiles; jv++) {
            tile_regs_acquire();
            for (uint32_t ik = 0; ik < Dk_tiles; ik++) matmul_tiles(cb_k, cur_state, ik, ik * Dv_tiles + jv, 0);
            tile_regs_commit(); tile_regs_wait(); pack_tile(0, cb_kS0); tile_regs_release();
        }
        cb_push_back(cb_kS0, Dv_tiles);
        mm_init(cb_q, cur_state, cb_qS0);
        cb_reserve_back(cb_qS0, Dv_tiles);
        for (uint32_t jv = 0; jv < Dv_tiles; jv++) {
            tile_regs_acquire();
            for (uint32_t ik = 0; ik < Dk_tiles; ik++) matmul_tiles(cb_q, cur_state, ik, ik * Dv_tiles + jv, 0);
            tile_regs_commit(); tile_regs_wait(); pack_tile(0, cb_qS0); tile_regs_release();
        }
        cb_push_back(cb_qS0, Dv_tiles);

        // A = trils * (Kdec @ KiT)  -> cb_p (=N)
        mm_init(cb_Kdec, cb_KiT, cb_t0);
        cb_reserve_back(cb_t0, 1);
        tile_regs_acquire();
        for (uint32_t ik = 0; ik < Dk_tiles; ik++) matmul_tiles(cb_Kdec, cb_KiT, ik, ik, 0);
        tile_regs_commit(); tile_regs_wait(); pack_tile(0, cb_t0); tile_regs_release();
        cb_push_back(cb_t0, 1);
        cb_wait_front(cb_t0, 1); cb_wait_front(cb_trils, 1);
        binary_op_init_common(cb_trils, cb_t0, cb_trils);
        mul_tiles_init(cb_trils, cb_t0);
        cb_reserve_back(cb_p, 1);
        tile_regs_acquire(); mul_tiles(cb_trils, cb_t0, 0, 0, 0);
        tile_regs_commit(); tile_regs_wait(); pack_tile(0, cb_p); tile_regs_release();
        cb_push_back(cb_p, 1);
        cb_pop_front(cb_t0, 1);

        // inv = I - A   -> cb_inv
        cb_wait_front(cb_p, 1); cb_wait_front(cb_ident, 1);
        binary_op_init_common(cb_ident, cb_p, cb_ident);
        sub_tiles_init(cb_ident, cb_p);
        cb_reserve_back(cb_inv, 1);
        tile_regs_acquire(); sub_tiles(cb_ident, cb_p, 0, 0, 0);
        tile_regs_commit(); tile_regs_wait(); pack_tile(0, cb_inv); tile_regs_release();
        cb_push_back(cb_inv, 1);
        // Neumann doubling: 4 steps, P in cb_p -> P^2 in cb_p2 (distinct CBs, no aliasing)
        for (uint32_t step = 0; step < 4; step++) {
            cb_wait_front(cb_p, 1);
            // P@P: copy P to a distinct CB first — matmul with both operands the same
            // CB tile is unreliable (same bug as the ttnn-op self-aliased matmul).
            copy_tile_init(cb_p);
            cb_reserve_back(cb_t0, 1);
            tile_regs_acquire(); copy_tile(cb_p, 0, 0);
            tile_regs_commit(); tile_regs_wait(); pack_tile(0, cb_t0); tile_regs_release();
            cb_push_back(cb_t0, 1);
            cb_wait_front(cb_t0, 1);
            mm_init(cb_p, cb_t0, cb_p2);
            cb_reserve_back(cb_p2, 1);
            tile_regs_acquire(); matmul_tiles(cb_p, cb_t0, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait(); pack_tile(0, cb_p2); tile_regs_release();
            cb_push_back(cb_p2, 1);
            cb_pop_front(cb_t0, 1);
            cb_pop_front(cb_p, 1);
            // I+P^2 -> cb_ipp
            cb_wait_front(cb_p2, 1);
            binary_op_init_common(cb_ident, cb_p2, cb_ident);
            add_tiles_init(cb_ident, cb_p2);
            cb_reserve_back(cb_ipp, 1);
            tile_regs_acquire(); add_tiles(cb_ident, cb_p2, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait(); pack_tile(0, cb_ipp); tile_regs_release();
            cb_push_back(cb_ipp, 1);
            // inv = inv @ (I+P^2) -> cb_t0 -> cb_inv
            cb_wait_front(cb_inv, 1); cb_wait_front(cb_ipp, 1);
            mm_init(cb_inv, cb_ipp, cb_t0);
            cb_reserve_back(cb_t0, 1);
            tile_regs_acquire(); matmul_tiles(cb_inv, cb_ipp, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait(); pack_tile(0, cb_t0); tile_regs_release();
            cb_push_back(cb_t0, 1);
            cb_pop_front(cb_inv, 1); cb_pop_front(cb_ipp, 1);
            cb_wait_front(cb_t0, 1);
            copy_tile_init(cb_t0);
            cb_reserve_back(cb_inv, 1);
            tile_regs_acquire(); copy_tile(cb_t0, 0, 0);
            tile_regs_commit(); tile_regs_wait(); pack_tile(0, cb_inv); tile_regs_release();
            cb_push_back(cb_inv, 1);
            cb_pop_front(cb_t0, 1);
            // P <- P^2 for next step
            cb_wait_front(cb_p2, 1);
            copy_tile_init(cb_p2);
            cb_reserve_back(cb_p, 1);
            tile_regs_acquire(); copy_tile(cb_p2, 0, 0);
            tile_regs_commit(); tile_regs_wait(); pack_tile(0, cb_p); tile_regs_release();
            cb_push_back(cb_p, 1);
            cb_pop_front(cb_p2, 1);
        }
        cb_pop_front(cb_p, 1);   // discard final P

        // rhs = betacol * (v - dcol*kS0)
        cb_wait_front(cb_kS0, Dv_tiles);
        binary_op_init_common(cb_dcol, cb_kS0, cb_dcol);
        mul_tiles_init(cb_dcol, cb_kS0);
        cb_reserve_back(cb_t0, Dv_tiles);
        for (uint32_t j = 0; j < Dv_tiles; j++) {
            tile_regs_acquire(); mul_tiles(cb_dcol, cb_kS0, j, j, 0);
            tile_regs_commit(); tile_regs_wait(); pack_tile(0, cb_t0); tile_regs_release();
        }
        cb_push_back(cb_t0, Dv_tiles);
        cb_pop_front(cb_kS0, Dv_tiles);
        cb_wait_front(cb_t0, Dv_tiles);
        binary_op_init_common(cb_v, cb_t0, cb_v);
        sub_tiles_init(cb_v, cb_t0);
        cb_reserve_back(cb_t1, Dv_tiles);
        for (uint32_t j = 0; j < Dv_tiles; j++) {
            tile_regs_acquire(); sub_tiles(cb_v, cb_t0, j, j, 0);
            tile_regs_commit(); tile_regs_wait(); pack_tile(0, cb_t1); tile_regs_release();
        }
        cb_push_back(cb_t1, Dv_tiles);
        cb_pop_front(cb_t0, Dv_tiles);
        cb_wait_front(cb_t1, Dv_tiles);
        binary_op_init_common(cb_betacol, cb_t1, cb_betacol);
        mul_tiles_init(cb_betacol, cb_t1);
        cb_reserve_back(cb_rhs, Dv_tiles);
        for (uint32_t j = 0; j < Dv_tiles; j++) {
            tile_regs_acquire(); mul_tiles(cb_betacol, cb_t1, j, j, 0);
            tile_regs_commit(); tile_regs_wait(); pack_tile(0, cb_rhs); tile_regs_release();
        }
        cb_push_back(cb_rhs, Dv_tiles);
        cb_pop_front(cb_t1, Dv_tiles);

        // U = inv @ rhs
        cb_wait_front(cb_inv, 1); cb_wait_front(cb_rhs, Dv_tiles);
        mm_init(cb_inv, cb_rhs, cb_U);
        cb_reserve_back(cb_U, Dv_tiles);
        for (uint32_t jv = 0; jv < Dv_tiles; jv++) {
            tile_regs_acquire(); matmul_tiles(cb_inv, cb_rhs, 0, jv, 0);
            tile_regs_commit(); tile_regs_wait(); pack_tile(0, cb_U); tile_regs_release();
        }
        cb_push_back(cb_U, Dv_tiles);
        cb_pop_front(cb_inv, 1); cb_pop_front(cb_rhs, Dv_tiles);

        // M = trili * (Qd @ KiT) -> cb_p (reuse)
        mm_init(cb_Qd, cb_KiT, cb_t0);
        cb_reserve_back(cb_t0, 1);
        tile_regs_acquire();
        for (uint32_t ik = 0; ik < Dk_tiles; ik++) matmul_tiles(cb_Qd, cb_KiT, ik, ik, 0);
        tile_regs_commit(); tile_regs_wait(); pack_tile(0, cb_t0); tile_regs_release();
        cb_push_back(cb_t0, 1);
        cb_wait_front(cb_t0, 1); cb_wait_front(cb_trili, 1);
        binary_op_init_common(cb_trili, cb_t0, cb_trili);
        mul_tiles_init(cb_trili, cb_t0);
        cb_reserve_back(cb_p, 1);
        tile_regs_acquire(); mul_tiles(cb_trili, cb_t0, 0, 0, 0);
        tile_regs_commit(); tile_regs_wait(); pack_tile(0, cb_p); tile_regs_release();
        cb_push_back(cb_p, 1);
        cb_pop_front(cb_t0, 1);

        // Oraw = dcol*qS0 + M@U  -> cb_kS0 (reuse as raw out)
        cb_wait_front(cb_p, 1); cb_wait_front(cb_U, Dv_tiles);
        mm_init(cb_p, cb_U, cb_t0);
        cb_reserve_back(cb_t0, Dv_tiles);
        for (uint32_t jv = 0; jv < Dv_tiles; jv++) {
            tile_regs_acquire(); matmul_tiles(cb_p, cb_U, 0, jv, 0);
            tile_regs_commit(); tile_regs_wait(); pack_tile(0, cb_t0); tile_regs_release();
        }
        cb_push_back(cb_t0, Dv_tiles);
        cb_pop_front(cb_p, 1);
        cb_wait_front(cb_qS0, Dv_tiles);
        binary_op_init_common(cb_dcol, cb_qS0, cb_dcol);
        mul_tiles_init(cb_dcol, cb_qS0);
        cb_reserve_back(cb_t1, Dv_tiles);
        for (uint32_t j = 0; j < Dv_tiles; j++) {
            tile_regs_acquire(); mul_tiles(cb_dcol, cb_qS0, j, j, 0);
            tile_regs_commit(); tile_regs_wait(); pack_tile(0, cb_t1); tile_regs_release();
        }
        cb_push_back(cb_t1, Dv_tiles);
        cb_pop_front(cb_qS0, Dv_tiles);
        cb_wait_front(cb_t0, Dv_tiles); cb_wait_front(cb_t1, Dv_tiles);
        binary_op_init_common(cb_t1, cb_t0, cb_t1);
        add_tiles_init(cb_t1, cb_t0);
        cb_reserve_back(cb_kS0, Dv_tiles);
        for (uint32_t j = 0; j < Dv_tiles; j++) {
            tile_regs_acquire(); add_tiles(cb_t1, cb_t0, j, j, 0);
            tile_regs_commit(); tile_regs_wait(); pack_tile(0, cb_kS0); tile_regs_release();
        }
        cb_push_back(cb_kS0, Dv_tiles);
        cb_pop_front(cb_t0, Dv_tiles); cb_pop_front(cb_t1, Dv_tiles);

        // Snew = dlast * (S0 + KiT@U)  -> new_state CB (compute-produced, writer/next-chunk consume)
        cb_wait_front(cb_KiT, Dk_tiles); cb_wait_front(cb_dlast, 1);
        cb_reserve_back(new_state, state_tiles);
        for (uint32_t ik = 0; ik < Dk_tiles; ik++) {
            for (uint32_t jv = 0; jv < Dv_tiles; jv++) {
                mm_init(cb_KiT, cb_U, cb_t0);
                cb_reserve_back(cb_t0, 1);
                tile_regs_acquire(); matmul_tiles(cb_KiT, cb_U, ik, jv, 0);
                tile_regs_commit(); tile_regs_wait(); pack_tile(0, cb_t0); tile_regs_release();
                cb_push_back(cb_t0, 1);
                cb_wait_front(cb_t0, 1);
                binary_op_init_common(cur_state, cb_t0, cur_state);
                add_tiles_init(cur_state, cb_t0);
                cb_reserve_back(cb_t1, 1);
                tile_regs_acquire(); add_tiles(cur_state, cb_t0, ik * Dv_tiles + jv, 0, 0);
                tile_regs_commit(); tile_regs_wait(); pack_tile(0, cb_t1); tile_regs_release();
                cb_push_back(cb_t1, 1);
                cb_pop_front(cb_t0, 1);
                cb_wait_front(cb_t1, 1);
                binary_op_init_common(cb_t1, cb_dlast, cb_t1);
                mul_tiles_bcast_scalar_init_short(cb_t1, cb_dlast);
                tile_regs_acquire(); mul_tiles_bcast_scalar(cb_t1, cb_dlast, 0, 0, 0);
                tile_regs_commit(); tile_regs_wait(); pack_tile(0, new_state); tile_regs_release();
                cb_pop_front(cb_t1, 1);
            }
        }
        cb_push_back(new_state, state_tiles);
        cb_pop_front(cb_U, Dv_tiles);
        cb_pop_front(cb_dlast, 1);

        // gated RMSNorm(Oraw=cb_kS0, z, norm_w) -> cb_output
        cb_wait_front(cb_kS0, Dv_tiles); cb_wait_front(cb_z, Dv_tiles);
        cb_wait_front(cb_scaler, 1); cb_wait_front(cb_eps, 1); cb_wait_front(cb_normw, Dv_tiles);
        binary_op_init_common(cb_kS0, cb_kS0, cb_kS0);
        mul_tiles_init(cb_kS0, cb_kS0);
        cb_reserve_back(cb_t0, Dv_tiles);
        for (uint32_t j = 0; j < Dv_tiles; j++) {
            tile_regs_acquire(); mul_tiles(cb_kS0, cb_kS0, j, j, 0);
            tile_regs_commit(); tile_regs_wait(); pack_tile(0, cb_t0); tile_regs_release();
        }
        cb_push_back(cb_t0, Dv_tiles);
        cb_wait_front(cb_t0, Dv_tiles);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_t0, cb_scaler, cb_t1);
        cb_reserve_back(cb_t1, 1);
        tile_regs_acquire();
        for (uint32_t j = 0; j < Dv_tiles; j++) reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_t0, cb_scaler, j, 0, 0);
        tile_regs_commit(); tile_regs_wait(); pack_tile(0, cb_t1); tile_regs_release();
        cb_push_back(cb_t1, 1);
        cb_pop_front(cb_t0, Dv_tiles);
        reduce_uninit();
        cb_wait_front(cb_t1, 1);
        binary_op_init_common(cb_t1, cb_eps, cb_t1);
        add_tiles_init(cb_t1, cb_eps);
        cb_reserve_back(cb_t0, 1);
        tile_regs_acquire(); add_tiles(cb_t1, cb_eps, 0, 0, 0); rsqrt_tile_init(); rsqrt_tile(0);
        tile_regs_commit(); tile_regs_wait(); pack_tile(0, cb_t0); tile_regs_release();
        cb_push_back(cb_t0, 1);
        cb_pop_front(cb_t1, 1);
        cb_wait_front(cb_t0, 1);
        binary_op_init_common(cb_kS0, cb_t0, cb_kS0);
        mul_bcast_cols_init_short(cb_kS0, cb_t0);
        cb_reserve_back(cb_t1, Dv_tiles);
        for (uint32_t j = 0; j < Dv_tiles; j++) {
            tile_regs_acquire(); mul_tiles_bcast_cols(cb_kS0, cb_t0, j, 0, 0);
            tile_regs_commit(); tile_regs_wait(); pack_tile(0, cb_t1); tile_regs_release();
        }
        cb_push_back(cb_t1, Dv_tiles);
        cb_pop_front(cb_t0, 1); cb_pop_front(cb_kS0, Dv_tiles);
        cb_wait_front(cb_t1, Dv_tiles);
        binary_op_init_common(cb_t1, cb_normw, cb_t1);
        mul_bcast_rows_init_short(cb_t1, cb_normw);
        cb_reserve_back(cb_t0, Dv_tiles);
        for (uint32_t j = 0; j < Dv_tiles; j++) {
            tile_regs_acquire(); mul_tiles_bcast_rows(cb_t1, cb_normw, j, j, 0);
            tile_regs_commit(); tile_regs_wait(); pack_tile(0, cb_t0); tile_regs_release();
        }
        cb_push_back(cb_t0, Dv_tiles);     // cb_t0 = normed pre-gate (O*rsqrt*normw)
        cb_pop_front(cb_t1, Dv_tiles);
        // silu(z) -> cb_t1. unary_op_init_common reconfigs unpack/pack for the SFPU
        // (the existing prefill kernel does this; omitting it leaves stale matmul config).
        unary_op_init_common(cb_z, cb_t1);
        copy_tile_to_dst_init_short(cb_z); silu_tile_init();
        cb_reserve_back(cb_t1, Dv_tiles);
        for (uint32_t j = 0; j < Dv_tiles; j++) {
            tile_regs_acquire(); copy_tile(cb_z, j, 0); silu_tile(0);
            tile_regs_commit(); tile_regs_wait(); pack_tile(0, cb_t1); tile_regs_release();
        }
        cb_push_back(cb_t1, Dv_tiles);
        // output = normed * silu(z)
        cb_wait_front(cb_t0, Dv_tiles); cb_wait_front(cb_t1, Dv_tiles);
        binary_op_init_common(cb_t0, cb_t1, cb_output);
        mul_tiles_init(cb_t0, cb_t1);
        cb_reserve_back(cb_output, Dv_tiles);
        for (uint32_t j = 0; j < Dv_tiles; j++) {
            tile_regs_acquire(); mul_tiles(cb_t0, cb_t1, j, j, 0);
            tile_regs_commit(); tile_regs_wait(); pack_tile(0, cb_output); tile_regs_release();
        }
        cb_push_back(cb_output, Dv_tiles);
        cb_pop_front(cb_t0, Dv_tiles); cb_pop_front(cb_t1, Dv_tiles);
        cb_pop_front(cb_z, Dv_tiles); cb_pop_front(cb_scaler, 1);
        cb_pop_front(cb_eps, 1); cb_pop_front(cb_normw, Dv_tiles);

        // consume entering state (new_state already produced above)
        cb_pop_front(cur_state, state_tiles);

        cb_pop_front(cb_k, Dk_tiles); cb_pop_front(cb_q, Dk_tiles); cb_pop_front(cb_v, Dv_tiles);
        cb_pop_front(cb_Kdec, Dk_tiles); cb_pop_front(cb_KiT, Dk_tiles); cb_pop_front(cb_Qd, Dk_tiles);
        cb_pop_front(cb_dcol, Dv_tiles); cb_pop_front(cb_betacol, Dv_tiles);
    }
}
