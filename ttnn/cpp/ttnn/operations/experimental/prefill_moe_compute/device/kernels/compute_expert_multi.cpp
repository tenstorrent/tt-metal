// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Prefill MoE compute: Multi-expert two-phase expert compute kernel
// For each expert e = 0..num_experts-1:
//   Phase A: gate_up matmul [1, K_gu] x [K_gu, N_per_core_gu] -> [1, N_per_core_gu]
//            then SwiGLU on dest registers -> packs N_per_core_gu/2 tiles to CB_OUT
//   Phase B: down matmul [1, K_dn] x [K_dn, N_per_core_dn] -> [1, N_per_core_dn]
//            packs N_per_core_dn tiles to CB_OUT
//
// CB0 (activation): 1 BF16 tile per K iteration (reused both phases)
// CB1 (weights): max(N_per_core_gu, N_per_core_dn) tiles per K iteration (reused)
// CB2 (output): max(N_per_core_gu/2, N_per_core_dn) BF16 tiles (reused)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/matmul.h"

#ifdef TRISC_PACK
#include "swiglu_sfpu.h"
#endif

void kernel_main() {
    constexpr uint32_t num_k_tiles_gu = get_compile_time_arg_val(0);  // K dimension in tiles for gate_up
    constexpr uint32_t n_per_core_gu = get_compile_time_arg_val(1);   // gate_up weight tiles per core
    constexpr uint32_t n_per_core_dn = get_compile_time_arg_val(2);   // down weight tiles per core
    constexpr uint32_t num_experts = get_compile_time_arg_val(3);     // number of experts to process
    constexpr uint32_t N_BLOCK_GU = get_compile_time_arg_val(4);      // matmul block size for gate_up (ct_dim <= 7)
    constexpr uint32_t N_BLOCK_DN = get_compile_time_arg_val(5);      // matmul block size for down (ct_dim <= 7)
    constexpr uint32_t num_k_tiles_dn = get_compile_time_arg_val(6);  // K dimension in tiles for down
    constexpr uint32_t n_swiglu = n_per_core_gu / 2;                  // SwiGLU output tiles

    constexpr uint32_t n_blocks_gu = n_per_core_gu / N_BLOCK_GU;
    constexpr uint32_t n_blocks_dn = n_per_core_dn / N_BLOCK_DN;

    constexpr uint32_t cb_act = 0;
    constexpr uint32_t cb_weights = 1;
    constexpr uint32_t cb_out = 2;

    for (uint32_t expert = 0; expert < num_experts; ++expert) {
        // ========== Phase A: gate_up matmul + SwiGLU ==========
        mm_block_init(
            cb_act, cb_weights, cb_out, /*transpose=*/false, /*ct_dim=*/N_BLOCK_GU, /*rt_dim=*/1, /*kt_dim=*/1);

        tile_regs_acquire();

        for (uint32_t k = 0; k < num_k_tiles_gu; ++k) {
            cb_wait_front(cb_act, 1);
            cb_wait_front(cb_weights, n_per_core_gu);

            for (uint32_t b = 0; b < n_blocks_gu; ++b) {
                uint32_t n_offset = b * N_BLOCK_GU;
                matmul_block(
                    cb_act,
                    cb_weights,
                    /*in0_index=*/0,
                    /*in1_index=*/n_offset,
                    /*idst=*/n_offset,
                    /*transpose=*/false,
                    /*ct_dim=*/N_BLOCK_GU,
                    /*rt_dim=*/1,
                    /*kt_dim=*/1);
            }

            cb_pop_front(cb_act, 1);
            cb_pop_front(cb_weights, n_per_core_gu);
        }

        tile_regs_commit();
        tile_regs_wait();

#ifdef TRISC_PACK
        ckernel::llk_math_eltwise_binary_sfpu_swiglu_init<true>();
        for (uint32_t j = 0; j < n_swiglu; ++j) {
            ckernel::llk_math_eltwise_binary_sfpu_swiglu<true, false>(j, j + n_swiglu, j);
        }
#endif

        cb_reserve_back(cb_out, n_swiglu);
        for (uint32_t i = 0; i < n_swiglu; ++i) {
            pack_tile(i, cb_out);
        }
        cb_push_back(cb_out, n_swiglu);

        tile_regs_release();

        // ========== Phase B: down matmul ==========
        mm_block_init(
            cb_act, cb_weights, cb_out, /*transpose=*/false, /*ct_dim=*/N_BLOCK_DN, /*rt_dim=*/1, /*kt_dim=*/1);

        tile_regs_acquire();

        for (uint32_t k = 0; k < num_k_tiles_dn; ++k) {
            cb_wait_front(cb_act, 1);
            cb_wait_front(cb_weights, n_per_core_dn);

            for (uint32_t b = 0; b < n_blocks_dn; ++b) {
                uint32_t n_offset = b * N_BLOCK_DN;
                matmul_block(
                    cb_act,
                    cb_weights,
                    /*in0_index=*/0,
                    /*in1_index=*/n_offset,
                    /*idst=*/n_offset,
                    /*transpose=*/false,
                    /*ct_dim=*/N_BLOCK_DN,
                    /*rt_dim=*/1,
                    /*kt_dim=*/1);
            }

            cb_pop_front(cb_act, 1);
            cb_pop_front(cb_weights, n_per_core_dn);
        }

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_out, n_per_core_dn);
        for (uint32_t i = 0; i < n_per_core_dn; ++i) {
            pack_tile(i, cb_out);
        }
        cb_push_back(cb_out, n_per_core_dn);

        tile_regs_release();
    }
}
