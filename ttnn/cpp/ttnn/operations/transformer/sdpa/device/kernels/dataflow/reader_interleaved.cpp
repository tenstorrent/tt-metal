// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t DHt = get_compile_time_arg_val(8);
    constexpr uint32_t vDHt = get_compile_time_arg_val(9);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(10);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(12);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(13);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(26);

    const uint32_t local_batch_start = get_arg_val<uint32_t>(8);
    const uint32_t local_batch_end = get_arg_val<uint32_t>(9);
    const uint32_t local_nh_start = get_arg_val<uint32_t>(10);
    const uint32_t local_nh_end = get_arg_val<uint32_t>(11);
    const uint32_t local_q_start = get_arg_val<uint32_t>(12);
    const uint32_t local_q_end = get_arg_val<uint32_t>(13);
    const uint32_t num_phases = get_arg_val<uint32_t>(14);

    const uint32_t q_chunks_per_core = local_q_end - local_q_start;

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t v_chunk_tiles = Sk_chunk_t * vDHt;

    constexpr uint32_t q_num_subblocks = Sq_chunk_t / qk_subblock_h;
    constexpr bool use_q_subblock_push = (q_num_subblocks > 1);

    for (uint32_t phase = 0; phase < num_phases; ++phase) {
        for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
            for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
                for (uint32_t q_iter = 0; q_iter < q_chunks_per_core; ++q_iter) {
                    if constexpr (!use_q_subblock_push) {
                        cb_reserve_back(cb_q_in, q_chunk_tiles);
                        cb_push_back(cb_q_in, q_chunk_tiles);
                    }

                    for (uint32_t k_chunk = 0; k_chunk < k_num_chunks; ++k_chunk) {
                        cb_reserve_back(cb_k_in, k_chunk_tiles);
                        cb_push_back(cb_k_in, k_chunk_tiles);

                        if constexpr (use_q_subblock_push) {
                            if (k_chunk == 0) {
                                for (uint32_t q_sub = 0; q_sub < q_num_subblocks; ++q_sub) {
                                    constexpr uint32_t sb_tiles = qk_subblock_h * DHt;
                                    cb_reserve_back(cb_q_in, sb_tiles);
                                    cb_push_back(cb_q_in, sb_tiles);
                                }
                            }
                        }

                        cb_reserve_back(cb_v_in, v_chunk_tiles);
                        cb_push_back(cb_v_in, v_chunk_tiles);
                    }
                }
            }
        }
    }
}
