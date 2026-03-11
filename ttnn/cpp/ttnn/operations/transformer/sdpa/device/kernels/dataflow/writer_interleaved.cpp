// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    constexpr uint32_t vDHt = get_compile_time_arg_val(7);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(8);
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(12);
    constexpr bool use_lightweight_mask = get_compile_time_arg_val(20) == 1;

    const uint32_t local_batch_start = get_arg_val<uint32_t>(2);
    const uint32_t local_batch_end = get_arg_val<uint32_t>(3);
    const uint32_t local_nh_start = get_arg_val<uint32_t>(4);
    const uint32_t local_nh_end = get_arg_val<uint32_t>(5);
    const uint32_t local_q_start = get_arg_val<uint32_t>(6);
    const uint32_t local_q_end = get_arg_val<uint32_t>(7);
    const uint32_t num_phases = get_arg_val<uint32_t>(8);

    const uint32_t q_chunks_per_core = local_q_end - local_q_start;

    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_7;

    generate_reduce_scaler(cb_identity_scale_in, identity_scalar_packed);
    generate_bcast_col_scalar(cb_col_identity, identity_scalar_packed);

    if constexpr (use_lightweight_mask) {
        const uint32_t mask_tile_size_bytes = get_tile_size(cb_mask_in);
        cb_reserve_back(cb_mask_in, 1);
        auto* ptr = reinterpret_cast<uint32_t*>(get_write_ptr(cb_mask_in));
        for (uint32_t i = 0; i < mask_tile_size_bytes / sizeof(uint32_t); i++) {
            ptr[i] = 0xFF80FF80;
        }
        cb_push_back(cb_mask_in, 1);
    }

    for (uint32_t phase = 0; phase < num_phases; ++phase) {
        for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
            for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
                for (uint32_t q_iter = 0; q_iter < q_chunks_per_core; ++q_iter) {
                    cb_wait_front(cb_out, out_chunk_tiles);
                    cb_pop_front(cb_out, out_chunk_tiles);
                }
            }
        }
    }
}
