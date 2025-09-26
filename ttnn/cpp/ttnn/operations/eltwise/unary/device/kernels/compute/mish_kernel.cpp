// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/log1p.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {

template <uint32_t num_tiles>
FORCE_INLINE void process_mish_tiles() {
    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    cb_wait_front(cb_input, num_tiles);
    tile_regs_acquire();

    copy_tile_to_dst_init_short(cb_input);
    copy_block_matmul_partials(cb_input, 0, 0, num_tiles);
    exp_tile_init<1u>();
    for (uint32_t i = 0; i < num_tiles; ++i) {
        exp_tile<1u>(i);
    }

    log1p_tile_init();
    for (uint32_t i = 0; i < num_tiles; ++i) {
        log1p_tile(i);
    }

    tanh_tile_init();
    for (uint32_t i = 0; i < num_tiles; ++i) {
        tanh_tile(i);
    }

    binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_input);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_input, i, i);
    }

    tile_regs_commit();
    tile_regs_wait();

    pack_tile_block(0, cb_output, num_tiles);

    tile_regs_release();
    cb_pop_front(cb_input, num_tiles);
}

void MAIN {
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;
    init_sfpu(cb_input, cb_output);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(cb_output, per_core_block_dim);
        constexpr uint32_t chunk_size = 8;
        constexpr uint32_t full_chunks = per_core_block_dim / chunk_size;
        constexpr uint32_t leftover_tiles = per_core_block_dim % chunk_size;
        // Process full chunks of 8 tiles
        for (uint32_t chunk_index = 0; chunk_index < full_chunks; ++chunk_index) {
            process_mish_tiles<chunk_size>();
        }
        if constexpr (leftover_tiles > 0) {
            process_mish_tiles<leftover_tiles>();
        }
        cb_push_back(cb_output, per_core_block_dim);
    }
}
}  // namespace NAMESPACE
