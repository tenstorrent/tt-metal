// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_SCALAR

#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/rsqrt.h"

template <
    uint32_t input_cb,
    uint32_t scalars_cb,
    uint32_t interm_cb,
    uint32_t weight_cb,
    uint32_t output_cb,
    bool fp32_acc,
    uint32_t num_tiles,
    uint32_t epsilon_index,
    uint32_t scalar_index,
    bool pop_input>
void compute_rmsnorm() {
    // TODO: #32998: Fuse this without having to spill output of square to interm cb
    {
        // Square the input
        mul_tiles_init(input_cb, input_cb);
        cb_wait_front(input_cb, num_tiles);
        cb_reserve_back(interm_cb, num_tiles + 1);  // Plus 1 for the RMS tile
        tile_regs_acquire();
        for (uint32_t i = 0; i < num_tiles; i++) {
            mul_tiles(input_cb, input_cb, i, i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile_block(0, interm_cb, num_tiles);
        tile_regs_release();
        cb_push_back(interm_cb, num_tiles);

        // Calculate the avg of the sum of the squares
        cb_wait_front(interm_cb, num_tiles);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR, fp32_acc>(interm_cb, scalars_cb, interm_cb);
        tile_regs_acquire();
        // TODO: #32998: Instead of accumulating to index 0, accumulate to num_tiles + 1 once bcast reuse is supported
        for (uint32_t i = 0; i < num_tiles; i++) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_SCALAR, fp32_acc>(interm_cb, scalars_cb, i, scalar_index, 0);
        }
    }
    // TODO: #32998: Avoid having to spill 1/RMS to interm cb
    {
        // Add epsilon
        DPRINT << "here1" << ENDL();
        binary_dest_reuse_tiles_init<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(scalars_cb);
        DPRINT << "here1.a" << ENDL();
        binary_dest_reuse_tiles<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(scalars_cb, epsilon_index, 0);
        DPRINT << "here1.b" << ENDL();
        // Calculate the 1/RMS
        // TODO: #32998: Use index num_tiles + 1 once bcast reuse is supported
        rsqrt_tile<false, true>(0);
        DPRINT << "here1.c" << ENDL();
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, interm_cb);
        tile_regs_release();
        reduce_uninit();
        DPRINT << "here1.d" << ENDL();
        cb_pop_front(interm_cb, num_tiles);
        cb_push_back(interm_cb, 1);  // 1/RMS tile should now be index 0
    }
    {
        DPRINT << "here2" << ENDL();
        // Multiply input by 1/RMS
        cb_wait_front(interm_cb, 1);
        cb_reserve_back(output_cb, num_tiles);
        DPRINT << "here2.a" << ENDL();
        mul_tiles_bcast_scalar_init_short(input_cb, interm_cb);
        DPRINT << "here2.b" << ENDL();
        tile_regs_acquire();
        DPRINT << "here2.c" << ENDL();
        for (uint32_t i = 0; i < num_tiles; i++) {
            // TODO: #32998: Once we have bcast reuse, we will use input_cb index i, reuse dst index num_tiles + 1,
            // output dst index i
            mul_tiles_bcast_scalar(input_cb, interm_cb, i, 0, i);
        }
        if constexpr (pop_input) {
            cb_pop_front(input_cb, num_tiles);
        }
    }
    {
        DPRINT << "here3" << ENDL();
        // Multiply by the weight
        binary_dest_reuse_tiles_init<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(weight_cb);
        for (uint32_t i = 0; i < num_tiles; i++) {
            binary_dest_reuse_tiles<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(weight_cb, i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile_block(0, output_cb, num_tiles);
        tile_regs_release();
        cb_pop_front(interm_cb, 1);
        cb_push_back(output_cb, num_tiles);
    }
    DPRINT << "Done" << ENDL();
}

#define DEFINE_RMSNORM_COMPUTE_VARS(prefix)                                                                  \
    constexpr uint32_t prefix##_input_cb = get_named_compile_time_arg_val(#prefix "_input_cb");              \
    constexpr uint32_t prefix##_scalars_cb = get_named_compile_time_arg_val(#prefix "_scalars_cb");          \
    constexpr uint32_t prefix##_interm_cb = get_named_compile_time_arg_val(#prefix "_interm_cb");            \
    constexpr uint32_t prefix##_gamma_cb = get_named_compile_time_arg_val(#prefix "_gamma_cb");              \
    constexpr uint32_t prefix##_output_cb = get_named_compile_time_arg_val(#prefix "_output_cb");            \
    constexpr bool prefix##_fp32_acc = get_named_compile_time_arg_val(#prefix "_enforce_fp32_accumulation"); \
    constexpr uint32_t prefix##_num_tiles = get_named_compile_time_arg_val(#prefix "_num_tiles");            \
    constexpr uint32_t prefix##_epsilon_index = get_named_compile_time_arg_val(#prefix "_epsilon_index");    \
    constexpr uint32_t prefix##_scalar_index = get_named_compile_time_arg_val(#prefix "_scalar_index");

#define COMPUTE_RMSNORM(prefix, pop_input) \
    compute_rmsnorm<                       \
        prefix##_input_cb,                 \
        prefix##_scalars_cb,               \
        prefix##_interm_cb,                \
        prefix##_gamma_cb,                 \
        prefix##_output_cb,                \
        prefix##_fp32_acc,                 \
        prefix##_num_tiles,                \
        prefix##_epsilon_index,            \
        prefix##_scalar_index,             \
        pop_input>();
