// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"

#if defined(ADD_INT32_INIT) || defined(BITWISE_INIT) || defined(SHIFT_INIT)
#define INT32_INIT
#endif

#define PREPROCESS(init, apply, cb_pre, cb_post, cb_out, per_core_block_size) \
    do {                                                                      \
        using namespace ckernel;                                              \
                                                                              \
        copy_tile_to_dst_init_short();                                        \
                                                                              \
        reconfig_data_format_srca(/*old*/ cb_post, /*new*/ cb_pre);           \
        pack_reconfig_data_format(/*old*/ cb_out, /*new*/ cb_post);           \
                                                                              \
        cb_wait_front(cb_pre, per_core_block_size);                           \
        cb_reserve_back(cb_post, per_core_block_size);                        \
                                                                              \
        tile_regs_acquire();                                                  \
        init();                                                               \
        for (uint32_t i = 0; i < per_core_block_size; ++i) {                  \
            copy_tile(cb_pre, i, i);                                          \
            apply(i);                                                         \
        }                                                                     \
        tile_regs_commit();                                                   \
                                                                              \
        tile_regs_wait();                                                     \
        for (uint32_t i = 0; i < per_core_block_size; ++i) {                  \
            pack_tile(i, cb_post); /* DST[0]->cb */                           \
        }                                                                     \
        tile_regs_release();                                                  \
                                                                              \
        cb_pop_front(cb_pre, per_core_block_size);                            \
        cb_push_back(cb_post, per_core_block_size);                           \
                                                                              \
        reconfig_data_format_srca(/*old*/ cb_pre, /*new*/ cb_post);           \
        pack_reconfig_data_format(/*old*/ cb_post, /*new*/ cb_out);           \
    } while (0)

#define PREPROCESS_SFPU_A(cb_pre, cb_post, cb_out, per_core_block_size) \
    do {                                                                \
        using namespace ckernel;                                        \
                                                                        \
        copy_tile_to_dst_init_short();                                  \
                                                                        \
        cb_wait_front(cb_pre, per_core_block_size);                     \
        cb_reserve_back(cb_post, per_core_block_size);                  \
                                                                        \
        tile_regs_acquire();                                            \
        SFPU_OP_INIT_PRE_IN0_0                                          \
        for (uint32_t i = 0; i < per_core_block_size; ++i) {            \
            copy_tile(cb_pre, i, i);                                    \
            SFPU_OP_FUNC_PRE_IN0_0                                      \
        }                                                               \
        tile_regs_commit();                                             \
                                                                        \
        tile_regs_wait();                                               \
        for (uint32_t i = 0; i < per_core_block_size; ++i) {            \
            pack_tile(i, cb_post); /* DST[0]->cb */                     \
        }                                                               \
        tile_regs_release();                                            \
                                                                        \
        cb_pop_front(cb_pre, per_core_block_size);                      \
        cb_push_back(cb_post, per_core_block_size);                     \
    } while (0)

#define PREPROCESS_SFPU_B(cb_pre, cb_post, cb_out, per_core_block_size) \
    do {                                                                \
        using namespace ckernel;                                        \
                                                                        \
        copy_tile_to_dst_init_short();                                  \
                                                                        \
        cb_wait_front(cb_pre, per_core_block_size);                     \
        cb_reserve_back(cb_post, per_core_block_size);                  \
                                                                        \
        tile_regs_acquire();                                            \
        SFPU_OP_INIT_PRE_IN1_0                                          \
        for (uint32_t i = 0; i < per_core_block_size; ++i) {            \
            copy_tile(cb_pre, i, i);                                    \
            SFPU_OP_FUNC_PRE_IN1_0                                      \
        }                                                               \
        tile_regs_commit();                                             \
                                                                        \
        tile_regs_wait();                                               \
        for (uint32_t i = 0; i < per_core_block_size; ++i) {            \
            pack_tile(i, cb_post); /* DST[0]->cb */                     \
        }                                                               \
        tile_regs_release();                                            \
                                                                        \
        cb_pop_front(cb_pre, per_core_block_size);                      \
        cb_push_back(cb_post, per_core_block_size);                     \
    } while (0)
