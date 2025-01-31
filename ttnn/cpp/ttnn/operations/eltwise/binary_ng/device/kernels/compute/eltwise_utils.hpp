// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"

#define PREPROCESS(op, ...) P_CAT(PREPROCESS_, HAS_ACTIVATIONS(op))(op, __VA_ARGS__)
#define PREPROCESS_0(...)
#define PREPROCESS_1(op, cb_pre, cb_post, cb_out, per_core_block_size) \
    do {                                                               \
        using namespace ckernel;                                       \
                                                                       \
        reconfig_data_format_srca(/*old*/ cb_post, /*new*/ cb_pre);    \
        pack_reconfig_data_format(/*old*/ cb_out, /*new*/ cb_post);    \
                                                                       \
        cb_wait_front(cb_pre, per_core_block_size);                    \
        cb_reserve_back(cb_post, per_core_block_size);                 \
                                                                       \
        tile_regs_acquire();                                           \
        for (uint32_t i = 0; i < per_core_block_size; ++i) {           \
            copy_tile_to_dst_init_short(cb_pre);                       \
            copy_tile(cb_pre, i, i);                                   \
            PROCESS_ACTIVATIONS(op, i);                                \
        }                                                              \
        tile_regs_commit();                                            \
                                                                       \
        tile_regs_wait();                                              \
        for (uint32_t i = 0; i < per_core_block_size; ++i) {           \
            pack_tile(i, cb_post);                                     \
        }                                                              \
        tile_regs_release();                                           \
                                                                       \
        cb_pop_front(cb_pre, per_core_block_size);                     \
        cb_push_back(cb_post, per_core_block_size);                    \
                                                                       \
        reconfig_data_format_srca(/*old*/ cb_pre, /*new*/ cb_post);    \
        pack_reconfig_data_format(/*old*/ cb_post, /*new*/ cb_out);    \
    } while (0)
