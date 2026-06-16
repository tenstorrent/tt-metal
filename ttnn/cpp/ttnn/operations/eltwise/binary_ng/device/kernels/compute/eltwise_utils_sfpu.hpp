// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

#define PREPROCESS(op, ...) P_CAT(PREPROCESS_, HAS_ACTIVATIONS(op))(op, __VA_ARGS__)
#define PREPROCESS_0(...)
#define PREPROCESS_1(op, cb_pre_id, cb_post_id, cb_out_id, per_core_block_size)    \
    do {                                                                           \
        using namespace ckernel;                                                   \
                                                                                   \
        CircularBuffer cb_pre(cb_pre_id);                                          \
        CircularBuffer cb_post(cb_post_id);                                        \
                                                                                   \
        pack_reconfig_data_format(/*old*/ cb_out_id, /*new*/ cb_post.get_cb_id()); \
                                                                                   \
        cb_pre.wait_front(per_core_block_size);                                    \
        cb_post.reserve_back(per_core_block_size);                                 \
                                                                                   \
        tile_regs_acquire();                                                       \
        for (uint32_t i = 0; i < per_core_block_size; ++i) {                       \
            copy_tile_to_dst_init_short(cb_pre.get_cb_id());                       \
            copy_tile(cb_pre.get_cb_id(), i, i);                                   \
            PROCESS_ACTIVATIONS(op, i);                                            \
        }                                                                          \
        tile_regs_commit();                                                        \
                                                                                   \
        tile_regs_wait();                                                          \
        for (uint32_t i = 0; i < per_core_block_size; ++i) {                       \
            pack_tile(i, cb_post.get_cb_id());                                     \
        }                                                                          \
        tile_regs_release();                                                       \
                                                                                   \
        cb_pre.pop_front(per_core_block_size);                                     \
        cb_post.push_back(per_core_block_size);                                    \
                                                                                   \
        pack_reconfig_data_format(/*old*/ cb_post.get_cb_id(), /*new*/ cb_out_id); \
    } while (0)
