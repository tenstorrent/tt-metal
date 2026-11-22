// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The compute API and the two block helpers the recipe kernels take from outside the recipe core (recipe_step.inc's
// reduce-path copy init, the sparse ring wrapper's zero block). Every recipe compute kernel includes this first, then
// the recipe headers. SDPA decode keeps its own block primitives (sdpa_decode/.../compute/sdpa_block_ops.hpp).

#pragma once

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/debug/assert.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/reduce_custom.h"
#include "api/dataflow/circular_buffer.h"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/q_chunk_remapping.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/chunked_prefill_utils.hpp"
#include "cpp/ttnn/kernel_lib/dest_helpers.hpp"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "experimental/llk_sfpu/ckernel_sfpu_sdpa.h"
#endif

ALWI void sdpa_reduce_copy_tile_to_dst_init_short(uint32_t cbid, uint32_t transpose = 0) {
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        transpose, true /*transpose within 16x16 face*/, cbid)));

    MATH((llk_math_eltwise_unary_datacopy_init<
          DataCopyType::A2D,
          DST_ACCUM_MODE,
          BroadcastType::NONE,
          false,  // is_int_fpu_en
          PackMode::Default>(cbid)));
}

// Pushes num_tiles zero tiles to out_cb, made as x - x from the front tile of like_cb (which stays).
void zero_block(uint32_t like_cb, uint32_t out_cb, uint32_t num_tiles) {
    CircularBuffer cb_like(like_cb);
    CircularBuffer cb_out(out_cb);
    sub_init(like_cb, like_cb);
    cb_like.wait_front(1);
    cb_out.reserve_back(num_tiles);
    tile_regs_acquire();
    sub_tiles(like_cb, like_cb, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < num_tiles; ++i) {
        pack_tile(0, out_cb);
    }
    tile_regs_release();
    cb_out.push_back(num_tiles);
}
