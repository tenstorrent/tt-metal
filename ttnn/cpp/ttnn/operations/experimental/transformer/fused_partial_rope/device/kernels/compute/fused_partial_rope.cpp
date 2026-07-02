// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/kernel/compute/dest_format_helpers.hpp"

// Fused partial RoPE compute (one tile-row / 32 rows per core):
//   out[0 .. nope_Wt)      = in[0 .. nope_Wt)                       (pass-through)
//   out[nope_Wt .. Dt)     = in_rope * cos + (in_rope @ trans_mat) * sin
// where in_rope is the trailing rope_Wt tiles of the input row.
namespace {
// DST tile budget per acquire/commit batch (safe for fp32 dest-acc mode).
constexpr uint32_t kDstBatch = 8;
}  // namespace

void kernel_main() {
    constexpr uint32_t onetile = 1;
    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t cos_cb = get_compile_time_arg_val(1);
    constexpr uint32_t sin_cb = get_compile_time_arg_val(2);
    constexpr uint32_t trans_mat_cb = get_compile_time_arg_val(3);
    constexpr uint32_t rotated_interm_cb = get_compile_time_arg_val(4);
    constexpr uint32_t cos_interm_cb = get_compile_time_arg_val(5);
    constexpr uint32_t sin_interm_cb = get_compile_time_arg_val(6);
    constexpr uint32_t out_cb = get_compile_time_arg_val(7);
    constexpr uint32_t Dt = get_compile_time_arg_val(8);
    constexpr uint32_t rope_Wt = get_compile_time_arg_val(9);
    constexpr uint32_t nope_Wt = get_compile_time_arg_val(10);
    // When set, cos/sin hold a single tile-row that is broadcast across all 32 input rows.
    constexpr bool cos_bcast = get_compile_time_arg_val(11) != 0;

    CircularBuffer in_cb_obj(in_cb);
    CircularBuffer cos_cb_obj(cos_cb);
    CircularBuffer sin_cb_obj(sin_cb);
    CircularBuffer trans_mat_cb_obj(trans_mat_cb);
    CircularBuffer rotated_interm_cb_obj(rotated_interm_cb);
    CircularBuffer cos_interm_cb_obj(cos_interm_cb);
    CircularBuffer sin_interm_cb_obj(sin_interm_cb);
    CircularBuffer out_cb_obj(out_cb);

    compute_kernel_hw_startup<SrcOrder::Reverse>(in_cb, trans_mat_cb, out_cb);
    matmul_init(in_cb, trans_mat_cb);
    binary_op_init_common(in_cb, cos_cb, out_cb);

    // trans_mat + cos/sin are streamed in from DRAM by the reader.
    trans_mat_cb_obj.wait_front(onetile);
    cos_cb_obj.wait_front(rope_Wt);
    sin_cb_obj.wait_front(rope_Wt);

    // X is the resident L1 shard (globally-allocated CB); signal it available.
    in_cb_obj.reserve_back(Dt);
    in_cb_obj.push_back(Dt);
    in_cb_obj.wait_front(Dt);
    out_cb_obj.reserve_back(Dt);

    // 1) Pass-through the leading "nope" tiles: out[j] = in[j] for j in [0, nope_Wt).
    for (uint32_t base = 0; base < nope_Wt; base += kDstBatch) {
        const uint32_t g = (nope_Wt - base) < kDstBatch ? (nope_Wt - base) : kDstBatch;
        copy_tile_init_with_dt(in_cb);
        tile_regs_acquire();
        for (uint32_t j = 0; j < g; ++j) {
            copy_tile(in_cb, base + j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < g; ++j) {
            pack_tile(j, out_cb, base + j);
        }
        tile_regs_release();
    }

    // 2) Rotate the trailing rope tiles: rotated = in_rope @ trans_mat.
    matmul_init(in_cb, trans_mat_cb);
    rotated_interm_cb_obj.reserve_back(rope_Wt);
    tile_regs_acquire();
    for (uint32_t j = 0; j < rope_Wt; ++j) {
        matmul_tiles(in_cb, trans_mat_cb, nope_Wt + j, 0, j);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < rope_Wt; ++j) {
        pack_tile(j, rotated_interm_cb, j);
    }
    tile_regs_release();
    rotated_interm_cb_obj.push_back(rope_Wt);
    rotated_interm_cb_obj.wait_front(rope_Wt);

    // sin_interm = rotated * sin  (broadcast sin's single row across all input rows if cos_bcast)
    if constexpr (cos_bcast) {
        mul_bcast_rows_init_short(rotated_interm_cb, sin_cb);
    } else {
        mul_tiles_init(rotated_interm_cb, sin_cb);
    }
    sin_interm_cb_obj.reserve_back(rope_Wt);
    tile_regs_acquire();
    for (uint32_t j = 0; j < rope_Wt; ++j) {
        if constexpr (cos_bcast) {
            mul_tiles_bcast_rows(rotated_interm_cb, sin_cb, j, j, j);
        } else {
            mul_tiles(rotated_interm_cb, sin_cb, j, j, j);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < rope_Wt; ++j) {
        pack_tile(j, sin_interm_cb, j);
    }
    tile_regs_release();
    sin_interm_cb_obj.push_back(rope_Wt);
    rotated_interm_cb_obj.pop_front(rope_Wt);

    // cos_interm = in_rope * cos  (broadcast cos's single row across all input rows if cos_bcast)
    if constexpr (cos_bcast) {
        mul_bcast_rows_init_short(in_cb, cos_cb);
    } else {
        mul_tiles_init(in_cb, cos_cb);
    }
    cos_interm_cb_obj.reserve_back(rope_Wt);
    tile_regs_acquire();
    for (uint32_t j = 0; j < rope_Wt; ++j) {
        if constexpr (cos_bcast) {
            mul_tiles_bcast_rows(in_cb, cos_cb, nope_Wt + j, j, j);
        } else {
            mul_tiles(in_cb, cos_cb, nope_Wt + j, j, j);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < rope_Wt; ++j) {
        pack_tile(j, cos_interm_cb, j);
    }
    tile_regs_release();
    cos_interm_cb_obj.push_back(rope_Wt);

    // out_rope = cos_interm + sin_interm -> out[nope_Wt + j]
    sin_interm_cb_obj.wait_front(rope_Wt);
    cos_interm_cb_obj.wait_front(rope_Wt);
    add_tiles_init(cos_interm_cb, sin_interm_cb);
    tile_regs_acquire();
    for (uint32_t j = 0; j < rope_Wt; ++j) {
        add_tiles(cos_interm_cb, sin_interm_cb, j, j, j);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < rope_Wt; ++j) {
        pack_tile(j, out_cb, nope_Wt + j);
    }
    tile_regs_release();
    sin_interm_cb_obj.pop_front(rope_Wt);
    cos_interm_cb_obj.pop_front(rope_Wt);

    out_cb_obj.push_back(Dt);
    cos_cb_obj.pop_front(rope_Wt);
    sin_cb_obj.pop_front(rope_Wt);
    trans_mat_cb_obj.pop_front(onetile);
}
