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

// Fused partial RoPE compute for a width-sharded input: this core owns all Ht row-tiles but only
// Wt_local column-tiles of the head dim. Its leading `nope_local` tiles fall in the pass-through
// region and its trailing `rope_local` tiles in the rope region (one of the two may be empty):
//   out[.., nope tile]  = in[.., nope tile]
//   out[.., rope tile]  = in * cos + (in @ trans_mat) * sin
// The rotation is block-diagonal per tile, so each rope tile rotates independently of the split.
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
    constexpr uint32_t Ht = get_compile_time_arg_val(8);
    constexpr uint32_t Wt_local = get_compile_time_arg_val(9);
    // When set, cos/sin hold a single tile-row that is broadcast across all 32 input rows.
    constexpr bool cos_bcast = get_compile_time_arg_val(10) != 0;

    // This core's split of its own column slice; both are runtime args because they depend on
    // where the shard sits relative to the global nope/rope boundary.
    uint32_t argrt = 0;
    const uint32_t nope_local = get_arg_val<uint32_t>(argrt++);
    const uint32_t rope_local = get_arg_val<uint32_t>(argrt++);

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
    compute_kernel_hw_startup(in_cb, cos_cb, out_cb);

    // trans_mat + cos/sin are streamed in from DRAM by the reader (nothing at all for a core with
    // no rope columns).
    const uint32_t cos_sin_tiles = rope_local * (cos_bcast ? 1 : Ht);
    if (rope_local > 0) {
        trans_mat_cb_obj.wait_front(onetile);
        cos_cb_obj.wait_front(cos_sin_tiles);
        sin_cb_obj.wait_front(cos_sin_tiles);
    }

    // X is the resident L1 shard (globally-allocated CB); signal it available.
    constexpr uint32_t shard_tiles = Ht * Wt_local;
    in_cb_obj.reserve_back(shard_tiles);
    in_cb_obj.push_back(shard_tiles);
    in_cb_obj.wait_front(shard_tiles);
    out_cb_obj.reserve_back(shard_tiles);

    for (uint32_t rt = 0; rt < Ht; ++rt) {
        const uint32_t row_base = rt * Wt_local;

        // 1) Pass-through this row-tile's leading "nope" tiles.
        copy_tile_init_with_dt(in_cb);
        for (uint32_t base = 0; base < nope_local; base += kDstBatch) {
            const uint32_t g = (nope_local - base) < kDstBatch ? (nope_local - base) : kDstBatch;
            tile_regs_acquire();
            for (uint32_t j = 0; j < g; ++j) {
                copy_tile(in_cb, row_base + base + j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < g; ++j) {
                pack_tile(j, out_cb, row_base + base + j);
            }
            tile_regs_release();
        }

        if (rope_local == 0) {
            continue;
        }

        const uint32_t rope_base = row_base + nope_local;
        const uint32_t cos_base = cos_bcast ? 0 : rt * rope_local;

        // 2) Rotate this row-tile's rope tiles: rotated = in_rope @ trans_mat.
        matmul_init(in_cb, trans_mat_cb);
        rotated_interm_cb_obj.reserve_back(rope_local);
        for (uint32_t base = 0; base < rope_local; base += kDstBatch) {
            const uint32_t g = (rope_local - base) < kDstBatch ? (rope_local - base) : kDstBatch;
            tile_regs_acquire();
            for (uint32_t j = 0; j < g; ++j) {
                matmul_tiles(in_cb, trans_mat_cb, rope_base + base + j, 0, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < g; ++j) {
                pack_tile(j, rotated_interm_cb, base + j);
            }
            tile_regs_release();
        }
        rotated_interm_cb_obj.push_back(rope_local);
        rotated_interm_cb_obj.wait_front(rope_local);

        // sin_interm = rotated * sin  (broadcast sin's single row across all input rows if cos_bcast)
        if constexpr (cos_bcast) {
            mul_bcast_rows_init(rotated_interm_cb, sin_cb);
        } else {
            mul_init(rotated_interm_cb, sin_cb);
        }
        sin_interm_cb_obj.reserve_back(rope_local);
        for (uint32_t base = 0; base < rope_local; base += kDstBatch) {
            const uint32_t g = (rope_local - base) < kDstBatch ? (rope_local - base) : kDstBatch;
            tile_regs_acquire();
            for (uint32_t j = 0; j < g; ++j) {
                if constexpr (cos_bcast) {
                    mul_tiles_bcast_rows(rotated_interm_cb, sin_cb, base + j, cos_base + base + j, j);
                } else {
                    mul_tiles(rotated_interm_cb, sin_cb, base + j, cos_base + base + j, j);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < g; ++j) {
                pack_tile(j, sin_interm_cb, base + j);
            }
            tile_regs_release();
        }
        sin_interm_cb_obj.push_back(rope_local);
        rotated_interm_cb_obj.pop_front(rope_local);

        // cos_interm = in_rope * cos  (broadcast cos's single row across all input rows if cos_bcast)
        if constexpr (cos_bcast) {
            mul_bcast_rows_init(in_cb, cos_cb);
        } else {
            mul_init(in_cb, cos_cb);
        }
        cos_interm_cb_obj.reserve_back(rope_local);
        for (uint32_t base = 0; base < rope_local; base += kDstBatch) {
            const uint32_t g = (rope_local - base) < kDstBatch ? (rope_local - base) : kDstBatch;
            tile_regs_acquire();
            for (uint32_t j = 0; j < g; ++j) {
                if constexpr (cos_bcast) {
                    mul_tiles_bcast_rows(in_cb, cos_cb, rope_base + base + j, cos_base + base + j, j);
                } else {
                    mul_tiles(in_cb, cos_cb, rope_base + base + j, cos_base + base + j, j);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < g; ++j) {
                pack_tile(j, cos_interm_cb, base + j);
            }
            tile_regs_release();
        }
        cos_interm_cb_obj.push_back(rope_local);

        // out_rope = cos_interm + sin_interm
        sin_interm_cb_obj.wait_front(rope_local);
        cos_interm_cb_obj.wait_front(rope_local);
        add_init(cos_interm_cb, sin_interm_cb);
        for (uint32_t base = 0; base < rope_local; base += kDstBatch) {
            const uint32_t g = (rope_local - base) < kDstBatch ? (rope_local - base) : kDstBatch;
            tile_regs_acquire();
            for (uint32_t j = 0; j < g; ++j) {
                add_tiles(cos_interm_cb, sin_interm_cb, base + j, base + j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < g; ++j) {
                pack_tile(j, out_cb, rope_base + base + j);
            }
            tile_regs_release();
        }
        sin_interm_cb_obj.pop_front(rope_local);
        cos_interm_cb_obj.pop_front(rope_local);
    }

    out_cb_obj.push_back(shard_tiles);
    if (rope_local > 0) {
        cos_cb_obj.pop_front(cos_sin_tiles);
        sin_cb_obj.pop_front(cos_sin_tiles);
        trans_mat_cb_obj.pop_front(onetile);
    }
}
