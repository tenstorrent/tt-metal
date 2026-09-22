// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/kernel/compute/dest_format_helpers.hpp"

// Fused partial RoPE compute for a width-sharded input: this core owns all Ht row-tiles but only
// Wt_local column-tiles of the last dim. The last dim repeats every `tiles_per_head` tiles:
//   tile_in_head < nope_Wt  -> pass-through
//   otherwise               -> in * cos + (in @ trans_mat) * sin
// using cos/sin tile (tile_in_head - nope_Wt). The rotation is block-diagonal per tile, so each
// rope tile rotates independently of the split. A core may straddle several head-block
// boundaries; runs are processed without crossing a head.
//
// A ROW_MAJOR X is 1x32 while cos/sin/trans_mat stay 32x32, so each stage re-programs the source
// tile/face geometry it needs (see the height-sharded kernel for why the matmul cannot inherit it).
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
    // When set, cos/sin hold a single tile-row that is broadcast across all input rows.
    constexpr bool cos_bcast = get_compile_time_arg_val(10) != 0;
    constexpr uint32_t tiles_per_head = get_compile_time_arg_val(11);
    constexpr uint32_t nope_Wt = get_compile_time_arg_val(12);
    constexpr uint32_t rope_Wt = get_compile_time_arg_val(13);

    uint32_t argrt = 0;
    const uint32_t g0 = get_arg_val<uint32_t>(argrt++);
    const uint32_t rope_tiles_local = get_arg_val<uint32_t>(argrt++);

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
    const uint32_t cos_sin_tiles = rope_Wt * (cos_bcast ? 1 : Ht);
    if (rope_tiles_local > 0) {
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
        const uint32_t cos_base = cos_bcast ? 0 : rt * rope_Wt;

        uint32_t j = 0;
        while (j < Wt_local) {
            const uint32_t t = (g0 + j) % tiles_per_head;
            if (t < nope_Wt) {
                const uint32_t run = (nope_Wt - t) < (Wt_local - j) ? (nope_Wt - t) : (Wt_local - j);
                reconfig_full_operand_srca(in_cb);
                copy_tile_init_with_dt(in_cb);
                for (uint32_t base = 0; base < run; base += kDstBatch) {
                    const uint32_t g = (run - base) < kDstBatch ? (run - base) : kDstBatch;
                    tile_regs_acquire();
                    for (uint32_t k = 0; k < g; ++k) {
                        copy_tile(in_cb, row_base + j + base + k, k);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t k = 0; k < g; ++k) {
                        pack_tile(k, out_cb, row_base + j + base + k);
                    }
                    tile_regs_release();
                }
                j += run;
                continue;
            }

            const uint32_t run = (tiles_per_head - t) < (Wt_local - j) ? (tiles_per_head - t) : (Wt_local - j);
            const uint32_t rope_base = row_base + j;
            const uint32_t cos_off = cos_base + (t - nope_Wt);

            // rotated = in_rope @ trans_mat
            reconfig_full_operand_srca(trans_mat_cb);
            reconfig_full_operand_srcb(in_cb);
            matmul_init(in_cb, trans_mat_cb);
            rotated_interm_cb_obj.reserve_back(run);
            for (uint32_t base = 0; base < run; base += kDstBatch) {
                const uint32_t g = (run - base) < kDstBatch ? (run - base) : kDstBatch;
                tile_regs_acquire();
                for (uint32_t k = 0; k < g; ++k) {
                    matmul_tiles(in_cb, trans_mat_cb, rope_base + base + k, 0, k);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t k = 0; k < g; ++k) {
                    pack_tile(k, rotated_interm_cb, base + k);
                }
                tile_regs_release();
            }
            rotated_interm_cb_obj.push_back(run);
            rotated_interm_cb_obj.wait_front(run);

            reconfig_full_operand_srca(rotated_interm_cb);
            reconfig_full_operand_srcb(sin_cb);
            if constexpr (cos_bcast) {
                mul_bcast_rows_init(rotated_interm_cb, sin_cb);
            } else {
                mul_init(rotated_interm_cb, sin_cb);
            }
            sin_interm_cb_obj.reserve_back(run);
            for (uint32_t base = 0; base < run; base += kDstBatch) {
                const uint32_t g = (run - base) < kDstBatch ? (run - base) : kDstBatch;
                tile_regs_acquire();
                for (uint32_t k = 0; k < g; ++k) {
                    if constexpr (cos_bcast) {
                        mul_tiles_bcast_rows(rotated_interm_cb, sin_cb, base + k, cos_off + base + k, k);
                    } else {
                        mul_tiles(rotated_interm_cb, sin_cb, base + k, cos_off + base + k, k);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t k = 0; k < g; ++k) {
                    pack_tile(k, sin_interm_cb, base + k);
                }
                tile_regs_release();
            }
            sin_interm_cb_obj.push_back(run);
            rotated_interm_cb_obj.pop_front(run);

            reconfig_full_operand_srca(in_cb);
            reconfig_full_operand_srcb(cos_cb);
            if constexpr (cos_bcast) {
                mul_bcast_rows_init(in_cb, cos_cb);
            } else {
                mul_init(in_cb, cos_cb);
            }
            cos_interm_cb_obj.reserve_back(run);
            for (uint32_t base = 0; base < run; base += kDstBatch) {
                const uint32_t g = (run - base) < kDstBatch ? (run - base) : kDstBatch;
                tile_regs_acquire();
                for (uint32_t k = 0; k < g; ++k) {
                    if constexpr (cos_bcast) {
                        mul_tiles_bcast_rows(in_cb, cos_cb, rope_base + base + k, cos_off + base + k, k);
                    } else {
                        mul_tiles(in_cb, cos_cb, rope_base + base + k, cos_off + base + k, k);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t k = 0; k < g; ++k) {
                    pack_tile(k, cos_interm_cb, base + k);
                }
                tile_regs_release();
            }
            cos_interm_cb_obj.push_back(run);

            sin_interm_cb_obj.wait_front(run);
            cos_interm_cb_obj.wait_front(run);
            reconfig_full_operand_srca(cos_interm_cb);
            reconfig_full_operand_srcb(sin_interm_cb);
            add_init(cos_interm_cb, sin_interm_cb);
            for (uint32_t base = 0; base < run; base += kDstBatch) {
                const uint32_t g = (run - base) < kDstBatch ? (run - base) : kDstBatch;
                tile_regs_acquire();
                for (uint32_t k = 0; k < g; ++k) {
                    add_tiles(cos_interm_cb, sin_interm_cb, base + k, base + k, k);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t k = 0; k < g; ++k) {
                    pack_tile(k, out_cb, rope_base + base + k);
                }
                tile_regs_release();
            }
            sin_interm_cb_obj.pop_front(run);
            cos_interm_cb_obj.pop_front(run);
            j += run;
        }
    }

    out_cb_obj.push_back(shard_tiles);
    if (rope_tiles_local > 0) {
        cos_cb_obj.pop_front(cos_sin_tiles);
        sin_cb_obj.pop_front(cos_sin_tiles);
        trans_mat_cb_obj.pop_front(onetile);
    }
}
