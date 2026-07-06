// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/kernel/compute/dest_format_helpers.hpp"

void kernel_main() {
    constexpr uint32_t onetile = 1;
    constexpr uint32_t in_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t cos_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t sin_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t scalar_cb_id = get_compile_time_arg_val(3);

    constexpr uint32_t rotated_in_interm_cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t cos_interm_cb_id = get_compile_time_arg_val(5);
    constexpr uint32_t sin_interm_cb_id = get_compile_time_arg_val(6);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(7);
    constexpr uint32_t Wt = get_compile_time_arg_val(8);
    constexpr uint32_t Ht = get_compile_time_arg_val(9);  // Total rows (tiles) owned by this core
    constexpr uint32_t heads_per_batch_t = get_compile_time_arg_val(10);
    constexpr uint32_t batch_per_core = get_compile_time_arg_val(11);
    constexpr uint32_t half_Wt = Wt / 2;
    (void)Ht;

    CircularBuffer in_cb(in_cb_id);
    CircularBuffer cos_cb(cos_cb_id);
    CircularBuffer sin_cb(sin_cb_id);
    CircularBuffer scalar_cb(scalar_cb_id);
    CircularBuffer rotated_in_interm_cb(rotated_in_interm_cb_id);
    CircularBuffer cos_interm_cb(cos_interm_cb_id);
    CircularBuffer sin_interm_cb(sin_interm_cb_id);
    CircularBuffer out_cb(out_cb_id);

    binary_op_init_common(in_cb_id, sin_cb_id, sin_interm_cb_id);  // General Init for all binary ops

    // Wait for the reader kernel (reader_rotary_embedding_hf_sharded.cpp) to
    // write -1.0 into the scalar CB and push it.
    scalar_cb.wait_front(onetile);

    for (uint32_t batch_idx = 0; batch_idx < batch_per_core; ++batch_idx) {
        // For decode mode, cos/sin are [1, batch, 1, head_dim] and this core's shard
        // may contain multiple batch rows. Push one row at a time and advance the CB.
        sin_cb.reserve_back(Wt);
        cos_cb.reserve_back(Wt);
        sin_cb.push_back(Wt);
        cos_cb.push_back(Wt);

        for (uint32_t ht = 0; ht < heads_per_batch_t; ++ht) {
            rotated_in_interm_cb.reserve_back(Wt);
            sin_interm_cb.reserve_back(Wt);
            cos_interm_cb.reserve_back(Wt);
            out_cb.reserve_back(Wt);

            // Get the input
            in_cb.reserve_back(Wt);
            in_cb.push_back(Wt);
            in_cb.wait_front(Wt);

            // Process second half: multiply by -1 and store in rotated buffer
            mul_tiles_bcast_scalar_init_short(in_cb_id, scalar_cb_id);
            tile_regs_acquire();
            for (uint32_t j = 0; j < half_Wt; ++j) {
                mul_tiles_bcast_scalar(in_cb_id, scalar_cb_id, j + half_Wt, 0, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < half_Wt; ++j) {
                pack_tile(j, rotated_in_interm_cb_id, j);
            }
            tile_regs_release();

            // Copy first half to second half of rotated buffer
            tile_regs_acquire();
            for (uint32_t j = 0; j < half_Wt; ++j) {
                copy_tile_init_with_dt(in_cb_id);
                copy_tile(in_cb_id, j, j + half_Wt);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < half_Wt; ++j) {
                pack_tile(j + half_Wt, rotated_in_interm_cb_id, j + half_Wt);
            }
            tile_regs_release();

            rotated_in_interm_cb.push_back(Wt);
            rotated_in_interm_cb.wait_front(Wt);

            // sin_interim = rotated * sin (broadcast rows)
            mul_bcast_rows_init_short(rotated_in_interm_cb_id, sin_cb_id);
            tile_regs_acquire();
            for (uint32_t j = 0; j < Wt; ++j) {
                mul_tiles_bcast<BroadcastType::ROW>(rotated_in_interm_cb_id, sin_cb_id, j, j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < Wt; ++j) {
                pack_tile(j, sin_interm_cb_id, j);
            }
            tile_regs_release();
            sin_interm_cb.push_back(Wt);
            rotated_in_interm_cb.pop_front(Wt);

            tile_regs_acquire();
            for (uint32_t j = 0; j < Wt; ++j) {
                mul_tiles_bcast<BroadcastType::ROW>(in_cb_id, cos_cb_id, j, j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < Wt; ++j) {
                pack_tile(j, cos_interm_cb_id, j);
            }
            tile_regs_release();
            cos_interm_cb.push_back(Wt);
            in_cb.pop_front(Wt);

            // out = cos_interim + sin_interim
            sin_interm_cb.wait_front(Wt);
            cos_interm_cb.wait_front(Wt);
            add_tiles_init(cos_interm_cb_id, sin_interm_cb_id);
            tile_regs_acquire();
            for (uint32_t j = 0; j < Wt; ++j) {
                add_tiles(cos_interm_cb_id, sin_interm_cb_id, j, j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < Wt; ++j) {
                pack_tile(j, out_cb_id, j);
            }
            tile_regs_release();
            out_cb.push_back(Wt);
            sin_interm_cb.pop_front(Wt);
            cos_interm_cb.pop_front(Wt);
        }

        sin_cb.pop_front(Wt);
        cos_cb.pop_front(Wt);
    }

    // Done with the scalar, so remove from CB
    scalar_cb.pop_front(onetile);
}
