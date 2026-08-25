// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/kernel/compute/dest_format_helpers.hpp"

void kernel_main() {
    constexpr uint32_t onetile = 1;

    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t Ht = get_arg(args::Ht);  // Total rows (tiles) owned by this core
    constexpr uint32_t heads_per_batch_t = get_arg(args::heads_per_batch_t);
    constexpr uint32_t batch_per_core = get_arg(args::batch_per_core);
    constexpr uint32_t half_Wt = Wt / 2;
    (void)Ht;

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_cos(dfb::cos);
    DataflowBuffer dfb_sin(dfb::sin);
    DataflowBuffer dfb_scalar(dfb::scalar);
    DataflowBuffer dfb_rotated_in_interm(dfb::rotated_in_interm);
    DataflowBuffer dfb_cos_interm(dfb::cos_interm);
    DataflowBuffer dfb_sin_interm(dfb::sin_interm);
    DataflowBuffer dfb_out(dfb::out);

    compute_kernel_hw_startup(dfb::in, dfb::sin, dfb::sin_interm);  // General Init for all binary ops

    // Wait for the reader kernel (reader_rotary_embedding_hf_sharded.cpp) to
    // write -1.0 into the scalar DFB and push it.
    dfb_scalar.wait_front(onetile);

    for (uint32_t batch_idx = 0; batch_idx < batch_per_core; ++batch_idx) {
        // For decode mode, cos/sin are [1, batch, 1, head_dim] and this core's shard
        // may contain multiple batch rows. Push one row at a time and advance the DFB.
        dfb_sin.reserve_back(Wt);
        dfb_cos.reserve_back(Wt);
        dfb_sin.push_back(Wt);
        dfb_cos.push_back(Wt);

        for (uint32_t ht = 0; ht < heads_per_batch_t; ++ht) {
            dfb_rotated_in_interm.reserve_back(Wt);
            dfb_sin_interm.reserve_back(Wt);
            dfb_cos_interm.reserve_back(Wt);
            dfb_out.reserve_back(Wt);

            // Get the input
            dfb_in.reserve_back(Wt);
            dfb_in.push_back(Wt);
            dfb_in.wait_front(Wt);

            // Process second half: multiply by -1 and store in rotated buffer
            mul_bcast_scalar_init(dfb::in, dfb::scalar);
            tile_regs_acquire();
            for (uint32_t j = 0; j < half_Wt; ++j) {
                mul_tiles_bcast_scalar(dfb::in, dfb::scalar, j + half_Wt, 0, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < half_Wt; ++j) {
                pack_tile(j, dfb::rotated_in_interm, j);
            }
            tile_regs_release();

            // Copy first half to second half of rotated buffer
            tile_regs_acquire();
            for (uint32_t j = 0; j < half_Wt; ++j) {
                copy_tile_init_with_dt(dfb::in);
                copy_tile(dfb::in, j, j + half_Wt);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < half_Wt; ++j) {
                pack_tile(j + half_Wt, dfb::rotated_in_interm, j + half_Wt);
            }
            tile_regs_release();

            dfb_rotated_in_interm.push_back(Wt);
            dfb_rotated_in_interm.wait_front(Wt);

            // sin_interim = rotated * sin (broadcast rows)
            mul_bcast_rows_init(dfb::rotated_in_interm, dfb::sin);
            tile_regs_acquire();
            for (uint32_t j = 0; j < Wt; ++j) {
                mul_tiles_bcast<BroadcastType::ROW>(dfb::rotated_in_interm, dfb::sin, j, j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < Wt; ++j) {
                pack_tile(j, dfb::sin_interm, j);
            }
            tile_regs_release();
            dfb_sin_interm.push_back(Wt);
            dfb_rotated_in_interm.pop_front(Wt);

            tile_regs_acquire();
            for (uint32_t j = 0; j < Wt; ++j) {
                mul_tiles_bcast<BroadcastType::ROW>(dfb::in, dfb::cos, j, j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < Wt; ++j) {
                pack_tile(j, dfb::cos_interm, j);
            }
            tile_regs_release();
            dfb_cos_interm.push_back(Wt);
            dfb_in.pop_front(Wt);

            // out = cos_interim + sin_interim
            dfb_sin_interm.wait_front(Wt);
            dfb_cos_interm.wait_front(Wt);
            add_init(dfb::cos_interm, dfb::sin_interm);
            tile_regs_acquire();
            for (uint32_t j = 0; j < Wt; ++j) {
                add_tiles(dfb::cos_interm, dfb::sin_interm, j, j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < Wt; ++j) {
                pack_tile(j, dfb::out, j);
            }
            tile_regs_release();
            dfb_out.push_back(Wt);
            dfb_sin_interm.pop_front(Wt);
            dfb_cos_interm.pop_front(Wt);
        }

        dfb_sin.pop_front(Wt);
        dfb_cos.pop_front(Wt);
    }

    // Done with the scalar, so remove from DFB
    dfb_scalar.pop_front(onetile);
}
