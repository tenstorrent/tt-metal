// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/transpose.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t in1_num_blocks = get_arg(args::in1_num_blocks);
    uint32_t in1_num_blocks_h = get_arg(args::in1_num_blocks_h);

    constexpr uint32_t onetile = 1;
    constexpr uint32_t num_rows_in_one_tile = 32;

#ifdef REPEAT_INTERLEAVE_IN1
    compute_kernel_hw_startup(
        dfb::in0_transposed, dfb::in1_bcast_row, dfb::out);  // TODO: Is there a specific one for bcast mul?
#else
    compute_kernel_hw_startup(dfb::in0, dfb::in1, dfb::out);
#endif

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_out(dfb::out);
    DataflowBuffer dfb_in0_transposed(dfb::in0_transposed);
    DataflowBuffer dfb_in1_transposed(dfb::in1_transposed);
    DataflowBuffer dfb_in1_bcast_row(dfb::in1_bcast_row);
    DataflowBuffer dfb_out_transposed(dfb::out_transposed);

    for (uint32_t block_h_id = 0; block_h_id < in1_num_blocks_h; block_h_id++) {
#ifdef REPEAT_IN0
        // Transpose in0
        dfb_in0.wait_front(onetile);
// No need to transpose in0 if in1 is not repeat_interleaved
#ifdef REPEAT_INTERLEAVE_IN1
        tile_regs_acquire();
        tile_regs_wait();

        transpose_init(dfb::in0);
        reconfig_data_format_srca(dfb::out_transposed, dfb::in0);
        pack_reconfig_data_format(dfb::out, dfb::in0_transposed);
        transpose_tile(dfb::in0, 0, 0);

        dfb_in0_transposed.reserve_back(onetile);
        pack_tile(0, dfb::in0_transposed);

        tile_regs_commit();
        tile_regs_release();
        dfb_in0_transposed.push_back(onetile);
        dfb_in0.pop_front(onetile);

        dfb_in0_transposed.wait_front(onetile);
#endif
#endif

        for (uint32_t in1_block = 0; in1_block < in1_num_blocks; in1_block++) {
            // Transpose in1
            dfb_in1.wait_front(onetile);
            tile_regs_acquire();
            tile_regs_wait();

// If input b is not repeat_interleaved, then no need to transpose, bcast row
#ifndef REPEAT_INTERLEAVE_IN1
            mul_init(dfb::in0, dfb::in1);
            reconfig_data_format_srca(dfb::out, dfb::in0);
            pack_reconfig_data_format(dfb::in0_transposed, dfb::out);
            mul_tiles(dfb::in0, dfb::in1, 0, 0, 0);

            dfb_out.reserve_back(onetile);
            pack_tile(0, dfb::out);

            tile_regs_commit();
            tile_regs_release();
            dfb_out.push_back(onetile);
            dfb_in1.pop_front(onetile);
#else
            transpose_init(dfb::in1);
            reconfig_data_format_srca(dfb::in1);
            pack_reconfig_data_format(dfb::in1_transposed);
            transpose_tile(dfb::in1, 0, 0);

            dfb_in1_transposed.reserve_back(onetile);
            pack_tile(0, dfb::in1_transposed);

            tile_regs_commit();
            tile_regs_release();
            dfb_in1_transposed.push_back(onetile);
            dfb_in1.pop_front(onetile);

            // Receive in1 as single rows to bcast mul with in0
            for (uint32_t tile_row_id = 0; tile_row_id < num_rows_in_one_tile; tile_row_id++) {
#ifndef REPEAT_IN0
                // Transpose in0
                dfb_in0.wait_front(onetile);
                tile_regs_acquire();
                tile_regs_wait();

                transpose_init(dfb::in0);
                reconfig_data_format_srca(dfb::in0);
                pack_reconfig_data_format(dfb::in0_transposed);
                transpose_tile(dfb::in0, 0, 0);

                dfb_in0_transposed.reserve_back(onetile);
                pack_tile(0, dfb::in0_transposed);

                tile_regs_commit();
                tile_regs_release();
                dfb_in0_transposed.push_back(onetile);
                dfb_in0.pop_front(onetile);

                dfb_in0_transposed.wait_front(onetile);
#endif

                dfb_in1_bcast_row.wait_front(onetile);
                tile_regs_acquire();
                tile_regs_wait();

                mul_bcast_rows_init(dfb::in0_transposed, dfb::in1_bcast_row);
                reconfig_data_format_srca(dfb::in0_transposed);
                pack_reconfig_data_format(dfb::out_transposed);
                mul_tiles_bcast_rows(dfb::in0_transposed, dfb::in1_bcast_row, 0, 0, 0);

                dfb_out_transposed.reserve_back(onetile);
                pack_tile(0, dfb::out_transposed);

                tile_regs_commit();
                tile_regs_release();
                dfb_out_transposed.push_back(onetile);
#ifndef REPEAT_IN0
                dfb_in0_transposed.pop_front(onetile);
#endif
                dfb_in1_bcast_row.pop_front(onetile);

                // Transpose output back
                dfb_out_transposed.wait_front(onetile);
                tile_regs_acquire();
                tile_regs_wait();

                transpose_init(dfb::out_transposed);
                reconfig_data_format(dfb::in0_transposed, dfb::out_transposed);
                pack_reconfig_data_format(dfb::out_transposed, dfb::out);
                transpose_tile(dfb::out_transposed, 0, 0);

                dfb_out.reserve_back(onetile);
                pack_tile(0, dfb::out);

                tile_regs_commit();
                tile_regs_release();
                dfb_out.push_back(onetile);
                dfb_out_transposed.pop_front(onetile);
            }

            dfb_in1_transposed.pop_front(onetile);
#endif
        }
#ifdef REPEAT_IN0
#ifdef REPEAT_INTERLEAVE_IN1
        dfb_in0_transposed.pop_front(onetile);
#else
        dfb_in0.pop_front(onetile);
#endif
#endif
    }
}
