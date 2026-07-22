// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/transpose.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    uint32_t in1_num_blocks = get_arg_val<uint32_t>(0);
    uint32_t in1_num_blocks_h = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(2);
    constexpr uint32_t cb_in0_transposed = get_compile_time_arg_val(3);
    constexpr uint32_t cb_in1_transposed = get_compile_time_arg_val(4);
    constexpr uint32_t cb_in1_bcast_row = get_compile_time_arg_val(5);
    constexpr uint32_t cb_out_transposed = get_compile_time_arg_val(6);

    constexpr uint32_t onetile = 1;
    constexpr uint32_t num_rows_in_one_tile = 32;

#ifdef REPEAT_INTERLEAVE_IN1
    binary_op_init_common(
        cb_in0_transposed, cb_in1_bcast_row, cb_id_out);  // TODO: Is there a specific one for bcast mul?
#else
    binary_op_init_common(cb_id_in0, cb_id_in1, cb_id_out);
#endif

    CircularBuffer cb_in0(cb_id_in0);
    CircularBuffer cb_in1(cb_id_in1);
    CircularBuffer cb_out(cb_id_out);
    CircularBuffer cb_in0_transposed_buf(cb_in0_transposed);
    CircularBuffer cb_in1_transposed_buf(cb_in1_transposed);
    CircularBuffer cb_in1_bcast_row_buf(cb_in1_bcast_row);
    CircularBuffer cb_out_transposed_buf(cb_out_transposed);

    for (uint32_t block_h_id = 0; block_h_id < in1_num_blocks_h; block_h_id++) {
#ifdef REPEAT_IN0
        // Transpose in0
        cb_in0.wait_front(onetile);
// No need to transpose in0 if in1 is not repeat_interleaved
#ifdef REPEAT_INTERLEAVE_IN1
        tile_regs_acquire();
        tile_regs_wait();

        transpose_init(cb_id_in0);
        reconfig_data_format_srca(cb_out_transposed, cb_id_in0);
        pack_reconfig_data_format(cb_id_out, cb_in0_transposed);
        transpose_tile(cb_id_in0, 0, 0);

        cb_in0_transposed_buf.reserve_back(onetile);
        pack_tile(0, cb_in0_transposed);

        tile_regs_commit();
        tile_regs_release();
        cb_in0_transposed_buf.push_back(onetile);
        cb_in0.pop_front(onetile);

        cb_in0_transposed_buf.wait_front(onetile);
#endif
#endif

        for (uint32_t in1_block = 0; in1_block < in1_num_blocks; in1_block++) {
            // Transpose in1
            cb_in1.wait_front(onetile);
            tile_regs_acquire();
            tile_regs_wait();

// If input b is not repeat_interleaved, then no need to transpose, bcast row
#ifndef REPEAT_INTERLEAVE_IN1
            mul_init(cb_id_in0, cb_id_in1);
            reconfig_data_format_srca(cb_id_out, cb_id_in0);
            pack_reconfig_data_format(cb_in0_transposed, cb_id_out);
            mul_tiles(cb_id_in0, cb_id_in1, 0, 0, 0);

            cb_out.reserve_back(onetile);
            pack_tile(0, cb_id_out);

            tile_regs_commit();
            tile_regs_release();
            cb_out.push_back(onetile);
            cb_in1.pop_front(onetile);
#else
            transpose_init(cb_id_in1);
            reconfig_data_format_srca(cb_id_in1);
            pack_reconfig_data_format(cb_in1_transposed);
            transpose_tile(cb_id_in1, 0, 0);

            cb_in1_transposed_buf.reserve_back(onetile);
            pack_tile(0, cb_in1_transposed);

            tile_regs_commit();
            tile_regs_release();
            cb_in1_transposed_buf.push_back(onetile);
            cb_in1.pop_front(onetile);

            // Receive in1 as single rows to bcast mul with in0
            for (uint32_t tile_row_id = 0; tile_row_id < num_rows_in_one_tile; tile_row_id++) {
#ifndef REPEAT_IN0
                // Transpose in0
                cb_in0.wait_front(onetile);
                tile_regs_acquire();
                tile_regs_wait();

                transpose_init(cb_id_in0);
                reconfig_data_format_srca(cb_id_in0);
                pack_reconfig_data_format(cb_in0_transposed);
                transpose_tile(cb_id_in0, 0, 0);

                cb_in0_transposed_buf.reserve_back(onetile);
                pack_tile(0, cb_in0_transposed);

                tile_regs_commit();
                tile_regs_release();
                cb_in0_transposed_buf.push_back(onetile);
                cb_in0.pop_front(onetile);

                cb_in0_transposed_buf.wait_front(onetile);
#endif

                cb_in1_bcast_row_buf.wait_front(onetile);
                tile_regs_acquire();
                tile_regs_wait();

                mul_bcast_rows_init_short(cb_in0_transposed, cb_in1_bcast_row);
                reconfig_data_format_srca(cb_in0_transposed);
                pack_reconfig_data_format(cb_out_transposed);
                mul_tiles_bcast_rows(cb_in0_transposed, cb_in1_bcast_row, 0, 0, 0);

                cb_out_transposed_buf.reserve_back(onetile);
                pack_tile(0, cb_out_transposed);

                tile_regs_commit();
                tile_regs_release();
                cb_out_transposed_buf.push_back(onetile);
#ifndef REPEAT_IN0
                cb_in0_transposed_buf.pop_front(onetile);
#endif
                cb_in1_bcast_row_buf.pop_front(onetile);

                // Transpose output back
                cb_out_transposed_buf.wait_front(onetile);
                tile_regs_acquire();
                tile_regs_wait();

                transpose_init(cb_out_transposed);
                reconfig_data_format(cb_in0_transposed, cb_out_transposed);
                pack_reconfig_data_format(cb_out_transposed, cb_id_out);
                transpose_tile(cb_out_transposed, 0, 0);

                cb_out.reserve_back(onetile);
                pack_tile(0, cb_id_out);

                tile_regs_commit();
                tile_regs_release();
                cb_out.push_back(onetile);
                cb_out_transposed_buf.pop_front(onetile);
            }

            cb_in1_transposed_buf.pop_front(onetile);
#endif
        }
#ifdef REPEAT_IN0
#ifdef REPEAT_INTERLEAVE_IN1
        cb_in0_transposed_buf.pop_front(onetile);
#else
        cb_in0.pop_front(onetile);
#endif
#endif
    }
}
