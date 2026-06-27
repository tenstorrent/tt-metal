// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// matmul writer (BRISC). Drains one output block from cb_out (packed
// SubblockMajor by matmul_block) and scatters each tile to its interleaved
// DRAM position. Out-of-range (ragged/phantom) tiles are skipped.
//
// SubblockMajor read order matches matmul_block's pack order:
//   for sb_m: for sb_n: for r: for c   (r,c row-major within a subblock).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t CB_OUT = 16;

void kernel_main() {
    constexpr uint32_t Mt = get_compile_time_arg_val(0);
    constexpr uint32_t Nt = get_compile_time_arg_val(1);
    constexpr uint32_t batch = get_compile_time_arg_val(2);
    constexpr uint32_t block_M_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t block_N_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(5);
    constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(6);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(7);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(8);
    constexpr uint32_t per_core_M_blocks = get_compile_time_arg_val(9);
    constexpr uint32_t per_core_N_blocks = get_compile_time_arg_val(10);
    constexpr uint32_t tileC_bytes = get_compile_time_arg_val(11);
    constexpr auto c_args = TensorAccessorArgs<12>();

    const uint32_t C_addr = get_arg_val<uint32_t>(0);
    const uint32_t grid_row = get_arg_val<uint32_t>(1);  // Y
    const uint32_t grid_col = get_arg_val<uint32_t>(2);  // X

    constexpr uint32_t out_block_tiles = block_M_tiles * block_N_tiles;

    Noc noc;
    CircularBuffer cb_out(CB_OUT);
    const auto c_acc = TensorAccessor(c_args, C_addr, tileC_bytes);

    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t local_mb = 0; local_mb < per_core_M_blocks; local_mb++) {
            const uint32_t global_mb = grid_row * per_core_M_blocks + local_mb;
            const uint32_t mb_base = global_mb * block_M_tiles;

            for (uint32_t local_nb = 0; local_nb < per_core_N_blocks; local_nb++) {
                const uint32_t global_nb = grid_col * per_core_N_blocks + local_nb;
                const uint32_t nb_base = global_nb * block_N_tiles;

                cb_out.wait_front(out_block_tiles);
                uint32_t t = 0;
                for (uint32_t sb_m = 0; sb_m < in0_num_subblocks; sb_m++) {
                    for (uint32_t sb_n = 0; sb_n < in1_num_subblocks; sb_n++) {
                        for (uint32_t r = 0; r < out_subblock_h; r++) {
                            const uint32_t m_tile = mb_base + sb_m * out_subblock_h + r;
                            for (uint32_t c = 0; c < out_subblock_w; c++) {
                                const uint32_t n_tile = nb_base + sb_n * out_subblock_w + c;
                                if (m_tile < Mt && n_tile < Nt) {
                                    const uint32_t tid = b * Mt * Nt + m_tile * Nt + n_tile;
                                    noc.async_write(
                                        cb_out,
                                        c_acc,
                                        tileC_bytes,
                                        {.offset_bytes = t * tileC_bytes},
                                        {.page_id = tid});
                                }
                                t++;
                            }
                        }
                    }
                }
                noc.async_write_barrier();
                cb_out.pop_front(out_block_tiles);
            }
        }
    }
}
