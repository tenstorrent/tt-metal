// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"

using std::uint32_t;

// matmul C=A*B using dims MK*KN = MN (row major order)
//
void kernel_main() {
    constexpr int onetile = 1;

    int dst_tile_index = 0;
    int in0_block_tile_index = 0;

    uint32_t batch = get_compile_time_arg_val(0);
    uint32_t Mt = get_compile_time_arg_val(1);
    uint32_t Kt = get_compile_time_arg_val(2);
    uint32_t Nt = get_compile_time_arg_val(3);

    constexpr uint32_t dfb_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t dfb_in1 = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t dfb_out = get_named_compile_time_arg_val("cb_out");

    DataflowBuffer in0_dfb(dfb_in0);
    DataflowBuffer in1_dfb(dfb_in1);
    DataflowBuffer out_dfb(dfb_out);

    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb_in0, dfb_in1, dfb_out);
    matmul_init(dfb_in0, dfb_in1);

    // the simplest possible version of outer product blocked matmul
    // the reader is expected to read the A's and B's tile rows and tile columns for each output tile
    for (uint32_t nb = 0; nb < batch; nb++) {
        for (uint32_t mt_C = 0; mt_C < Mt; ++mt_C) {    // output tile of C
            for (uint32_t nt_C = 0; nt_C < Nt; ++nt_C)  // output tile index of C
            {
                tile_regs_acquire();
                for (uint32_t kt = 0; kt < Kt; kt++) {
                    in0_dfb.wait_front(onetile);
                    in1_dfb.wait_front(onetile);

                    matmul_tiles(dfb_in0, dfb_in1, 0, 0, 0);

                    in0_dfb.pop_front(onetile);
                    in1_dfb.pop_front(onetile);
                }

                tile_regs_commit();

                out_dfb.reserve_back(onetile);

                tile_regs_wait();
                pack_tile(0, dfb_out);
                tile_regs_release();

                out_dfb.push_back(onetile);
            }
        }
    }
}
