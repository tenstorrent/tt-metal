// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

using std::uint32_t;

// matmul C=A*B using dims MK*KN = MN (row major order)
//
void kernel_main() {
    constexpr int onetile = 1;

    int dst_tile_index = 0;
    int in0_block_tile_index = 0;

    uint32_t batch = get_arg(args::batch);
    uint32_t Mt = get_arg(args::Mt);
    uint32_t Kt = get_arg(args::Kt);
    uint32_t Nt = get_arg(args::Nt);

    DataflowBuffer in0_cb(dfb::in0);
    DataflowBuffer in1_cb(dfb::in1);
    DataflowBuffer out_cb(dfb::out);

    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::in0, dfb::in1, dfb::out);
    matmul_init(dfb::in0, dfb::in1);

    // the simplest possible version of outer product blocked matmul
    // the reader is expected to read the A's and B's tile rows and tile columns for each output tile
    for (uint32_t nb = 0; nb < batch; nb++) {
        for (uint32_t mt_C = 0; mt_C < Mt; ++mt_C) {    // output tile of C
            for (uint32_t nt_C = 0; nt_C < Nt; ++nt_C)  // output tile index of C
            {
                tile_regs_acquire();
                for (uint32_t kt = 0; kt < Kt; kt++) {
                    in0_cb.wait_front(onetile);
                    in1_cb.wait_front(onetile);

                    matmul_tiles(dfb::in0, dfb::in1, 0, 0, 0);

                    in0_cb.pop_front(onetile);
                    in1_cb.pop_front(onetile);
                }

                tile_regs_commit();

                out_cb.reserve_back(onetile);

                tile_regs_wait();
                pack_tile(0, dfb::out);
                tile_regs_release();

                out_cb.push_back(onetile);
            }
        }
    }
}
