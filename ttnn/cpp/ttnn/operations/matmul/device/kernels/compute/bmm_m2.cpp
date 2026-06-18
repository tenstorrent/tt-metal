// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of bmm.cpp. Forked (not edited in place) because bmm.cpp is also
// used by moreh_matmul; only the MatmulMultiCore factory's compute kernel moves to
// the Metal 2.0 named-binding form. Logic is identical to bmm.cpp.

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "experimental/kernel_args.h"

using std::uint32_t;

// matmul C=A*B using dims MK*KN = MN (row major order)
//
void kernel_main() {
    constexpr int onetile = 1;

    constexpr uint32_t batch = get_arg(args::batch);
    constexpr uint32_t Mt = get_arg(args::Mt);
    constexpr uint32_t Kt = get_arg(args::Kt);
    constexpr uint32_t Nt = get_arg(args::Nt);

    constexpr uint32_t cb_in0 = dfb::in0;
    constexpr uint32_t cb_in1 = dfb::in1;
    constexpr uint32_t cb_out = dfb::out;

    DataflowBuffer in0_cb(dfb::in0);
    DataflowBuffer in1_cb(dfb::in1);
    DataflowBuffer out_cb(dfb::out);

    mm_init(cb_in0, cb_in1, cb_out);

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

                    matmul_tiles(cb_in0, cb_in1, 0, 0, 0);

                    in0_cb.pop_front(onetile);
                    in1_cb.pop_front(onetile);
                }

                tile_regs_commit();

                out_cb.reserve_back(onetile);

                tile_regs_wait();
                pack_tile(0, cb_out);
                tile_regs_release();

                out_cb.push_back(onetile);
            }
        }
    }
}
