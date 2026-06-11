// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/circular_buffer.h"

// Metal 2.0 (sharded_to_interleaved private copy): data-format conversion compute kernel, used only when
// the input and output dtypes differ. It copies each tile from the sharded input DFB (dfb::in0) to the
// output DFB (dfb::out), repacking into the output data format. Only-allowed changes from the descriptor
// era: the CB ids come from the DFB binding tokens (dfb::in0 / dfb::out) instead of the hardcoded
// tt::CBIndex::c_0 / c_16, and the per-core tile count comes from the named-arg namespace (args::). The
// copy/pack logic is unchanged.
void kernel_main() {
    const uint32_t per_core_tile_cnt = get_arg(args::per_core_tile_cnt);

    constexpr uint32_t cb_in = dfb::in0;
    constexpr uint32_t cb_out = dfb::out;

    CircularBuffer cb_in0(cb_in);
    CircularBuffer cb_out0(cb_out);

    unary_op_init_common(cb_in, cb_out);
    copy_tile_init(cb_in);
    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        tile_regs_acquire();

        // Pop tile after tile, copy to DST and pack
        cb_in0.wait_front(1);
        cb_out0.reserve_back(1);
        copy_tile(cb_in, 0, 0);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);

        cb_in0.pop_front(1);
        cb_out0.push_back(1);

        tile_regs_release();
    }
}
