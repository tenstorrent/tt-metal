// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of eltwise_copy.cpp. Copies tiles one at a time from the input DFB to the output DFB
// through DST, which is what performs the data-format conversion when the two DFBs carry different
// formats. Only the plumbing changes: the hardcoded buffer indices 0 and 16 become the dfb::in /
// dfb::out binding tokens (which convert implicitly to the raw buffer id the LLK primitives still
// take), the buffer objects become DataflowBuffers, and the positional compile-time arg becomes a
// named one. The copy loop is untouched.
// Forked rather than converted in place because the legacy file is still bound by factories on the
// legacy positional-arg API.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t per_core_tile_cnt = get_arg(args::per_core_tile_cnt);
    constexpr uint32_t onetile = 1;

    unary_op_init_common(dfb::in, dfb::out);
    copy_tile_init(dfb::in);

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        tile_regs_acquire();

        // Pop tile after tile, copy to DST and pack
        dfb_in.wait_front(onetile);
        dfb_out.reserve_back(onetile);
        copy_tile(dfb::in, 0, 0);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::out);

        dfb_in.pop_front(onetile);
        dfb_out.push_back(onetile);

        tile_regs_release();
    }
}
