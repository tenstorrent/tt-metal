// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// add_ writer (block operation `release_result_block`, regime `sharded_resident`).
//
// cb_result_shard is BORROWED from input_tensor_a's shard: the compute kernel packs each result tile over
// the operand tile it has just unpacked, so the result is already in the caller's buffer. This kernel moves
// no data (no NoC transfer, no TensorAccessor, no transaction tag); it acknowledges the packed tiles so the
// ring balances at kernel exit.
//
// Raw DFB credit calls, no helper: see the reader's header (no kernel-library helper acknowledges a
// resident ring).
//
// Thread plan: hart w of W serves NEOs w, w+W, ...; NEO t's share is the same round-robin deal the reader
// and the compute kernel use. The ring is listed in this kernel's disable_dfb_implicit_sync_for.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t core_tiles = get_arg(args::core_tiles);
    constexpr uint32_t W = get_arg(args::writer_threads);
    constexpr uint32_t C = get_arg(args::compute_threads);

    DataflowBuffer dfb_result(dfb::cb_result_shard);

    for (uint32_t t = get_my_thread_id(); t < C; t += W) {
        const uint32_t share = core_tiles / C + (t < core_tiles % C ? 1u : 0u);
        if (share == 0) {
            break;
        }
        dfb_result.wait_front(share);
        dfb_result.pop_front(share);
    }
}
