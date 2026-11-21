// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rotary_embedding_llama reader: publish_shards.
//
// Every input ring is borrowed from its tensor's resident L1 shard (zero-copy), so there is nothing to read:
// this kernel only posts the credits of tiles that are already there. No NoC transfer is issued, so no
// transaction tag is set and none needs handing back. The host opts every ring out of implicit sync.
//
// Hart p of R publishes the core's tiles p, p+R, ... of x, cos and sin; on the STRIDED rings its k-th push
// lands with NEO (p + k*R) mod C, which is the NEO that computes that tile (the same dealing on all three
// rings, so a NEO's k-th x, cos and sin tiles are the same head_dim tile j). A DM STRIDED side's share is one
// entry, so one reserve/push per entry. trans_mat (one tile, ALL ring) is pushed once: every NEO sees it.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // The plan's count, not get_num_threads(): were a thread not to run, its tiles would be missing.
    constexpr uint32_t R = get_arg(args::reader_threads);
    constexpr uint32_t users_per_core = get_arg(args::users_per_core);
    constexpr uint32_t tensor_dim_tiles = get_arg(args::tensor_dim_tiles);
    constexpr uint32_t tiles_per_core = users_per_core * tensor_dim_tiles;
    const uint32_t p = get_my_thread_id();

    DataflowBuffer input_tiles(dfb::input_tiles);
    DataflowBuffer cos_tiles(dfb::cos_tiles);
    DataflowBuffer sin_tiles(dfb::sin_tiles);
    DataflowBuffer trans_mat(dfb::trans_mat);

    if (p == 0) {
        trans_mat.reserve_back(1);
        trans_mat.push_back(1);
    }
    for (uint32_t i = p; i < tiles_per_core; i += R) {
        input_tiles.reserve_back(1);
        input_tiles.push_back(1);
        cos_tiles.reserve_back(1);
        cos_tiles.push_back(1);
        sin_tiles.reserve_back(1);
        sin_tiles.push_back(1);
    }
}
