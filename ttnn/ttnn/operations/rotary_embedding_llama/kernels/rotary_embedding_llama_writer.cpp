// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rotary_embedding_llama writer: release_output.
//
// The output ring is borrowed from the output tensor's L1 shard: the packer has already written every tile
// where it belongs, so this kernel only drains the credits (no NoC transfer, no transaction tag). Hart w of W
// drains the core's tiles w, w+W, ...; its k-th pop comes from NEO (w + k*W) mod C. One wait/pop per entry
// (a DM STRIDED side's share is one entry).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t W = get_arg(args::writer_threads);
    constexpr uint32_t users_per_core = get_arg(args::users_per_core);
    constexpr uint32_t tensor_dim_tiles = get_arg(args::tensor_dim_tiles);
    constexpr uint32_t tiles_per_core = users_per_core * tensor_dim_tiles;

    DataflowBuffer output_tiles(dfb::output_tiles);

    for (uint32_t i = get_my_thread_id(); i < tiles_per_core; i += W) {
        output_tiles.wait_front(1);
        output_tiles.pop_front(1);
    }
}
