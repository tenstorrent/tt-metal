// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port of reader_unary_start_id.cpp (op-private copy). The legacy reader is still consumed
// positionally by the un-migrated untilize variants (and by untilize_with_unpadding via its shared file
// path), so the migrated single-core / default-multi-core interleaved factories carry their own copy
// here. Only the binding mechanism changed: the input CB id comes from the DFB producer token (dfb::),
// the source address from the TensorAccessor binding (ta::), and num_tiles / start_page_id from named
// runtime args (args::). The page-by-page interleaved read loop is preserved.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t start_page_id = get_arg(args::start_page_id);

    constexpr uint32_t cb_id_in0 = dfb::cb_id_in0;

    const uint32_t tile_bytes = get_tile_size(cb_id_in0);

    const auto s = TensorAccessor(ta::src_args);

    Noc noc;
    CircularBuffer cb_in(cb_id_in0);

    uint32_t end_page_id = start_page_id + num_tiles;
    for (uint32_t page_id = start_page_id; page_id < end_page_id; ++page_id) {
        cb_in.reserve_back(1);
        noc.async_read(s, cb_in, tile_bytes, {.page_id = page_id, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_in.push_back(1);
    }
}
