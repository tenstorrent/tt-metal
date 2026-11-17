// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
// NOTE: A Metal 2.0 fork of this kernel lives beside it, as
// reader_unary_interleaved_start_id_gen_metal2.cpp. Ops ported to Metal 2.0 bind the fork; this file serves
// the consumers still on the legacy API. Until the last of them migrates and
// this file is retired, changes here likely belong in the fork too.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr uint32_t cb_id_in0 = 0;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;

    const uint32_t page_bytes = get_arg_val<uint32_t>(3);
    // Third argument page_size from runtime args overrides TensorAccessorArgs::AlignedPageSize, which may be stale on
    // program cache hits.
    const auto s = TensorAccessor(src_args, src_addr, page_bytes);

    Noc noc;
    CircularBuffer cb_in0(cb_id_in0);

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_in0.reserve_back(onetile);
        noc.async_read(s, cb_in0, page_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_in0.push_back(onetile);
    }
}
