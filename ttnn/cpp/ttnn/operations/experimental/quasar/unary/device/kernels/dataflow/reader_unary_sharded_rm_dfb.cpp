// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar unary reader, ROW_MAJOR sharded input: streams one shard of the input tensor into the "in" DFB as raw
// tile-sized chunks, the Metal 2.0 counterpart of the upstream sharded ROW_MAJOR path (which borrows the shard as
// the CB and calls every tile_size bytes of it a tile). An elementwise op does not care how the values are
// arranged inside a tile, so the shard's bytes go through the compute kernel as is and the writer puts each chunk
// back at the same offset of the output shard.
// The shard is addressed with ShardView(src) + {shard_id, offset_bytes}; chunk c is bytes [c * entry, (c + 1) *
// entry) of the shard. The last chunk may be short (last_chunk_bytes); the rest of its DFB entry is stale and is
// never written back. The DFB is bound with implicit sync disabled, so reserve_back/push_back are the credit
// handshake with the compute kernel.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t shard_id = get_arg(args::shard_id);
    const uint32_t num_chunks = get_arg(args::num_chunks);
    const uint32_t last_chunk_bytes = get_arg(args::last_chunk_bytes);

    Noc noc;
    DataflowBuffer dfb(dfb::in);
    const uint32_t chunk_bytes = dfb.get_entry_size();
    constexpr uint32_t onepage = 1;

    const auto src = TensorAccessor(tensor::src);
    const ShardView src_shard(src);

    for (uint32_t c = 0; c < num_chunks; ++c) {
        const uint32_t bytes = c + 1 == num_chunks ? last_chunk_bytes : chunk_bytes;
        dfb.reserve_back(onepage);
        noc.async_read(src_shard, dfb, bytes, {.shard_id = shard_id, .offset_bytes = c * chunk_bytes}, {});
        noc.async_read_barrier();
        dfb.push_back(onepage);
    }
}
