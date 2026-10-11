// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar unary writer, ROW_MAJOR sharded output: the counterpart of reader_unary_sharded_rm_dfb.cpp. Drains
// num_chunks tile-sized entries from the "out" DFB into one shard of the output tensor, chunk c at byte offset
// c * entry of the shard (ShardView(dst) + {shard_id, offset_bytes}); the last chunk writes only
// last_chunk_bytes. In place (output == input) is safe: chunk c is written only after the compute kernel has
// consumed it, and the reader never goes back to an offset below the one it is reading. The DFB is bound with
// implicit sync disabled, so wait_front/pop_front are the credit handshake with the compute kernel.

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
    DataflowBuffer dfb(dfb::out);
    const uint32_t chunk_bytes = dfb.get_entry_size();
    constexpr uint32_t onepage = 1;

    const auto dst = TensorAccessor(tensor::dst);
    const ShardView dst_shard(dst);

    for (uint32_t c = 0; c < num_chunks; ++c) {
        const uint32_t bytes = c + 1 == num_chunks ? last_chunk_bytes : chunk_bytes;
        dfb.wait_front(onepage);
        noc.async_write(dfb, dst_shard, bytes, {}, {.shard_id = shard_id, .offset_bytes = c * chunk_bytes});
        // The chunk has left the DFB slot once the write is flushed, so the slot can go back to the producer.
        noc.async_writes_flushed();
        dfb.pop_front(onepage);
    }
    noc.async_write_barrier();
}
