// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;

    // READER RUNTIME ARGS
    uint32_t num_blocks = get_arg(args::num_blocks);
    uint32_t in0_tensor_tile_id = get_arg(args::in0_tensor_tile_id);
    uint32_t in1_tensor_tile_id = get_arg(args::in1_tensor_tile_id);

    // COMPILE TIME ARGS
    // READER COMPILE TIME ARGS
    constexpr uint32_t q_num_tiles = get_arg(args::q_num_tiles);
    constexpr uint32_t kv_num_tiles = get_arg(args::kv_num_tiles);
    constexpr bool head_parallel = get_arg(args::head_parallel) != 0;
    constexpr uint32_t head_tiles = get_arg(args::head_tiles);
    constexpr uint32_t seq_tiles = get_arg(args::seq_tiles);

    const auto s0 = TensorAccessor(tensor::input_q);

#ifdef READ_FROM_INPUT_TENSOR_KV
    const auto s1 = TensorAccessor(tensor::input_kv);
#endif

    DataflowBuffer dfb_qv(dfb::qv);  // dfb for Q, V heads
#ifdef TRANSPOSE_K_HEADS
    DataflowBuffer dfb_k(dfb::k);  // dfb for K heads (used by compute)
#else
    DataflowBuffer& dfb_k = dfb_qv;  // K heads share the Q, V dfb (directly to writer)
#endif

    const uint32_t tile_bytes_qv = dfb_qv.get_tile_size();
    const uint32_t tile_bytes_k = dfb_k.get_tile_size();

    if constexpr (head_parallel) {
        constexpr uint32_t heads = q_num_tiles / head_tiles;
        constexpr uint32_t transfer_tiles = head_tiles % 4 == 0 ? 4 : (head_tiles % 2 == 0 ? 2 : 1);
        for (uint32_t block = in0_tensor_tile_id; block < in0_tensor_tile_id + num_blocks; ++block) {
            const uint32_t batch_head = block / seq_tiles;
            const uint32_t row = block % seq_tiles;
            uint32_t source =
                ((batch_head / heads) * seq_tiles + row) * q_num_tiles + (batch_head % heads) * head_tiles;
            for (uint32_t tile = 0; tile < head_tiles; tile += transfer_tiles) {
                dfb_qv.reserve_back(transfer_tiles);
                uint32_t destination = dfb_qv.get_write_ptr();
                for (uint32_t j = 0; j < transfer_tiles; ++j) {
                    noc.async_read(s0, CoreLocalMem<uint32_t>(destination), tile_bytes_qv, {.page_id = source++}, {});
                    destination += tile_bytes_qv;
                }
                noc.async_read_barrier();
                dfb_qv.push_back(transfer_tiles);
            }
        }
    } else {
        // One head (head_tiles contiguous tiles of a row block) per DFB transaction: issue all of the
        // head's reads, one barrier, push. The DFBs hold two heads, so the next head's reads overlap the
        // writer draining the previous one (a per-tile barrier made this path NoC-latency bound).
        const auto read_head = [&](const auto& src, uint32_t& tile_id, DataflowBuffer& dfb, uint32_t tile_bytes) {
            dfb.reserve_back(head_tiles);
            uint32_t l1_write_addr = dfb.get_write_ptr();
            for (uint32_t i = 0; i < head_tiles; i++) {
                noc.async_read(src, CoreLocalMem<uint32_t>(l1_write_addr), tile_bytes, {.page_id = tile_id}, {});
                l1_write_addr += tile_bytes;
                tile_id++;
            }
            noc.async_read_barrier();
            dfb.push_back(head_tiles);
        };
        static_assert(q_num_tiles % head_tiles == 0 && kv_num_tiles % head_tiles == 0);
#ifdef READ_FROM_INPUT_TENSOR_KV
#define KV_SRC s1
#define KV_TILE_ID in1_tensor_tile_id
#else
#define KV_SRC s0
#define KV_TILE_ID in0_tensor_tile_id
#endif
        for (uint32_t block = 0; block < num_blocks; block++) {
            // Q
            for (uint32_t h = 0; h < q_num_tiles / head_tiles; h++) {
                read_head(s0, in0_tensor_tile_id, dfb_qv, tile_bytes_qv);
            }
            // K
            for (uint32_t h = 0; h < kv_num_tiles / head_tiles; h++) {
                read_head(KV_SRC, KV_TILE_ID, dfb_k, tile_bytes_k);
            }
#ifdef KV_TIED
            // K and V come from the same columns (one projection, tied), so step back over the K tiles
            // just read instead of walking past them. The V loop below advances by the same count,
            // leaving the running id exactly one block width ahead, which is what the next block
            // expects. Rewind whichever cursor the V loop actually reads from.
            KV_TILE_ID -= kv_num_tiles;
#endif
            // V
            for (uint32_t h = 0; h < kv_num_tiles / head_tiles; h++) {
                read_head(KV_SRC, KV_TILE_ID, dfb_qv, tile_bytes_qv);
            }
        }
#undef KV_SRC
#undef KV_TILE_ID
    }
}
