// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc;

    uint32_t in_tile_offset_by_head = get_arg(args::in_tile_offset_by_head);

    constexpr uint32_t ELEMENT_SIZE = get_arg(args::element_size);
    constexpr uint32_t SUBTILE_LINE_BYTES = get_arg(args::subtile_line_bytes);
    constexpr uint32_t head_size = get_arg(args::head_size);
    constexpr uint32_t batch = get_arg(args::batch);
    constexpr uint32_t head_size_num_tiles = get_arg(args::head_size_num_tiles);
    constexpr uint32_t PHASES_TO_READ =
        get_arg(args::phases_to_read);  // 0 to read all phases, 1 to read only first phase, 2 to read only second phase

    constexpr uint32_t in_num_cores = get_arg(args::in_num_cores);
    constexpr uint32_t face_h = get_arg(args::face_h);
    constexpr uint32_t face_hw = get_arg(args::face_hw);

    // The input is height sharded with one user per shard, one shard per core of its grid: the
    // reads walk the shards in order, at an offset inside the current shard.
    const auto in_tensor = TensorAccessor(tensor::input);
    const ShardView in_shards(in_tensor);

    DataflowBuffer dfb_q_out(dfb::q_out);

    // Q
    uint32_t shard_id = 0;
    uint32_t total_input_cores = in_num_cores;
    uint32_t num_tiles_per_core = (head_size_num_tiles * batch) / total_input_cores;
    // Each shard's NoC address is looked up once, when the reads move onto it.
    const uint8_t noc_id = noc.get_noc_id();
    auto in_shard = in_shards.shard(shard_id, noc_id);

    uint32_t qkv_read_offset = in_tile_offset_by_head;
    uint32_t num_tiles_read_cur_core = 0;
    uint32_t q_write_addr = 0;
    uint32_t tile_size = head_size / head_size_num_tiles;
    const uint32_t dfb_write_ptr_base = dfb_q_out.get_write_ptr();

    for (uint32_t q = 0; q < batch; ++q) {
        uint32_t wptr_offset = q < face_h ? q * SUBTILE_LINE_BYTES : (q + face_h) * SUBTILE_LINE_BYTES;
        uint32_t q_write_addr = dfb_write_ptr_base + wptr_offset;
        for (uint32_t i = 0; i < head_size_num_tiles; ++i) {
            // Read first phase
            if constexpr (PHASES_TO_READ == 0 || PHASES_TO_READ == 1) {
                noc.async_read(
                    in_shard,
                    CoreLocalMem<uint32_t>(q_write_addr),
                    SUBTILE_LINE_BYTES,
                    {.offset_bytes = qkv_read_offset},
                    {});
            }
            // Read second phase
            if constexpr (PHASES_TO_READ == 0 || PHASES_TO_READ == 2) {
                noc.async_read(
                    in_shard,
                    CoreLocalMem<uint32_t>(q_write_addr + face_hw * ELEMENT_SIZE),
                    SUBTILE_LINE_BYTES,
                    {.offset_bytes = qkv_read_offset + face_hw * ELEMENT_SIZE},
                    {});
            }

            qkv_read_offset += tile_size;
            q_write_addr += tile_size;
            num_tiles_read_cur_core++;

            if (num_tiles_read_cur_core == num_tiles_per_core) {
                shard_id++;
                if (shard_id < total_input_cores) {
                    in_shard = in_shards.shard(shard_id, noc_id);
                }
                qkv_read_offset = in_tile_offset_by_head;
                num_tiles_read_cur_core = 0;
            }
        }
    }

    noc.async_read_barrier();
}
