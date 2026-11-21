// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "internal/risc_attribs.h"
#include "api/tile_constants.h"

using namespace tt::constants;
void kernel_main() {
    Noc noc;

    uint32_t index_in_cores = get_arg(args::index_in_cores);

    constexpr uint32_t ELEMENT_SIZE = get_arg(args::ELEMENT_SIZE);
    constexpr uint32_t SUBTILE_LINE_BYTES = get_arg(args::SUBTILE_LINE_BYTES);
    constexpr uint32_t head_size = get_arg(args::head_size);
    constexpr uint32_t num_q_heads = get_arg(args::num_q_heads);
    constexpr uint32_t num_kv_heads = get_arg(args::num_kv_heads);
    constexpr uint32_t head_size_num_tiles = get_arg(args::head_size_num_tiles);
    constexpr uint32_t PHASES_TO_READ =
        get_arg(args::PHASES_TO_READ);  // 0 to read all phases, 1 to read only first phase, 2 to read only second phase
    constexpr uint32_t in_num_cores = get_arg(args::in_num_cores);
    constexpr uint32_t index_stick_size = get_arg(args::index_stick_size);
    // PROCESS_QV / PROCESS_K (which head sections this instance handles) and USE_BATCH_OFFSET are
    // compile defines from the host: they gate resource bindings, so the gated names below exist
    // only in builds where the host actually binds them.

    // The input is width sharded, one shard per core of its grid: the reads walk the shards in order
    // (Q heads, then K, then V), at an offset inside the current shard.
    const auto in_tensor = TensorAccessor(tensor::qkv_in);
    const ShardView qkv_shards(in_tensor);

    uint32_t device_batch_offset = 0;

#ifdef USE_BATCH_OFFSET
    {
        const auto addrg = TensorAccessor(tensor::batch_offset_tensor);
        DataflowBuffer dfb_batch_offset(dfb::batch_offset);
        dfb_batch_offset.reserve_back(1);
        uint32_t index_wr_ptr = dfb_batch_offset.get_write_ptr();
        // Read the batch offset 1 page to read
        noc.async_read(addrg, CoreLocalMem<uint32_t>(index_wr_ptr), index_stick_size, {.page_id = 0}, {});
        noc.async_read_barrier();
        dfb_batch_offset.push_back(1);
        volatile tt_l1_ptr uint32_t* index_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_wr_ptr);
        // Always pick 1st value in tensor as batch offset
        device_batch_offset = index_ptr[0];
    }
#endif
    device_batch_offset += index_in_cores;
    uint32_t in_tile_offset_by_batch = device_batch_offset < 16
                                           ? device_batch_offset * SUBTILE_LINE_BYTES
                                           : (device_batch_offset - 16) * SUBTILE_LINE_BYTES + 512 * ELEMENT_SIZE;

#ifdef PROCESS_QV
    DataflowBuffer dfb_q_out(dfb::q_out);
    DataflowBuffer dfb_v_out(dfb::v_out);
#endif
#ifdef PROCESS_K
    DataflowBuffer dfb_k_out(dfb::k_out);
#endif

    // Q
    uint32_t shard_id = 0;
    uint32_t num_tiles_per_core = head_size_num_tiles * (num_q_heads + 2 * num_kv_heads) / in_num_cores;
    uint32_t num_q_cores = (num_q_heads * head_size_num_tiles) / num_tiles_per_core;
    uint32_t num_kv_cores = (num_kv_heads * head_size_num_tiles) / num_tiles_per_core;
    // Each shard's NoC address is looked up once, when the reads move onto it.
    const uint8_t noc_id = noc.get_noc_id();
    auto qkv_shard = qkv_shards.shard(shard_id, noc_id);

    uint32_t qkv_read_offset = in_tile_offset_by_batch;
    uint32_t num_tiles_read_cur_core = 0;
    uint32_t q_write_addr = 0;
    constexpr uint32_t tile_size = head_size / head_size_num_tiles;
    constexpr uint32_t HALF_TILE_ELEMENTS = FACE_HEIGHT * TILE_WIDTH;
    constexpr uint32_t SUBTILE_ROWS = FACE_HEIGHT;

    // Skip Q section if PROCESS_QV is False
#ifdef PROCESS_QV
    for (uint32_t q = 0; q < num_q_heads; ++q) {
        uint32_t tile_row_index = q / TILE_HEIGHT;
        uint32_t row_in_tile = q % TILE_HEIGHT;
        uint32_t offset_in_tile = row_in_tile < SUBTILE_ROWS ? row_in_tile * SUBTILE_LINE_BYTES
                                                             : (row_in_tile - SUBTILE_ROWS) * SUBTILE_LINE_BYTES +
                                                                   HALF_TILE_ELEMENTS * ELEMENT_SIZE;
        uint32_t wptr_offset = tile_row_index * head_size + offset_in_tile;
        uint32_t q_write_addr = dfb_q_out.get_write_ptr() + wptr_offset;
        for (uint32_t i = 0; i < head_size_num_tiles; ++i) {
            // Read first phase
            if constexpr (PHASES_TO_READ == 0 || PHASES_TO_READ == 1) {
                noc.async_read(
                    qkv_shard,
                    CoreLocalMem<uint32_t>(q_write_addr),
                    SUBTILE_LINE_BYTES,
                    {.offset_bytes = qkv_read_offset},
                    {});
            }
            // Read second phase
            if constexpr (PHASES_TO_READ == 0 || PHASES_TO_READ == 2) {
                noc.async_read(
                    qkv_shard,
                    CoreLocalMem<uint32_t>(q_write_addr + FACE_HW * ELEMENT_SIZE),
                    SUBTILE_LINE_BYTES,
                    {.offset_bytes = qkv_read_offset + FACE_HW * ELEMENT_SIZE},
                    {});
            }

            qkv_read_offset += tile_size;
            q_write_addr += tile_size;
            num_tiles_read_cur_core++;

            if (num_tiles_read_cur_core == num_tiles_per_core) {
                shard_id++;
                if (shard_id < in_num_cores) {
                    qkv_shard = qkv_shards.shard(shard_id, noc_id);
                }
                qkv_read_offset = in_tile_offset_by_batch;
                num_tiles_read_cur_core = 0;
            }
        }
    }
#else
    shard_id += num_q_cores;
    qkv_read_offset = in_tile_offset_by_batch;
    qkv_shard = qkv_shards.shard(shard_id, noc_id);
#endif

#ifdef PROCESS_K
    {
        // K
        uint32_t k_write_addr = 0;

        // Read 2 phases per tile, where there are num_q_heads * q_num_tiles tiles
        for (uint32_t k = 0; k < num_kv_heads; ++k) {
            uint32_t tile_row_index = k / TILE_HEIGHT;
            uint32_t row_in_tile = k % TILE_HEIGHT;
            uint32_t offset_in_tile = row_in_tile < SUBTILE_ROWS ? row_in_tile * SUBTILE_LINE_BYTES
                                                                 : (row_in_tile - SUBTILE_ROWS) * SUBTILE_LINE_BYTES +
                                                                       HALF_TILE_ELEMENTS * ELEMENT_SIZE;
            uint32_t wptr_offset = tile_row_index * head_size + offset_in_tile;
            uint32_t k_write_addr = dfb_k_out.get_write_ptr() + wptr_offset;
            for (uint32_t i = 0; i < head_size_num_tiles; ++i) {
                // Read first phase
                if constexpr (PHASES_TO_READ == 0 || PHASES_TO_READ == 1) {
                    noc.async_read(
                        qkv_shard,
                        CoreLocalMem<uint32_t>(k_write_addr),
                        SUBTILE_LINE_BYTES,
                        {.offset_bytes = qkv_read_offset},
                        {});
                }
                // Read second phase
                if constexpr (PHASES_TO_READ == 0 || PHASES_TO_READ == 2) {
                    noc.async_read(
                        qkv_shard,
                        CoreLocalMem<uint32_t>(k_write_addr + FACE_HW * ELEMENT_SIZE),
                        SUBTILE_LINE_BYTES,
                        {.offset_bytes = qkv_read_offset + FACE_HW * ELEMENT_SIZE},
                        {});
                }

                qkv_read_offset += tile_size;
                k_write_addr += tile_size;
                num_tiles_read_cur_core++;

                if (num_tiles_read_cur_core == num_tiles_per_core) {
                    shard_id++;
                    if (shard_id < in_num_cores) {
                        qkv_shard = qkv_shards.shard(shard_id, noc_id);
                    }
                    qkv_read_offset = in_tile_offset_by_batch;
                    num_tiles_read_cur_core = 0;
                }
            }
        }
    }
#else
    shard_id += num_kv_cores;
    qkv_read_offset = in_tile_offset_by_batch;
    qkv_shard = qkv_shards.shard(shard_id, noc_id);
#endif

#ifdef PROCESS_QV
    {
        // v
        uint32_t v_write_addr = 0;

        // Read 2 phases per tile, where there are num_q_heads * q_num_tiles tiles
        for (uint32_t v = 0; v < num_kv_heads; ++v) {
            uint32_t tile_row_index = v / TILE_HEIGHT;
            uint32_t row_in_tile = v % TILE_HEIGHT;
            uint32_t offset_in_tile = row_in_tile < SUBTILE_ROWS ? row_in_tile * SUBTILE_LINE_BYTES
                                                                 : (row_in_tile - SUBTILE_ROWS) * SUBTILE_LINE_BYTES +
                                                                       HALF_TILE_ELEMENTS * ELEMENT_SIZE;
            uint32_t wptr_offset = tile_row_index * head_size + offset_in_tile;
            uint32_t v_write_addr = dfb_v_out.get_write_ptr() + wptr_offset;
            for (uint32_t i = 0; i < head_size_num_tiles; ++i) {
                // Read first phase
                if constexpr (PHASES_TO_READ == 0 || PHASES_TO_READ == 1) {
                    noc.async_read(
                        qkv_shard,
                        CoreLocalMem<uint32_t>(v_write_addr),
                        SUBTILE_LINE_BYTES,
                        {.offset_bytes = qkv_read_offset},
                        {});
                }
                // Read second phase
                if constexpr (PHASES_TO_READ == 0 || PHASES_TO_READ == 2) {
                    noc.async_read(
                        qkv_shard,
                        CoreLocalMem<uint32_t>(v_write_addr + FACE_HW * ELEMENT_SIZE),
                        SUBTILE_LINE_BYTES,
                        {.offset_bytes = qkv_read_offset + FACE_HW * ELEMENT_SIZE},
                        {});
                }

                qkv_read_offset += tile_size;
                v_write_addr += tile_size;
                num_tiles_read_cur_core++;

                if (num_tiles_read_cur_core == num_tiles_per_core) {
                    shard_id++;
                    if (shard_id < in_num_cores) {
                        qkv_shard = qkv_shards.shard(shard_id, noc_id);
                    }
                    qkv_read_offset = in_tile_offset_by_batch;
                    num_tiles_read_cur_core = 0;
                }
            }
        }
    }
#endif

    noc.async_read_barrier();
}
