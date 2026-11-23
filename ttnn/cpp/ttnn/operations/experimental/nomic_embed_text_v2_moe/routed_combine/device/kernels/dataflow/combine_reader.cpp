// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RISCV_0 of every core: first the bias, once; then for each unit (a 32-token block times a chunk of
// output columns), each slot's 32 expert-row segments and the slots' weight tiles, in batches under
// rotating transaction ids (read_batches.hpp). A gather (combine 0) reads the row segments alone.
//
// Splitting a unit's reads with the writer RISC on NOC_1 was slower at 4096 tokens: the gather's reads
// and writes each move about 13 MB, about 0.65 TB/s on each NoC, and the writer's writes already load
// NOC_1.
//
// With bias_multicast, one core reads the bias and multicasts it into every core's bias buffer, which
// sits at the same address on each. 110 cores reading the same 24 pages queue on the banks that hold
// them (12 us on average, 20 on the slowest core, at 4096 tokens); from L1 they also stall the row
// reads of every core.

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/semaphore.h"
#include "api/tensor/noc_traits.h"
#include "read_batches.hpp"
#include "ttnn/operations/experimental/nomic_embed_text_v2_moe/common/device/kernels/dataflow/page_multicast.hpp"

void kernel_main() {
    constexpr bool combine = get_named_compile_time_arg_val("combine") == 1;
    constexpr uint32_t top_k = get_named_compile_time_arg_val("top_k");
    constexpr uint32_t hidden_tiles = get_named_compile_time_arg_val("hidden_tiles");
    constexpr uint32_t chunk_tiles = get_named_compile_time_arg_val("chunk_tiles");
    constexpr uint32_t token_blocks = get_named_compile_time_arg_val("token_blocks");
    constexpr uint32_t rows_cb_id = get_named_compile_time_arg_val("rows_cb");
    constexpr uint32_t weights_cb_id = get_named_compile_time_arg_val("weights_cb");
    constexpr uint32_t bias_cb_id = get_named_compile_time_arg_val("bias_cb");
    constexpr uint32_t index_cb_id = get_named_compile_time_arg_val("index_cb");
    constexpr bool bias_multicast = get_named_compile_time_arg_val("bias_multicast") == 1;
    constexpr uint32_t slots = get_named_compile_time_arg_val("slots");
    constexpr uint32_t batch_reads = get_named_compile_time_arg_val("batch_reads");
    constexpr uint32_t batch_depth = get_named_compile_time_arg_val("batch_depth");
    constexpr auto rows_args = TensorAccessorArgs<0>();
    constexpr auto index_args = TensorAccessorArgs<rows_args.next_compile_time_args_offset()>();
    constexpr auto weights_args = TensorAccessorArgs<index_args.next_compile_time_args_offset()>();
    constexpr auto bias_args = TensorAccessorArgs<weights_args.next_compile_time_args_offset()>();

    const uint32_t rows_address = get_arg_val<uint32_t>(0);
    const uint32_t index_address = get_arg_val<uint32_t>(1);
    const uint32_t weights_address = get_arg_val<uint32_t>(2);
    const uint32_t bias_address = get_arg_val<uint32_t>(3);
    const uint32_t first_unit = get_arg_val<uint32_t>(4);
    const uint32_t num_units = get_arg_val<uint32_t>(5);
    // bias_multicast: whether this core sends the bias, then the receivers' rectangles (page_multicast.hpp).
    const bool bias_sender = get_arg_val<uint32_t>(6) == 1;
    constexpr uint32_t rectangle_args = 7;

    constexpr uint32_t block_tokens = 32;
    constexpr uint32_t chunks = hidden_tiles / chunk_tiles;
    // One bf16 row segment per tile column: 32 values.
    constexpr uint32_t segment_bytes = chunk_tiles * block_tokens * sizeof(uint16_t);
    constexpr uint32_t index_block_bytes = block_tokens * sizeof(uint32_t);

    const Noc noc;
    DataflowBuffer rows_cb(rows_cb_id);
    DataflowBuffer weights_cb(weights_cb_id);
    DataflowBuffer bias_cb(bias_cb_id);
    DataflowBuffer index_cb(index_cb_id);
    const auto rows_accessor = TensorAccessor(rows_args, rows_address);
    const auto index_accessor = TensorAccessor(index_args, index_address);
    const auto weights_accessor = TensorAccessor(weights_args, weights_address);
    const auto bias_accessor = TensorAccessor(bias_args, bias_address);
    constexpr uint32_t bias_tile_bytes = get_tile_size(bias_cb_id);
    constexpr uint32_t weight_tile_bytes = get_tile_size(weights_cb_id);

    if constexpr (combine && bias_multicast) {
        Semaphore<> bias_ready(get_named_compile_time_arg_val("bias_ready_semaphore"));
        bias_cb.reserve_back(hidden_tiles);
        nomic_embed_text_v2_moe::multicast_pages(
            noc, bias_accessor, bias_cb, hidden_tiles, bias_tile_bytes, bias_ready, bias_sender, rectangle_args);
        bias_cb.push_back(hidden_tiles);
    } else if constexpr (combine) {
        // Units run token-block-major, so a core with fewer units than chunks covers a run of chunks
        // that may wrap. Each core starts at a different tile of its run, so the cores do not all
        // queue on one DRAM bank first.
        const uint32_t covered_chunks = num_units < chunks ? num_units : chunks;
        const uint32_t covered_tiles = covered_chunks * chunk_tiles;
        const uint32_t first_chunk = first_unit % chunks;
        bias_cb.reserve_back(hidden_tiles);
        for (uint32_t step = 0; step < covered_tiles; ++step) {
            const uint32_t covered_tile = (step + first_unit) % covered_tiles;
            const uint32_t chunk = (first_chunk + covered_tile / chunk_tiles) % chunks;
            const uint32_t tile = chunk * chunk_tiles + covered_tile % chunk_tiles;
            noc.async_read(
                bias_accessor, bias_cb, bias_tile_bytes, {.page_id = tile}, {.offset_bytes = tile * bias_tile_bytes});
        }
        noc.async_read_barrier();
        bias_cb.push_back(hidden_tiles);
    }

    constexpr uint32_t unit_reads = top_k * block_tokens;
    constexpr uint32_t unit_tiles = top_k * chunk_tiles;
    constexpr uint32_t unit_bytes = unit_reads * segment_bytes;
    const uint32_t rows_base = rows_cb.get_write_ptr();
    uint32_t weights_base = 0;
    if constexpr (combine) {
        weights_base = weights_cb.get_write_ptr();
    }
    routed_combine::ReadBatches<batch_reads, batch_depth> batches;
    uint32_t pushed = 0;
    const auto push_done = [&]() {
        for (; pushed < batches.done; ++pushed) {
            rows_cb.push_back(unit_tiles);
            if constexpr (combine) {
                weights_cb.push_back(top_k);
            }
        }
    };

    auto* indices = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_cb.get_write_ptr());
    uint32_t loaded_block = token_blocks;
    for (uint32_t local = 0; local < num_units; ++local) {
        const uint32_t unit = first_unit + local;
        const uint32_t block = unit / chunks;
        const uint32_t chunk = unit % chunks;
        if (block != loaded_block) {
            // The reads in flight took their row ids when they went out. The index read is the most recent:
            // the full barrier, which also lands every batch (prefetching the ids under their own
            // transaction id instead was slower, 2026-11-03: these drains cap the reads in flight).
            for (uint32_t k = 0; k < top_k; ++k) {
                noc.async_read(
                    index_accessor,
                    index_cb,
                    index_block_bytes,
                    {.page_id = k, .offset_bytes = block * index_block_bytes},
                    {.offset_bytes = k * index_block_bytes});
            }
            noc.async_read_barrier();
            batches.retire_all();
            push_done();
            loaded_block = block;
        }

        // This unit's slot is the one unit local - slots held: pushed, then popped by compute. Unit
        // local - slots ends before unit local - 1's batches, so its last batch has a later one behind it.
        while (pushed + slots <= local) {
            batches.retire_next(noc);
            push_done();
        }
        rows_cb.reserve_back((local - pushed + 1) * unit_tiles);
        if constexpr (combine) {
            weights_cb.reserve_back((local - pushed + 1) * top_k);
        }

        const uint32_t weights_slot = weights_base + local % slots * top_k * weight_tile_bytes;
        batches.issue(
            noc,
            rows_accessor,
            rows_base + local % slots * unit_bytes,
            indices,
            chunk * segment_bytes,
            segment_bytes,
            unit_reads,
            local,
            [&](uint32_t trid) {
                if constexpr (combine) {
                    for (uint32_t k = 0; k < top_k; ++k) {
                        noc.async_read<NocOptions::TXN_ID>(
                            weights_accessor,
                            CoreLocalMem<uint32_t>(weights_slot + k * weight_tile_bytes),
                            weight_tile_bytes,
                            {.page_id = k * token_blocks + block},
                            {},
                            {.trid = trid});
                    }
                }
            });
    }
    // The most recent reads: the full barrier.
    noc.async_read_barrier();
    batches.retire_all();
    push_done();
    if constexpr (combine && bias_multicast) {
        if (bias_sender) {
            noc.async_write_barrier();
        }
    }
}
