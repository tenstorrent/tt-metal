// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/local_tensor_accessor.h"
#include "experimental/kernel_args.h"

// A sharded tensor is laid out as its 2D flattening, so a height concat has to interleave per
// leading index rather than append whole shards (#55342): block b of input i lands
// b * output_block_stride into the output shard and starts b * input_block_stride into the input.
// num_blocks is 1 for width concat and for the rank-4 (1, 1, H, W) case, where the block loop
// runs once and the copy reduces to a plain append.
void kernel_main() {
    constexpr uint32_t page_size = get_arg(args::page_size);
    constexpr uint32_t output_stride = get_arg(args::output_stride);
    constexpr uint32_t output_block_stride = get_arg(args::output_block_stride);
    // This RISC's share of the leading blocks, baked in per kernel so the loop bound stays a
    // compile-time constant; as a runtime arg it cost a few percent on small copies.
    constexpr uint32_t block_start = get_arg(args::block_start);
    constexpr uint32_t block_count = get_arg(args::block_count);
    constexpr uint32_t block_end = block_start + block_count;

    Noc noc;
    // The input shards and the output shard are all in this core's L1, so each copy below is a
    // NoC read from this core to itself. Addresses are carried as byte offsets into each shard.
    const auto inputs = make_local_tensor_accessors<uint32_t>(tensor::inputs);
    const LocalTensorAccessor<uint32_t> output(tensor::output);

    for (uint32_t input_id = 0; input_id < inputs.size(); input_id++) {
        // input_num_sticks is this RISC's sticks per block, for each block in [block_start,
        // block_end): every stick of those blocks when the host split the blocks, or half of each
        // block's sticks when it split the sticks instead and both RISCs cover all the blocks.
        const uint32_t input_num_pages_per_stick = get_arg(args::input_num_pages_per_stick)[input_id];
        const uint32_t input_num_sticks = get_arg(args::input_num_sticks)[input_id];
        const uint32_t input_write_offset = get_arg(args::input_write_offset)[input_id];
        const uint32_t input_read_offset = get_arg(args::input_read_offset)[input_id];
        const uint32_t input_block_stride = get_arg(args::input_block_stride)[input_id];

        const LocalTensorAccessor<uint32_t>& input = inputs[input_id];

        // Once per input, not once per block. This programs page_size and the burst config;
        // every async_read_with_state below passes its own address, which is why l1_read_offset can
        // advance per page without re-setting state. Re-issuing it per block would cost a
        // leading-index multiple of the NOC state programming for no behavioural difference.
        // Over the limit, set-state is unusable: it holds the length in the command buffer and
        // the splitting variant rewrites it per packet, so overlapping reads see each other's length.
        if constexpr (page_size <= NOC_MAX_BURST_SIZE) {
            noc.set_async_read_state<NocOptions::DEFAULT, page_size>(
                input, page_size, {.offset_bytes = input_read_offset});
        }

        for (uint32_t block_idx = block_start; block_idx < block_end; block_idx++) {
            uint32_t l1_write_offset = input_write_offset + block_idx * output_block_stride;
            uint32_t l1_read_offset = input_read_offset + block_idx * input_block_stride;

            for (uint32_t stick_idx = 0; stick_idx < input_num_sticks; stick_idx++) {
                for (uint32_t page_idx = 0; page_idx < input_num_pages_per_stick; page_idx++) {
                    const uint32_t dst_offset = l1_write_offset + page_size * page_idx;
                    if constexpr (page_size <= NOC_MAX_BURST_SIZE) {
                        noc.async_read_with_state<NocOptions::DEFAULT, page_size>(
                            input, output, page_size, {.offset_bytes = l1_read_offset}, {.offset_bytes = dst_offset});
                    } else {
                        noc.async_read(
                            input, output, page_size, {.offset_bytes = l1_read_offset}, {.offset_bytes = dst_offset});
                    }
                    l1_read_offset += page_size;
                }
                l1_write_offset += output_stride;
            }
        }
    }

    noc.async_read_barrier();
}
