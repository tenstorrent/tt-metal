// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// conv2d_3x3 writer (BRISC / NoC1): prefetch_bias once, then per M-block stream_weight_blocks and
// store_block. Weight K-blocks of the next M-block are prefetched between output row groups, but only
// into slots that are already free, so the writer never blocks on cb_weights while compute waits on
// the output CB.
//
// Sharded output: cb_output_tiles is backed on the output shard, so compute packs in place and
// store_block is empty. The writer only streams bias and weights; compute never waits on the output
// CB there, so blocking weight reserves cannot deadlock.
//
// Weight multicast (regime R1, the `weight_mcast` channel): every K-block is one round. The sender
// reads it from DRAM into its own cb_weights slot and multicasts it to the same slot on every
// receiver; a receiver acks only after reserving that slot, so the slot is free. All participants
// push the same quanta in the same order, so their cb_weights write pointers agree every round.
// A core outside the channel (one with fewer M-blocks, or an absent channel) reads its own weights.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

void kernel_main() {
    constexpr uint32_t cb_weights = get_compile_time_arg_val(0);
    constexpr uint32_t cb_bias = get_compile_time_arg_val(1);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t block_m_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t block_k_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t block_n_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(6);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(7);
    constexpr uint32_t tensor_n_tiles = get_compile_time_arg_val(8);
    constexpr bool has_bias = get_compile_time_arg_val(9) != 0;
    constexpr bool output_sharded = get_compile_time_arg_val(10) != 0;
    constexpr auto weight_args = TensorAccessorArgs<11>();
    [[maybe_unused]] constexpr auto bias_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto output_args = TensorAccessorArgs<bias_args.next_compile_time_args_offset()>();
    constexpr auto weight_mcast = dataflow_kernel_lib::McastArgs<
        get_named_compile_time_arg_val("weight_mcast_ct_offset"),
        get_named_compile_time_arg_val("weight_mcast_rt_offset")>();

    const uint32_t weight_addr = get_arg_val<uint32_t>(0);
    [[maybe_unused]] const uint32_t bias_addr = get_arg_val<uint32_t>(1);
    [[maybe_unused]] const uint32_t output_addr = get_arg_val<uint32_t>(2);
    [[maybe_unused]] const uint32_t m_tile_start = get_arg_val<uint32_t>(3);
    [[maybe_unused]] const uint32_t actual_m_tiles = get_arg_val<uint32_t>(4);
    const uint32_t num_m_blocks_this_core = get_arg_val<uint32_t>(5);
    const uint32_t n_tile_start = get_arg_val<uint32_t>(6);

    constexpr uint32_t weight_k_block_tiles = block_k_tiles * block_n_tiles;
    constexpr uint32_t row_group_tiles = out_subblock_h * block_n_tiles;
    constexpr uint32_t row_groups_per_block = block_m_tiles / out_subblock_h;
    const uint32_t weight_tile_bytes = get_tile_size(cb_weights);
    const auto weights = TensorAccessor(weight_args, weight_addr, weight_tile_bytes);

    // ---- prefetch_bias: stays fronted in cb_bias for the whole kernel.
    if constexpr (has_bias) {
        const uint32_t bias_tile_bytes = get_tile_size(cb_bias);
        const auto bias = TensorAccessor(bias_args, bias_addr, bias_tile_bytes);
        cb_reserve_back(cb_bias, block_n_tiles);
        uint32_t l1 = get_write_ptr(cb_bias);
        for (uint32_t n = 0; n < block_n_tiles; ++n, l1 += bias_tile_bytes) {
            noc_async_read_page(n_tile_start + n, bias, l1);
        }
        noc_async_read_barrier();
        cb_push_back(cb_bias, block_n_tiles);
    }

    // ---- stream_weight_blocks: K-block tiles k-major, n-fastest (matmul in1 layout).
    const uint32_t total_k_blocks = num_m_blocks_this_core * num_k_blocks;
    uint32_t k_blocks_issued = 0;
    auto read_weight_k_block = [&](uint32_t l1) {
        const uint32_t k_tile0 = (k_blocks_issued % num_k_blocks) * block_k_tiles;
        for (uint32_t k = 0; k < block_k_tiles; ++k) {
            const uint32_t row_page = (k_tile0 + k) * tensor_n_tiles + n_tile_start;
            for (uint32_t n = 0; n < block_n_tiles; ++n, l1 += weight_tile_bytes) {
                noc_async_read_page(row_page + n, weights, l1);
            }
        }
        noc_async_read_barrier();
    };

    Noc noc;
    auto weight_sender = weight_mcast.optional_sender(noc);
    auto weight_receiver = weight_mcast.optional_receiver(noc);
    auto issue_weight_k_block = [&]() {
        cb_reserve_back(cb_weights, weight_k_block_tiles);
        const uint32_t l1 = get_write_ptr(cb_weights);
        if (weight_sender && weight_mcast.should_send(k_blocks_issued)) {
            read_weight_k_block(l1);
            weight_sender->send(l1, l1, weight_k_block_tiles * weight_tile_bytes);
        } else if (weight_receiver) {
            weight_receiver->receive(k_blocks_issued);
        } else {
            read_weight_k_block(l1);
        }
        cb_push_back(cb_weights, weight_k_block_tiles);
        ++k_blocks_issued;
    };

    for (uint32_t m_block_idx = 0; m_block_idx < num_m_blocks_this_core; ++m_block_idx) {
        while (k_blocks_issued < (m_block_idx + 1) * num_k_blocks) {
            issue_weight_k_block();
        }
        if constexpr (!output_sharded) {
            // ---- store_block: valid output tile-rows only; the ragged tail keeps nominal CB counts.
            const uint32_t out_tile_bytes = get_tile_size(cb_output_tiles);
            const auto output = TensorAccessor(output_args, output_addr, out_tile_bytes);
            const uint32_t block_tile0 = m_block_idx * block_m_tiles;
            for (uint32_t row_group = 0; row_group < row_groups_per_block; ++row_group) {
                cb_wait_front(cb_output_tiles, row_group_tiles);
                uint32_t l1 = get_read_ptr(cb_output_tiles);
                for (uint32_t r = 0; r < out_subblock_h; ++r) {
                    const uint32_t local_tile_row = block_tile0 + row_group * out_subblock_h + r;
                    if (local_tile_row < actual_m_tiles) {
                        const uint32_t row_page = (m_tile_start + local_tile_row) * tensor_n_tiles + n_tile_start;
                        for (uint32_t n = 0; n < block_n_tiles; ++n) {
                            noc_async_write_page(row_page + n, output, l1 + n * out_tile_bytes);
                        }
                    }
                    l1 += block_n_tiles * out_tile_bytes;
                }
                noc_async_writes_flushed();
                cb_pop_front(cb_output_tiles, row_group_tiles);

                while (k_blocks_issued < total_k_blocks &&
                       cb_pages_reservable_at_back(cb_weights, weight_k_block_tiles)) {
                    issue_weight_k_block();
                }
            }
        }
    }
    if constexpr (!output_sharded) {
        noc_async_write_barrier();
    }
}
