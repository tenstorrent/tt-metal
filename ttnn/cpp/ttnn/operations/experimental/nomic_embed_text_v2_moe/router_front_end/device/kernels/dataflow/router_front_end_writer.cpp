// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RISCV_1 of every core, on NOC_1: the router weight once, read by one core and multicast to the others
// (page_multicast.hpp) on the NoC the input reads leave free; the softmax's padding mask, ttnn.softmax's tile of 0
// on the experts' columns and -inf past them; then each block's rows up to the last token, as soon as the compute
// has untilized them, while it runs the softmax, and its score tile.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/semaphore.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/attention/dataflow/mask_padded_tile.hpp"
#include "ttnn/operations/experimental/nomic_embed_text_v2_moe/common/device/kernels/dataflow/page_multicast.hpp"

void kernel_main() {
    constexpr uint32_t hidden_tiles = get_named_compile_time_arg_val("hidden_tiles");
    constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");
    constexpr uint32_t weight_cb_id = get_named_compile_time_arg_val("weight_cb");
    constexpr uint32_t mask_cb_id = get_named_compile_time_arg_val("mask_cb");
    constexpr uint32_t scores_cb_id = get_named_compile_time_arg_val("scores_cb");
    constexpr uint32_t rows_cb_id = get_named_compile_time_arg_val("rows_cb");
    constexpr bool weight_multicast = get_named_compile_time_arg_val("weight_multicast") == 1;
    constexpr auto weight_args = TensorAccessorArgs<0>();
    constexpr auto scores_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();
    constexpr auto rows_args = TensorAccessorArgs<scores_args.next_compile_time_args_offset()>();

    const uint32_t weight_address = get_arg_val<uint32_t>(0);
    const uint32_t scores_address = get_arg_val<uint32_t>(1);
    const uint32_t rows_address = get_arg_val<uint32_t>(2);
    const uint32_t first_block = get_arg_val<uint32_t>(3);
    const uint32_t num_blocks = get_arg_val<uint32_t>(4);
    const uint32_t num_tokens = get_arg_val<uint32_t>(5);
    // weight_multicast: whether this core sends the weight, then the receivers' rectangles.
    const bool weight_sender = get_arg_val<uint32_t>(6) == 1;
    constexpr uint32_t rectangle_args = 7;

    constexpr uint32_t tile_width = 32;
    constexpr uint32_t row_bytes = hidden_tiles * tile_width * sizeof(uint16_t);

    const Noc noc;
    DataflowBuffer weight_cb(weight_cb_id);
    DataflowBuffer mask_cb(mask_cb_id);
    DataflowBuffer scores_cb(scores_cb_id);
    DataflowBuffer rows_cb(rows_cb_id);
    const auto weight = TensorAccessor(weight_args, weight_address);
    const auto scores = TensorAccessor(scores_args, scores_address);
    const auto rows = TensorAccessor(rows_args, rows_address);
    constexpr uint32_t weight_tile_bytes = get_tile_size(weight_cb_id);
    constexpr uint32_t score_tile_bytes = get_tile_size(scores_cb_id);

    weight_cb.reserve_back(hidden_tiles);
    if constexpr (weight_multicast) {
        Semaphore<> weight_ready(get_named_compile_time_arg_val("weight_ready_semaphore"));
        nomic_embed_text_v2_moe::multicast_pages(
            noc, weight, weight_cb, hidden_tiles, weight_tile_bytes, weight_ready, weight_sender, rectangle_args);
    } else {
        for (uint32_t tile = 0; tile < hidden_tiles; ++tile) {
            noc.async_read(
                weight, weight_cb, weight_tile_bytes, {.page_id = tile}, {.offset_bytes = tile * weight_tile_bytes});
        }
        noc.async_read_barrier();
    }
    weight_cb.push_back(hidden_tiles);

    push_mask_padded_tile<tile_width - num_experts, tile_width * tile_width>(mask_cb);

    for (uint32_t block = first_block; block < first_block + num_blocks; ++block) {
        rows_cb.wait_front(hidden_tiles);
        const uint32_t first_token = block * tile_width;
        const uint32_t block_tokens = num_tokens - first_token < tile_width ? num_tokens - first_token : tile_width;
        for (uint32_t row = 0; row < block_tokens; ++row) {
            noc.async_write(
                rows_cb, rows, row_bytes, {.offset_bytes = row * row_bytes}, {.page_id = first_token + row});
        }
        scores_cb.wait_front(1);
        noc.async_write(scores_cb, scores, score_tile_bytes, {}, {.page_id = block});
        noc.async_write_barrier();
        rows_cb.pop_front(hidden_tiles);
        scores_cb.pop_front(1);
    }
    if constexpr (weight_multicast) {
        if (weight_sender) {
            noc.async_write_barrier();
        }
    }
}
