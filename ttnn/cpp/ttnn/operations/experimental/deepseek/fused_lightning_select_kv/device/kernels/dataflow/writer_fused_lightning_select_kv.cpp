// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writes this core's index scores to the scores output, then drains one gathered kv entry
// (boilerplate) to the kv output.

#define COMPRESS_RATE 4
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

void kernel_main() {
    // ---- Compile-time args ----
    [[maybe_unused]] constexpr uint32_t k = get_arg(args::k);
    constexpr uint32_t page_block_size = get_arg(args::page_block_size);
    constexpr uint32_t chunks_per_block = page_block_size / 32;

    // ---- Runtime args ----
    const uint32_t core_index = get_arg(args::core_index);
    const uint32_t num_cores = get_arg(args::num_cores);

    // ---- Tensors ----
    const auto output = TensorAccessor(tensor::output);
    const auto scores = TensorAccessor(tensor::scores);  // [1, 1, 1, T] fp32, one row-major page

    // ---- Dataflow buffers ----
    DataflowBuffer kv_dfb(dfb::kv);            // consumer <- reader
    DataflowBuffer cur_pos_dfb(dfb::cur_pos);  // consumer <- reader
    DataflowBuffer scores_dfb(dfb::scores);    // consumer <- compute

    Noc noc;

    // Same block split as the reader.
    cur_pos_dfb.wait_front(1);
    const uint32_t cur_pos_value = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cur_pos_dfb.get_read_ptr())[0];
    cur_pos_dfb.pop_front(1);
    const uint32_t num_keys = (cur_pos_value + 1) / COMPRESS_RATE;
    const uint32_t total_num_blocks = tt::data_movement::common::div_up(num_keys, page_block_size);
    const uint32_t work_per_core = tt::data_movement::common::div_up(total_num_blocks, num_cores);
    const uint32_t start_block = std::min(work_per_core * core_index, total_num_blocks);
    const uint32_t end_block = std::min(start_block + work_per_core, total_num_blocks);

    // A 1x32 tile is 32 consecutive values, i.e. already the row-major layout of 32 scores.
    const uint32_t score_tile_bytes = scores_dfb.get_entry_size();
    for (uint32_t block = start_block; block < end_block; ++block) {
        for (uint32_t chunk = 0; chunk < chunks_per_block; ++chunk) {
            const uint32_t first_key = block * page_block_size + chunk * 32;
            scores_dfb.wait_front(1);
            noc.async_write(
                scores_dfb, scores, score_tile_bytes, {}, {.page_id = 0, .offset_bytes = first_key * sizeof(float)});
            noc.async_write_barrier();
            scores_dfb.pop_front(1);
        }
    }

    kv_dfb.wait_front(1);
    if (core_index == 0) {
        noc.async_write(kv_dfb, output, kv_dfb.get_entry_size(), {}, {.page_id = 0});
        noc.async_write_barrier();
    }
    kv_dfb.pop_front(1);
}
