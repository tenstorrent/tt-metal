// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RM last-dim interleaved; in-CB page doubling for odd page sizes.
// Reads from RM and writes to RM repeating the last dimension.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

using namespace tt::data_movement::common;

void kernel_main() {
    // Which set of pages to deal with.
    const auto page_start = get_arg(args::page_start);
    const auto page_end = get_arg(args::page_end);
    // If work is not divided up nicely between cores / tensor too small, skip this core.
    const auto nop = get_arg(args::nop);

    constexpr auto original_page_size_bytes = get_arg(args::original_page_size_bytes);
    constexpr auto num_repeats = get_arg(args::num_repeats);
    constexpr uint32_t dest_page_size_bytes = original_page_size_bytes * num_repeats;

    // Since we need to operate on a grid of cores but sometimes pages don't split properly,
    // if nop then don't use this core.
    if (nop == 1) {
        return;
    }

    const auto s = TensorAccessor(tensor::src);
    const auto d = TensorAccessor(tensor::dst);

    Noc noc;
    // dfb::in0 and dfb::in1 are each 1 page; size depends on alignment:
    //   multiple of 16 -> original_page_size_bytes + 128
    //   multiple of  8 -> original_page_size_bytes * 2  + 128
    //   multiple of  4 -> original_page_size_bytes * 4  + 128
    //   multiple of  2 -> original_page_size_bytes * 8  + 128
    //   odd            -> original_page_size_bytes * 16 + 128
    DataflowBuffer dfb0(dfb::in0);
    DataflowBuffer dfb1(dfb::in1);

    // Number of times we must double the input page to make it write-aligned.
    constexpr uint32_t num_doublings = ((original_page_size_bytes % 16) == 0)  ? 0
                                       : ((original_page_size_bytes % 8) == 0) ? 1
                                       : ((original_page_size_bytes % 4) == 0) ? 2
                                       : ((original_page_size_bytes % 2) == 0) ? 3
                                                                               : 4;
    // Max write size after doublings; used as template parameter to enable fast path.
    constexpr uint32_t max_write_size = original_page_size_bytes << num_doublings;

    dfb0.reserve_back(1);
    dfb1.reserve_back(1);
    uint32_t input_buffer = dfb0.get_write_ptr();
    uint32_t alignment_buffer = dfb1.get_write_ptr();
    dfb1.push_back(1);
    dfb0.push_back(1);

    constexpr uint64_t r_mask_to_use = decltype(s)::is_dram ? MASK_64 : MASK_16;
    constexpr uint64_t r_offset_to_use = decltype(s)::is_dram ? OFFSET_64 : OFFSET_16;
    constexpr uint32_t r_alignment_requirement = decltype(s)::is_dram ? 64 : 16;
    constexpr uint32_t w_alignment_requirement = 16;
    constexpr uint64_t w_mask_to_use = MASK_16;
    constexpr uint64_t w_offset_to_use = OFFSET_16;

    alignment_buffer = align_address<w_alignment_requirement>(alignment_buffer, w_mask_to_use);
    input_buffer = align_address<r_alignment_requirement>(input_buffer, r_mask_to_use);

    uint32_t cur_page_size = original_page_size_bytes;
    for (uint32_t i = page_start; i < page_end; i++) {
        uint64_t src_noc_addr = s.get_noc_addr(i, 0);
        uint64_t dst_noc_addr = d.get_noc_addr(i, 0);
        uint32_t data_location = input_buffer + (src_noc_addr & r_offset_to_use);

        CoreLocalMem<uint32_t> dst_mem(data_location);
        noc.async_read<NocOptions::DEFAULT, original_page_size_bytes>(
            s, dst_mem, original_page_size_bytes, {.page_id = i, .offset_bytes = 0}, {.offset_bytes = 0});
        cur_page_size = original_page_size_bytes;
        noc.async_read_barrier();
        if constexpr (num_doublings != 0) {
            uint32_t target_offset = original_page_size_bytes;
            for (uint32_t j = 0; j < num_doublings; j++) {
                tt_memmove<false, false, false, 16 * original_page_size_bytes>(
                    noc, data_location + target_offset, data_location, cur_page_size);
                target_offset += cur_page_size;
                cur_page_size *= 2;
            }
        }
        if ((data_location & w_offset_to_use) != (dst_noc_addr & w_offset_to_use)) {
            tt_memmove<false, false, false, 16 * original_page_size_bytes>(
                noc, alignment_buffer + (dst_noc_addr & w_offset_to_use), data_location, cur_page_size);
            data_location = alignment_buffer + (dst_noc_addr & w_offset_to_use);
        }

        uint64_t num_written = 0;
        while (num_written < dest_page_size_bytes) {
            uint32_t to_write = (dest_page_size_bytes - num_written) > cur_page_size
                                    ? cur_page_size
                                    : (dest_page_size_bytes - num_written);

            CoreLocalMem<uint32_t> src_mem(data_location);
            noc.async_write<NocOptions::DEFAULT, max_write_size>(
                src_mem,
                d,
                to_write,
                {.offset_bytes = 0},
                {.page_id = i, .offset_bytes = static_cast<uint32_t>(num_written)});
            num_written += to_write;
        }
        noc.async_write_barrier();
    }
}
