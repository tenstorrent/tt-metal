// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RM higher-dim interleaved; alignment dance for one-packet noc.async_* fast path.
// Tensor layout: <higher_dim, rep_dim, lower_dim, page_size>

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
    // Program factory controls the start and end of each of the 3 dims.
    const auto higher_dim_start = get_arg(args::higher_dim_start);
    const auto higher_dim_end = get_arg(args::higher_dim_end);
    const auto lower_dim_start = get_arg(args::lower_dim_start);
    const auto lower_dim_end = get_arg(args::lower_dim_end);
    const auto repetitions = get_arg(args::repetitions);
    // nop lets you intentionally skip this core if dims don't divide evenly.
    const auto nop = get_arg(args::nop);

    constexpr auto original_page_size_bytes = get_arg(args::original_page_size_bytes);
    constexpr auto LOWER_DIMS = get_arg(args::LOWER_DIMS);
    constexpr auto REP_DIM = get_arg(args::REP_DIM);

    constexpr uint32_t LOWER_DIMS_TIMES_REP_DIM = LOWER_DIMS * REP_DIM;

    // Since we need to operate on a grid of cores but sometimes pages don't split properly,
    // if nop then don't use this core.
    if (nop == 1) {
        return;
    }

    const auto s = TensorAccessor(tensor::src);
    const auto d = TensorAccessor(tensor::dst);

    Noc noc;
    // dfb::in0 and dfb::in1 are each 1 page of size: 128 + page_size_bytes.
    DataflowBuffer dfb0(dfb::in0);
    DataflowBuffer dfb1(dfb::in1);

    // Alignment pre-calculations.
    constexpr uint64_t r_mask_to_use = decltype(s)::is_dram ? MASK_64 : MASK_16;
    constexpr uint64_t r_offset_to_use = decltype(s)::is_dram ? OFFSET_64 : OFFSET_16;
    constexpr uint32_t r_alignment_requirement = decltype(s)::is_dram ? 64 : 16;
    constexpr uint32_t w_alignment_requirement = 16;
    const uint64_t w_mask_to_use = MASK_16;
    const uint64_t w_offset_to_use = OFFSET_16;

    dfb0.reserve_back(1);
    dfb1.reserve_back(1);
    uint32_t input_buffer = dfb0.get_write_ptr();
    uint32_t alignment_buffer = dfb1.get_write_ptr();
    dfb1.push_back(1);
    dfb0.push_back(1);

    alignment_buffer = align_address<w_alignment_requirement>(alignment_buffer, w_mask_to_use);  // aligned for writes
    input_buffer = align_address<r_alignment_requirement>(input_buffer, r_mask_to_use);          // aligned for reads

    uint64_t src_noc_addr = 0;
    uint32_t data_location = 0;

    for (uint32_t h = higher_dim_start; h < higher_dim_end; h++) {
        uint32_t h_offset = h * LOWER_DIMS_TIMES_REP_DIM;
        uint32_t h_offset_rep = h_offset * repetitions;
        for (uint32_t r = 0; r < REP_DIM; r++) {
            uint32_t r_offset = r * LOWER_DIMS;
            for (uint32_t l = lower_dim_start; l < lower_dim_end; l++) {
                uint32_t read_offset = h_offset + r_offset + l;
                src_noc_addr = s.get_noc_addr(read_offset, 0);
                data_location = input_buffer + (src_noc_addr & r_offset_to_use);

                CoreLocalMem<uint32_t> dst_mem(data_location);
                noc.async_read<NocOptions::DEFAULT, original_page_size_bytes>(
                    s,
                    dst_mem,
                    original_page_size_bytes,
                    {.page_id = read_offset, .offset_bytes = 0},
                    {.offset_bytes = 0});
                noc.async_read_barrier();

                for (uint32_t n = 0; n < repetitions; n++) {
                    uint32_t write_offset = h_offset_rep + n * LOWER_DIMS_TIMES_REP_DIM + r_offset + l;
                    const uint64_t dst_noc_addr = d.get_noc_addr(write_offset, 0);
                    if ((data_location & w_offset_to_use) != (dst_noc_addr & w_offset_to_use)) {
                        const uint32_t target_align_buffer = alignment_buffer + (dst_noc_addr & w_offset_to_use);
                        tt_memmove<false, false, false, original_page_size_bytes>(
                            noc, target_align_buffer, data_location, original_page_size_bytes);
                        data_location = alignment_buffer + (dst_noc_addr & w_offset_to_use);
                    }
                    CoreLocalMem<uint32_t> src_mem(data_location);
                    noc.async_write<NocOptions::DEFAULT, original_page_size_bytes>(
                        src_mem,
                        d,
                        original_page_size_bytes,
                        {.offset_bytes = 0},
                        {.page_id = write_offset, .offset_bytes = 0});
                }
                noc.async_write_barrier();
            }
        }
    }
}
