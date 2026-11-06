// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/kernel_helper_functions/pad_tile.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // RUNTIME ARGS
    // in0 tensor args
    auto in0_tensor_start_tile_id = get_arg(args::in0_tensor_start_tile_id);
    // batch args: this core computes `batch` consecutive output blocks, starting at M block start_m_block
    const auto batch = get_arg(args::batch);
    const auto start_m_block = get_arg(args::start_m_block);

    // COMPILE TIME ARGS
    // in0 tensor args
    constexpr auto in0_tensor_stride_w = get_arg(args::in0_tensor_stride_w);
    constexpr auto in0_tensor_stride_h = get_arg(args::in0_tensor_stride_h);
    constexpr auto in0_tensor_next_block_stride = get_arg(args::in0_tensor_next_block_stride);
    // in0 block args
    constexpr auto in0_block_w = get_arg(args::in0_block_w);
    constexpr auto in0_block_h = get_arg(args::in0_block_h);
    constexpr auto in0_block_num_tiles = get_arg(args::in0_block_num_tiles);
    constexpr auto last_ktile_w = get_arg(args::last_ktile_w);
    constexpr auto last_ktile_h = get_arg(args::last_ktile_h);
    // in0/in1 common args
    constexpr auto num_blocks = get_arg(args::num_blocks);
    // batch args
    constexpr auto bcast_B = get_arg(args::bcast_B);
    constexpr auto MtKt = get_arg(args::MtKt);
    // Consecutive blocks walk the M blocks of a batch before moving to the next batch
    constexpr auto m_blocks_per_batch = get_arg(args::m_blocks_per_batch);
    constexpr auto in0_m_block_stride = get_arg(args::in0_m_block_stride);

    const Noc noc;
    // in0 block staging: the reader fills it, the compute kernel drains it.
    DataflowBuffer dfb_in0(dfb::in0);

#ifdef IN0_SHARDED
    const uint32_t in0_num_tiles = batch * num_blocks * in0_block_h * in0_block_w;
    dfb_in0.reserve_back(in0_num_tiles);
    dfb_in0.push_back(in0_num_tiles);
#else

    constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(dfb::in0);
    // Tiles whose size is not a multiple of the DRAM alignment are padded to it in DRAM and the in0
    // buffer's entries are sized to match (see the program factory), so tiles must be laid out in L1
    // at the padded stride. The NOC still reads the unpadded tile of data into each padded slot.
    // No-op when the tile size is already aligned.
    constexpr uint32_t in0_aligned_tile_size_bytes =
        (in0_single_tile_size_bytes + (DRAM_ALIGNMENT - 1)) & ~(DRAM_ALIGNMENT - 1);

    const auto s0 = TensorAccessor(tensor::in0);

    uint32_t m_block = start_m_block;
    for (uint32_t b = 0; b < batch; ++b) {
        uint32_t in0_tensor_current_block_start_tile_id = in0_tensor_start_tile_id;
        for (uint32_t block = 0; block < num_blocks; ++block) {
            dfb_in0.reserve_back(in0_block_num_tiles);

            uint32_t in0_write_offset = 0;

            uint32_t in0_tensor_row_start_tile_id = in0_tensor_current_block_start_tile_id;
            for (uint32_t h = 0; h < in0_block_h; ++h) {
                uint32_t in0_tensor_tile_id = in0_tensor_row_start_tile_id;
                for (uint32_t w = 0; w < in0_block_w; ++w) {
                    noc.async_read(
                        s0,
                        dfb_in0,
                        in0_single_tile_size_bytes,
                        {.page_id = in0_tensor_tile_id},
                        {.offset_bytes = in0_write_offset});

                    // Zero out padded regions for the very last tile
                    if constexpr (last_ktile_w > 0) {
                        if ((block == num_blocks - 1) && (w == in0_block_w - 1)) {
                            noc.async_read_barrier();
                            constexpr DataFormat in0_data_format = get_dataformat(dfb::in0);
                            pad_last_ktile<in0_data_format, last_ktile_w>(dfb_in0.get_write_ptr() + in0_write_offset);
                        }
                    }
                    if constexpr (last_ktile_h > 0) {
                        if ((block == num_blocks - 1) && (w == in0_block_w - 1)) {
                            noc.async_read_barrier();
                            constexpr DataFormat in0_data_format = get_dataformat(dfb::in0);
                            pad_last_transposed_ktile<in0_data_format, last_ktile_h>(
                                dfb_in0.get_write_ptr() + in0_write_offset);
                        }
                    }

                    in0_write_offset += in0_aligned_tile_size_bytes;
                    in0_tensor_tile_id += in0_tensor_stride_w;
                }
                in0_tensor_row_start_tile_id += in0_tensor_stride_h;
            }
            in0_tensor_current_block_start_tile_id += in0_tensor_next_block_stride;

            noc.async_read_barrier();

            dfb_in0.push_back(in0_block_num_tiles);
        }
        if (++m_block == m_blocks_per_batch) {
            // Next batch, first M block
            m_block = 0;
            in0_tensor_start_tile_id += MtKt - (m_blocks_per_batch - 1) * in0_m_block_stride;
        } else {
            in0_tensor_start_tile_id += in0_m_block_stride;
        }
    }
#endif  // IN0_SHARDED
}
