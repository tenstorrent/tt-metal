// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/kernel_helper_functions/pad_tile.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    // RUNTIME ARGS
    uint32_t rt_args_idx = 0;
    // in0 tensor args
    const uint32_t in0_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t in0_tensor_start_tile_id = get_arg_val<uint32_t>(rt_args_idx++);
    // batch args
    const uint32_t batch = get_arg_val<uint32_t>(rt_args_idx++);

    // COMPILE TIME ARGS
    // in0 tensor args
    constexpr uint32_t in0_tensor_stride_w = get_compile_time_arg_val(0);
    constexpr uint32_t in0_tensor_stride_h = get_compile_time_arg_val(1);
    constexpr uint32_t in0_tensor_next_block_stride = get_compile_time_arg_val(2);
    // in0 block args
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(3);
    constexpr uint32_t in0_block_h = get_compile_time_arg_val(4);
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t last_ktile_w = get_compile_time_arg_val(6);
    constexpr uint32_t last_ktile_h = get_compile_time_arg_val(7);
    // in0/in1 common args
    constexpr uint32_t num_blocks = get_compile_time_arg_val(8);
    // batch args
    constexpr uint32_t bcast_B = get_compile_time_arg_val(9);
    constexpr uint32_t MtKt = get_compile_time_arg_val(10);

    constexpr auto in0_args = TensorAccessorArgs<11>();

    constexpr uint32_t dfb_id_in0 = get_named_compile_time_arg_val("dfb_in0");

    Noc noc;
    DataflowBuffer dfb_in0(dfb_id_in0);

#ifdef IN0_SHARDED
    const uint32_t in0_num_tiles = batch * num_blocks * in0_block_h * in0_block_w;
    dfb_in0.reserve_back(in0_num_tiles);
    dfb_in0.push_back(in0_num_tiles);
#else

    constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(dfb_id_in0);
    constexpr const uint32_t in0_tile_hw = get_tile_hw(dfb_id_in0);
    // Tiles whose size is not a multiple of the DRAM alignment are padded to it in DRAM and the in0
    // CB pages are sized to match (see the program factory), so tiles must be laid out in L1 at the
    // padded stride. The NOC still reads the unpadded tile of data into each padded slot. No-op when
    // the tile size is already aligned.
    constexpr uint32_t in0_aligned_tile_size_bytes =
        (in0_single_tile_size_bytes + (DRAM_ALIGNMENT - 1)) & ~(DRAM_ALIGNMENT - 1);

    const auto s0 = TensorAccessor(in0_args, in0_tensor_addr);

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
                            constexpr DataFormat in0_data_format = get_dataformat(dfb_id_in0);
                            pad_last_ktile<in0_data_format, last_ktile_w>(dfb_in0.get_write_ptr() + in0_write_offset);
                        }
                    }
                    if constexpr (last_ktile_h > 0) {
                        if ((block == num_blocks - 1) && (w == in0_block_w - 1)) {
                            noc.async_read_barrier();
                            constexpr DataFormat in0_data_format = get_dataformat(dfb_id_in0);
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
        in0_tensor_start_tile_id += MtKt;
    }
#endif  // IN0_SHARDED
}
