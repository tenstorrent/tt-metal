// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t has_work_for_q_heads = get_arg(args::has_work_for_q_heads);
    if (has_work_for_q_heads == 0) {
        return;
    }

    uint32_t Mt = get_arg(args::Mt);
    uint32_t Kt = get_arg(args::Kt);
    uint32_t Nt = get_arg(args::Nt);
    uint32_t MtKt = get_arg(args::MtKt);
    uint32_t blocks = get_arg(args::blocks);
    uint32_t in0_start_id = get_arg(args::in0_start_id);
    uint32_t out_start_id = get_arg(args::out_start_id);

    // matmul params
    uint32_t in0_block_w = get_arg(args::in0_block_w);
    uint32_t in1_num_subblocks = get_arg(args::in1_num_subblocks);
    uint32_t in1_num_blocks = get_arg(args::in1_num_blocks);
    uint32_t out_num_tiles = get_arg(args::out_num_tiles);

    // constants
    uint32_t bfloat16_row_bytes = get_arg(args::bfloat16_row_bytes);
    uint32_t bfloat16_Nt_bytes = get_arg(args::bfloat16_Nt_bytes);
    uint32_t bfloat16_last_row_bytes_read = get_arg(args::bfloat16_last_row_bytes_read);

    constexpr uint32_t out_subblock_w = get_arg(args::out_subblock_w);
    constexpr uint32_t intermediate_num_tiles = get_arg(args::intermediate_num_tiles);

    Noc noc;
    DataflowBuffer cb_in0_obj(dfb::in0);
    DataflowBuffer cb_intermed0_obj(dfb::intermed0);
    DataflowBuffer cb_intermed1_obj(dfb::intermed1);
    DataflowBuffer cb_out_obj(dfb::out);

    constexpr uint32_t onetile = 1;
    constexpr uint32_t num_rows_in_one_tile = 32;

    const uint32_t in0_tile_bytes = cb_in0_obj.get_tile_size();

#ifndef IN0_SHARDED
    const auto s0 = TensorAccessor(tensor::src0);
#endif

    const uint32_t out_tile_bytes = cb_out_obj.get_tile_size();
#ifndef OUT_SHARDED
    const auto s = TensorAccessor(tensor::dst);
#endif

#ifndef IN0_SHARDED
    // Only used for interleaved
    uint32_t in0_batch = in0_start_id;
    uint32_t in0_Mt;
    uint32_t in0_tensor_id;
#endif

#ifndef OUT_SHARDED
    uint32_t out_tensor_id = out_start_id;
#endif

    uint32_t bfloat16_row_bytes_read = bfloat16_row_bytes;

    uint32_t local_noc_x = my_x[noc.get_noc_id()];
    uint32_t local_noc_y = my_y[noc.get_noc_id()];
    UnicastEndpoint local_src;

    for (uint32_t b = 0; b < blocks; b++) {  // TODO: Must be 1
#ifndef IN0_SHARDED
        in0_Mt = in0_batch;
#endif

        for (uint32_t m = 0; m < Mt; m++) {  // TODO: Must be 1; generalize to support batch > 32 (ie. Mt > 1)
            // TODO: Generalize to support inner dim blocking; in0 reads has to moved within num_rows_in_one_tile loop
            cb_in0_obj.reserve_back(in0_block_w);
#ifndef IN0_SHARDED
            in0_tensor_id = in0_Mt;
            uint32_t write_offset_in0 = 0;
            for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                // Read in0 block
                noc.async_read(
                    s0, cb_in0_obj, in0_tile_bytes, {.page_id = in0_tensor_id}, {.offset_bytes = write_offset_in0});

                write_offset_in0 += in0_tile_bytes;
                in0_tensor_id++;
            }
            noc.async_read_barrier();
#endif

            cb_in0_obj.push_back(in0_block_w);

            cb_intermed1_obj.reserve_back(out_num_tiles);
            uint32_t cb_intermed1_addr = cb_intermed1_obj.get_write_ptr();
            for (uint32_t in1_block = 0; in1_block < in1_num_blocks; in1_block++) {
                const bool last_out = in1_block == in1_num_blocks - 1;
                if (last_out) {
                    bfloat16_row_bytes_read =
                        bfloat16_last_row_bytes_read;  // For padded subblocks, read partial untilized subblocks
                }

                uint32_t row_offset_bytes = 0;
                uint32_t cb_intermed1_addr_curr_block = cb_intermed1_addr;
                for (uint32_t tile_row_id = 0; tile_row_id < num_rows_in_one_tile; tile_row_id++) {
                    for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks;
                         in1_subblock++) {  // TODO: Must be 1; to generalize, need to handle untilizing + reading for
                                            // subblocks
                        // Read 32 untilized tiles and select correct rows to reconstruct single correct tile
                        cb_intermed0_obj.wait_front(intermediate_num_tiles);
                        uint32_t src_addr = cb_intermed0_obj.get_read_ptr() + row_offset_bytes;
                        CoreLocalMem<uint32_t> local_dst(cb_intermed1_addr_curr_block);
                        noc.async_read(
                            local_src,
                            local_dst,
                            bfloat16_row_bytes_read,
                            {.noc_x = local_noc_x, .noc_y = local_noc_y, .addr = src_addr},
                            {});
                        noc.async_read_barrier();
                        cb_intermed0_obj.pop_front(intermediate_num_tiles);
                        row_offset_bytes += bfloat16_row_bytes;
                        cb_intermed1_addr_curr_block += bfloat16_Nt_bytes;
                    }  // in1_num_subblocks loop

#ifndef IN0_SHARDED
                    in0_Mt += Kt;
#endif
                }  // 32 tiles loop
                cb_intermed1_addr += bfloat16_row_bytes;

            }  // in1_num_blocks loop
            cb_intermed1_obj.push_back(out_num_tiles);

#ifndef OUT_SHARDED
            cb_out_obj.wait_front(out_num_tiles);
            uint32_t read_offset_out = 0;
            for (uint32_t nt = 0; nt < Nt;
                 nt++) {  // TODO: Must be full MtNt; generalize to support Mt > 1 or blocks > 1
                noc.async_write(
                    cb_out_obj, s, out_tile_bytes, {.offset_bytes = read_offset_out}, {.page_id = out_tensor_id});
                read_offset_out += out_tile_bytes;
                out_tensor_id++;
            }
            noc.async_write_barrier();
            cb_out_obj.pop_front(out_num_tiles);
#endif
        }  // Mt loop

#ifndef IN0_SHARDED
        in0_batch += MtKt;
#endif
    }  // B loop

#ifdef OUT_SHARDED
    cb_out_obj.wait_front(out_num_tiles);
#endif
}
