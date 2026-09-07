// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: This is the Metal 2.0 fork of writer_unary_stick_layout_interleaved_blocks.cpp, which lives
// beside it. Ops ported to Metal 2.0 bind this file; the original serves the consumers still on the
// legacy API. Until the last of them migrates and the original is retired, changes here likely belong
// there too.
//
// The binding names below (dfb::out, tensor::dst) and the named argument set are this fork's
// interface: every later consumer inherits them, so they are taken from the kernel's own vocabulary
// rather than any one op's locals, and are not renamed once a consumer exists.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
// #include "api/debug/dprint.h"

template <typename DSpec>
inline void write_tiles_in_block(
    DataflowBuffer& dfb_out0,
    const Noc& noc,
    uint32_t block_height_ntiles,
    uint32_t block_width_ntiles,
    uint32_t block_start_row_id,
    uint32_t block_row_offset,
    uint32_t block_row_size,
    uint32_t block_row_size_unpadded,  // to remove padding from the last block in the row
    uint32_t num_rows_unpadded,
    const TensorAccessor<DSpec>& s) {
    constexpr uint32_t TILE_HEIGHT = 32;  // TODO: use common source of truth
    uint32_t block_row_id = block_start_row_id;
    for (uint32_t tile_row_id = 0; tile_row_id < block_height_ntiles; tile_row_id++) {
        // We reserve back an entire row of tiles in a block and issue a bunch of reads
        dfb_out0.wait_front(block_width_ntiles);
        uint32_t l1_read_addr = dfb_out0.get_read_ptr();
        for (uint32_t j = 0; j < TILE_HEIGHT; j++) {
            if (block_row_id >= num_rows_unpadded) {
                break;
            }
            CoreLocalMem<uint32_t> src(l1_read_addr);
            noc.async_write(
                src,
                s,
                block_row_size_unpadded,
                {.offset_bytes = 0},
                {.page_id = block_row_id, .offset_bytes = block_row_offset});
            l1_read_addr += block_row_size;
            block_row_id++;
        }  // for tile_nrows
        noc.async_write_barrier();
        dfb_out0.pop_front(block_width_ntiles);
    }  // for block_height_ntiles
}
void kernel_main() {
    auto num_rows_block = get_arg(args::num_rows_block);
    auto block_row_size = get_arg(args::block_row_size);  // in0_block_w * TILE_WIDTH * dtype_nbytes
    auto batch = get_arg(args::batch);
    auto num_blocks_h = get_arg(args::num_blocks_h);
    auto num_blocks_w = get_arg(args::num_blocks_w);
    auto last_block_row_size_unpadded = get_arg(args::last_block_row_size_unpadded);  // unpadded last block width
    auto num_output_rows_unpadded = get_arg(args::num_output_rows_unpadded);
    auto block_start_row_id = get_arg(args::block_start_row_id);
    auto block_start_row_offset = get_arg(args::block_start_row_offset);

    constexpr bool FLOAT32_DTYPE = get_arg(args::float32_dtype) == 1;
    // args::output_row_size is declared by every binding factory but never read here; it is carried
    // in the schema so the argument set matches what callers have always supplied. Reading it is not
    // required and it may be retired once every caller drops it.

    constexpr uint32_t TILE_HEIGHT = 32;  // TODO: use common source of truth

    const uint32_t block_width_ntiles =
        FLOAT32_DTYPE ? block_row_size >> 7
                      : block_row_size >> 6;  // Assuming 4/2 bytes per datum, there are 128/64 bytes per tile row
    const uint32_t block_height_ntiles = num_rows_block / TILE_HEIGHT;

    const auto s = TensorAccessor(tensor::dst);

    Noc noc;
    // NOTE: Row major layout only supports bfp16
    DataflowBuffer dfb_out0(dfb::out);

    uint32_t num_rows_unpadded = num_output_rows_unpadded + block_start_row_id;
    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t block_h = 0; block_h < num_blocks_h; block_h++) {
            uint32_t block_row_offset = block_start_row_offset;
            for (uint32_t block_w = 0; block_w < num_blocks_w; block_w++) {
                uint32_t current_block_row_size_unpadded = block_row_size;
                if (block_w == (num_blocks_w - 1)) {
                    current_block_row_size_unpadded = last_block_row_size_unpadded;
                }
                write_tiles_in_block(
                    dfb_out0,
                    noc,
                    block_height_ntiles,
                    block_width_ntiles,
                    block_start_row_id,
                    block_row_offset,
                    block_row_size,
                    current_block_row_size_unpadded,  // padding is only in the last block
                    num_rows_unpadded,
                    s);
                block_row_offset += block_row_size;
            }  // for num_blocks_w
            block_start_row_id += num_rows_block;
        }  // for num_blocks_h
    }  // for batch
}
