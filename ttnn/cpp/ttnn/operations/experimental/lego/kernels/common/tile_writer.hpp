// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Write num_tiles tiles from a CB to an interleaved tensor, starting at tile_offset, blk tiles at a time; every
// row_tiles tiles the destination jumps to the next row, row_stride tiles further (row_stride = row_tiles: contiguous).
// Port of layernorm_distributed/.../writer_unary_interleaved_start_id_blocked.cpp (+ the row stride of
// writer_unary_interleaved_row_strided_blocked.cpp).
//
// CT (A::): cb_out blk row_tiles row_stride cta_dst      RT: dst_addr num_tiles tile_offset

#pragma once

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/experimental/lego/kernels/lego.hpp"

namespace lego::common {

template <typename A>
struct TileWriter {
    uint32_t dst_addr, num_tiles, tile_offset;

    void init() {
        dst_addr = rt<A>(0);
        num_tiles = rt<A>(1);
        tile_offset = rt<A>(2);
    }

    void run() {
        constexpr auto dst_args = TensorAccessorArgs<A::cta_dst>();
        const auto s = TensorAccessor(dst_args, dst_addr);
        const Noc noc;
        DataflowBuffer dfb_out_buf(A::cb_out);
        const uint32_t tile_bytes = dfb_out_buf.get_tile_size();
        uint32_t row_start = tile_offset, in_row = 0;
        for (uint32_t i = 0; i < num_tiles; i += A::blk) {
            dfb_out_buf.wait_front(A::blk);
            uint32_t write_offset = 0;
            for (uint32_t j = 0; j < A::blk; j++) {
                noc.async_write(
                    dfb_out_buf, s, tile_bytes, {.offset_bytes = write_offset}, {.page_id = row_start + in_row});
                if (++in_row == A::row_tiles) {
                    in_row = 0;
                    row_start += A::row_stride;
                }
                write_offset += tile_bytes;
            }
            noc.async_write_barrier();
            dfb_out_buf.pop_front(A::blk);
        }
    }

    void teardown() {}
};

}  // namespace lego::common
