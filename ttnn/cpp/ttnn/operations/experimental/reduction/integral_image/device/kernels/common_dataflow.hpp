// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#include "common.hpp"

template <typename addr_gen_t>
FORCE_INLINE void write_to_dram(
    uint32_t cb, const addr_gen_t& addr_gtor, uint32_t write_tile_id, uint32_t num_tiles = ONE_TILE) {
    ReadCBGuard read_guard{cb, num_tiles};

    Noc noc;
    DataflowBuffer cb_obj(cb);
    noc.async_write(
        cb_obj,
        addr_gtor,
        addr_gtor.get_aligned_page_size(),
        {.offset_bytes = 0},
        {.page_id = write_tile_id, .offset_bytes = 0});
    noc.async_write_barrier();
}

template <typename addr_gen_type>
FORCE_INLINE void load_from_dram(
    uint32_t cb, const addr_gen_type& addr_gtor, uint32_t read_tile_id, uint32_t num_tiles = ONE_TILE) {
    WriteCBGuard write_guard{cb, num_tiles};

    Noc noc;
    DataflowBuffer cb_obj(cb);
    noc.async_read(
        addr_gtor,
        cb_obj,
        addr_gtor.get_aligned_page_size(),
        {.page_id = read_tile_id, .offset_bytes = 0},
        {.offset_bytes = 0});
    noc.async_read_barrier();
}

// The nine scalar CTAs shared by the reader / writer (and compute). The DFB indices that used to sit in this
// struct are now Metal 2.0 DFB bindings, referenced as dfb::<name> at the call sites; the input/output
// TensorAccessorArgs blocks are now TensorParameter/TensorBinding, referenced as tensor::input / tensor::output.
struct IntImgCTAs {
    const uint32_t tile_height;
    const uint32_t tile_width;
    const uint32_t block_depth;
    const uint32_t num_channels;  // axis 4/4
    const uint32_t input_height;  // axis 3/4
    const uint32_t input_depth;   // axis 2/4
    const uint32_t num_batches;   // axis 1/4
    const uint32_t cores_x;
    const uint32_t cores_y;
};

FORCE_INLINE constexpr IntImgCTAs get_ctas() {
    return IntImgCTAs{
        get_arg(args::tile_height),
        get_arg(args::tile_width),
        get_arg(args::block_depth),
        get_arg(args::num_channels),
        get_arg(args::input_height),
        get_arg(args::input_depth),
        get_arg(args::num_batches),
        get_arg(args::cores_x),
        get_arg(args::cores_y),
    };
}
