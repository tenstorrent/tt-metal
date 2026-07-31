// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: This is the Metal 2.0 fork of reader_binary_interleaved_start_id.cpp, which lives beside it.
// Ops ported to Metal 2.0 bind this file; the original serves the consumers still on the legacy API.
// Until the last of them migrates and the original is retired, changes here likely belong there too.
//
// The binding names below (dfb::in0 / dfb::in1, tensor::src0 / tensor::src1) and the named argument
// set are this fork's interface: every later consumer inherits them, so they are taken from the
// kernel's own vocabulary rather than any one op's locals, and are not renamed once a consumer
// exists.

// This code is temporarily copied from ttnn/operations/datamovement/binary/device/ to demonstrate
// the ability to keep the dataflow-buffer configs contiguous during dispatching.
// When broadcating is properly supported we expect this code to be deleted or refactored substantially.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // same arg set as in reader_binary_diff_lengths for compat
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t start_id = get_arg(args::start_id);
    uint32_t block_height = get_arg(args::block_height);
    uint32_t block_width = get_arg(args::block_width);
    uint32_t num_cores_y = get_arg(args::num_cores_y);

    constexpr bool block_or_width_sharded = get_arg(args::block_or_width_sharded) == 1;

    Noc noc;
    // dfb0 / dfb1 hold the two input operands' tiles; the host binds this kernel as their producer.
    DataflowBuffer dfb0(dfb::in0);
    DataflowBuffer dfb1(dfb::in1);

#ifdef IN0_SHARDED
    dfb0.reserve_back(num_tiles);
    dfb0.push_back(num_tiles);
#else
    uint32_t src0_tile_bytes = dfb0.get_tile_size();
    const auto s0 = TensorAccessor(tensor::src0);
#endif
#ifdef IN1_SHARDED
    dfb1.reserve_back(num_tiles);
    dfb1.push_back(num_tiles);
#else
    uint32_t src1_tile_bytes = dfb1.get_tile_size();
    const auto s1 = TensorAccessor(tensor::src1);
#endif

#if !(defined IN0_SHARDED && defined IN1_SHARDED)

    constexpr uint32_t onetile = 1;

    if constexpr (block_or_width_sharded) {
        uint32_t row_start_tile_id = start_id;
        for (uint32_t h = 0; h < block_height; h++) {
            uint32_t tile_id = row_start_tile_id;
            for (uint32_t w = 0; w < block_width; w++) {
#ifndef IN0_SHARDED
                dfb0.reserve_back(onetile);
                noc.async_read(s0, dfb0, src0_tile_bytes, {.page_id = tile_id}, {.offset_bytes = 0});
#endif

#ifndef IN1_SHARDED
                dfb1.reserve_back(onetile);
                noc.async_read(s1, dfb1, src1_tile_bytes, {.page_id = tile_id}, {.offset_bytes = 0});
#endif

                tile_id++;
                noc.async_read_barrier();

#ifndef IN0_SHARDED
                dfb0.push_back(onetile);
#endif

#ifndef IN1_SHARDED
                dfb1.push_back(onetile);
#endif
            }
            row_start_tile_id += num_cores_y * block_width;
        }
    } else {
        for (uint32_t tile_id = start_id; tile_id < start_id + num_tiles; tile_id++) {
#ifndef IN0_SHARDED
            dfb0.reserve_back(onetile);
            noc.async_read(s0, dfb0, src0_tile_bytes, {.page_id = tile_id}, {.offset_bytes = 0});
#endif

#ifndef IN1_SHARDED
            dfb1.reserve_back(onetile);
            noc.async_read(s1, dfb1, src1_tile_bytes, {.page_id = tile_id}, {.offset_bytes = 0});
#endif

            noc.async_read_barrier();

#ifndef IN0_SHARDED
            dfb0.push_back(onetile);
#endif

#ifndef IN1_SHARDED
            dfb1.push_back(onetile);
#endif
        }
    }
#endif
}
