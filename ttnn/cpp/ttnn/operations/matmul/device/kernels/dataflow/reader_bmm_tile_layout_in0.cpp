// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port. Used only by the MatmulMultiCoreReuseOptimized factory, so ported in
// place. Logic, #ifdefs, and loop bounds are unchanged from the legacy reader; only the
// access mechanism moves to named bindings: the in0 tensor address -> ta::a, CB ids ->
// dfb::cb_in0 / dfb::cb_in0_intermediate, positional CT/RT args -> get_arg(args::...).
// On the IN0_SHARDED path the in0 CB is a borrowed-memory DFB backed by tensor `a`, so
// the tensor is not accessed via a TensorAccessor; the ta::a construction lives inside
// the existing #ifndef IN0_SHARDED block, so the factory binds tensor `a` to this kernel
// only on the non-sharded (NoC-read) path.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/kernel_helper_functions/pad_tile.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // RUNTIME ARGS
    // in0 tensor args (addr now arrives via the ta::a binding)
    uint32_t in0_tensor_start_tile_id = get_arg(args::in0_tensor_start_tile_id);
    // batch args
    const uint32_t batch = get_arg(args::batch);

    // COMPILE TIME ARGS
    // in0 tensor args
    constexpr uint32_t in0_tensor_stride_w = get_arg(args::in0_tensor_stride_w);
    constexpr uint32_t in0_tensor_stride_h = get_arg(args::in0_tensor_stride_h);
    constexpr uint32_t in0_tensor_next_block_stride = get_arg(args::in0_tensor_next_block_stride);
    // in0 block args
    constexpr uint32_t in0_block_w = get_arg(args::in0_block_w);
    constexpr uint32_t in0_block_h = get_arg(args::in0_block_h);
    constexpr uint32_t in0_block_num_tiles = get_arg(args::in0_block_num_tiles);
    constexpr uint32_t last_ktile_w = get_arg(args::last_ktile_w);
    constexpr uint32_t last_ktile_h = get_arg(args::last_ktile_h);
    // in0/in1 common args
    constexpr uint32_t num_blocks = get_arg(args::num_blocks);
    // batch args
    constexpr uint32_t bcast_B = get_arg(args::bcast_B);
    constexpr uint32_t MtKt = get_arg(args::MtKt);

    constexpr uint32_t cb_id_in0 = dfb::cb_in0;
    constexpr uint32_t one_tile = 1;

    Noc noc;
    CircularBuffer cb_in0(cb_id_in0);

#ifdef IN0_SHARDED
    const uint32_t in0_num_tiles = batch * num_blocks * in0_block_h * in0_block_w;
    cb_in0.reserve_back(in0_num_tiles);
    cb_in0.push_back(in0_num_tiles);
#else

    constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(cb_id_in0);
    constexpr const uint32_t in0_tile_hw = get_tile_hw(cb_id_in0);

    const auto s0 = TensorAccessor(ta::a);

#ifdef INTERMEDIATE_CB_READ
    constexpr uint32_t in0_intermediate_cb_index = dfb::cb_in0_intermediate;
    CircularBuffer cb_helper(in0_intermediate_cb_index);
#endif

    for (uint32_t b = 0; b < batch; ++b) {
        uint32_t in0_tensor_current_block_start_tile_id = in0_tensor_start_tile_id;
        for (uint32_t block = 0; block < num_blocks; ++block) {
            cb_in0.reserve_back(in0_block_num_tiles);

#ifdef INTERMEDIATE_CB_READ
            cb_helper.reserve_back(one_tile);
#endif  // INTERMEDIATE_CB_READ

            uint32_t in0_write_offset = 0;

            uint32_t in0_tensor_row_start_tile_id = in0_tensor_current_block_start_tile_id;
            for (uint32_t h = 0; h < in0_block_h; ++h) {
                uint32_t in0_tensor_tile_id = in0_tensor_row_start_tile_id;
                for (uint32_t w = 0; w < in0_block_w; ++w) {
#ifndef INTERMEDIATE_CB_READ
                    noc.async_read(
                        s0,
                        cb_in0,
                        in0_single_tile_size_bytes,
                        {.page_id = in0_tensor_tile_id},
                        {.offset_bytes = in0_write_offset});
#else
                    noc.async_read(
                        s0,
                        cb_helper,
                        in0_single_tile_size_bytes,
                        {.page_id = in0_tensor_tile_id},
                        {.offset_bytes = 0});
                    noc.async_read_barrier();
                    memcpy(
                        /*dst=*/reinterpret_cast<void*>(cb_in0.get_write_ptr() + in0_write_offset),
                        /*src=*/reinterpret_cast<const void*>(cb_helper.get_write_ptr()),
                        /*size=*/in0_single_tile_size_bytes);
#endif  // INTERMEDIATE_CB_READ

                    // Zero out padded regions for the very last tile
                    if constexpr (last_ktile_w > 0) {
                        if ((block == num_blocks - 1) && (w == in0_block_w - 1)) {
                            noc.async_read_barrier();
                            constexpr DataFormat in0_data_format = get_dataformat(cb_id_in0);
                            pad_last_ktile<in0_data_format, last_ktile_w>(cb_in0.get_write_ptr() + in0_write_offset);
                        }
                    }
                    if constexpr (last_ktile_h > 0) {
                        if ((block == num_blocks - 1) && (w == in0_block_w - 1)) {
                            noc.async_read_barrier();
                            constexpr DataFormat in0_data_format = get_dataformat(cb_id_in0);
                            pad_last_transposed_ktile<in0_data_format, last_ktile_h>(
                                cb_in0.get_write_ptr() + in0_write_offset);
                        }
                    }

                    in0_write_offset += in0_single_tile_size_bytes;
                    in0_tensor_tile_id += in0_tensor_stride_w;
                }
                in0_tensor_row_start_tile_id += in0_tensor_stride_h;
            }
            in0_tensor_current_block_start_tile_id += in0_tensor_next_block_stride;

            noc.async_read_barrier();

            cb_in0.push_back(in0_block_num_tiles);

#ifdef INTERMEDIATE_CB_READ
            // Clean up helper CB
            cb_helper.push_back(one_tile);
            cb_helper.wait_front(one_tile);
            cb_helper.pop_front(one_tile);
#endif  // INTERMEDIATE_CB_READ
        }
        in0_tensor_start_tile_id += MtKt;
    }
#endif  // IN0_SHARDED
}
