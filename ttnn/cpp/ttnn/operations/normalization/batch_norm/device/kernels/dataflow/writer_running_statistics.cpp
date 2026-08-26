// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

template <uint32_t CTA_OFFSET, uint32_t ADDR_CRTA_OFFSET>
void read_old_write_new(
    const tensor_accessor::TensorBindingToken<CTA_OFFSET, ADDR_CRTA_OFFSET>& tensor_tok,
    DFBBindingToken old_tok,
    DFBBindingToken new_tok,
    Noc& noc,
    uint32_t tile_offset) {
    constexpr uint32_t onetile = 1;
    constexpr bool old_stat_is_fp32 = get_arg(args::old_stat_is_fp32) == 1;

    const auto accessor = TensorAccessor(tensor_tok);
    DataflowBuffer dfb_old(old_tok);
    DataflowBuffer dfb_new(new_tok);
    const uint32_t tile_bytes = dfb_old.get_entry_size();

    dfb_old.reserve_back(onetile);
    noc.async_read(accessor, dfb_old, tile_bytes, {.page_id = tile_offset}, {.offset_bytes = 0});
    noc.async_read_barrier();
    if constexpr (old_stat_is_fp32) {
        fill_tile_with_first_element<float>(dfb_old.get_write_ptr());
    } else {
        fill_tile_with_first_element_bfloat16(dfb_old.get_write_ptr());
    }
    dfb_old.push_back(onetile);

    dfb_new.wait_front(onetile);
    noc.async_write(dfb_new, accessor, tile_bytes, {.offset_bytes = 0}, {.page_id = tile_offset});
    noc.async_write_barrier();
    dfb_new.pop_front(onetile);
}

void kernel_main() {
    uint32_t start_tile_id = get_arg(args::start_tile_id);
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t HtWt = get_arg(args::HtWt);
    uint32_t n_stride = get_arg(args::n_stride);
    uint32_t c_stride = get_arg(args::c_stride);
    uint32_t N = get_arg(args::N);
    uint32_t C = get_arg(args::C);

    constexpr uint32_t onetile = 1;

    Noc noc;
    DataflowBuffer dfb_src(dfb::src);  // batch_var, read here for the compute kernel
    DataflowBuffer dfb_dst(dfb::dst);  // the op's output tensor

    const uint32_t src_tile_bytes = dfb_src.get_entry_size();
    const auto src = TensorAccessor(tensor::batch_var);

    const uint32_t dst_tile_bytes = dfb_dst.get_entry_size();
    const auto dst = TensorAccessor(tensor::output);

    uint32_t tiles_per_batch = HtWt * C;
    uint32_t start_n = start_tile_id / tiles_per_batch;
    uint32_t start_remaining = start_tile_id % tiles_per_batch;
    uint32_t start_c = start_remaining / HtWt;
    uint32_t start_t = start_remaining % HtWt;

    // this is the INPUT tile offset
    uint32_t tile_offset = start_n * n_stride + start_c * c_stride + start_t;
    uint32_t next_channel_shift = c_stride - HtWt;
    uint32_t next_batch_shift = n_stride - c_stride * C;

    uint32_t num_tiles_written = 0;
    for (uint32_t n = start_n; n < N && num_tiles_written < num_tiles; ++n, start_c = 0) {
        for (uint32_t c = start_c; c < C && num_tiles_written < num_tiles; ++c, start_t = 0) {
            for (uint32_t t = start_t; t < HtWt && num_tiles_written < num_tiles; ++t, ++num_tiles_written) {
                // read a tile from src
                dfb_src.reserve_back(onetile);
                noc.async_read(src, dfb_src, src_tile_bytes, {.page_id = tile_offset}, {.offset_bytes = 0});
                noc.async_read_barrier();
                dfb_src.push_back(onetile);

                with_nullable_token(tensor::running_mean, [&](const auto& mean_tok) {
                    with_nullable_token(dfb::old_mean, [&](const DFBBindingToken& old_tok) {
                        with_nullable_token(dfb::new_mean, [&](const DFBBindingToken& new_tok) {
                            read_old_write_new(mean_tok, old_tok, new_tok, noc, tile_offset);
                        });
                    });
                });

                with_nullable_token(tensor::running_var, [&](const auto& var_tok) {
                    with_nullable_token(dfb::old_var, [&](const DFBBindingToken& old_tok) {
                        with_nullable_token(dfb::new_var, [&](const DFBBindingToken& new_tok) {
                            read_old_write_new(var_tok, old_tok, new_tok, noc, tile_offset);
                        });
                    });
                });
                ++tile_offset;

                // write a tile to dst, since the dst shape is full, the tile offset simply grows linearly
                dfb_dst.wait_front(onetile);
                noc.async_write(
                    dfb_dst, dst, dst_tile_bytes, {.offset_bytes = 0}, {.page_id = start_tile_id + num_tiles_written});
                noc.async_write_barrier();
                dfb_dst.pop_front(onetile);
            }
            tile_offset += next_channel_shift;
        }
        tile_offset += next_batch_shift;
    }
}
