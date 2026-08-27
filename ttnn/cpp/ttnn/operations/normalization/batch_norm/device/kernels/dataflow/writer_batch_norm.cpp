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

void kernel_main() {
    uint32_t start_tile_id = get_arg(args::start_tile_id);
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t HtWt = get_arg(args::HtWt);
    uint32_t n_stride = get_arg(args::n_stride);
    uint32_t c_stride = get_arg(args::c_stride);
    uint32_t N = get_arg(args::N);
    uint32_t C = get_arg(args::C);

    constexpr uint32_t onetile = 1;

    constexpr bool batch_stat_is_fp32 = get_arg(args::batch_stat_is_fp32) == 1;
    constexpr bool param_is_fp32 = get_arg(args::param_is_fp32) == 1;

    Noc noc;
    DataflowBuffer dfb_src(dfb::src);              // batch_mean, read here for the compute kernel
    DataflowBuffer dfb_dst(dfb::dst);              // the buffer the compute kernel finally packs into
    DataflowBuffer dfb_batch_var(dfb::batch_var);  // batch_var, likewise read here
    DataflowBuffer dfb_weight(dfb::weight);        // affine scale; bound even when the tensor is absent
    DataflowBuffer dfb_bias(dfb::bias);            // affine shift; bound even when the tensor is absent
    auto weight = construct_nullable_tensor(tensor::weight);
    auto bias = construct_nullable_tensor(tensor::bias);

    // batch_mean
    const uint32_t src_tile_bytes = dfb_src.get_entry_size();
    const auto src = TensorAccessor(tensor::batch_mean);

    // output
    const uint32_t dst_tile_bytes = dfb_dst.get_entry_size();
    const auto dst = TensorAccessor(tensor::output);

    // batch_var
    const uint32_t batch_var_tile_bytes = dfb_batch_var.get_entry_size();
    const auto batch_var = TensorAccessor(tensor::batch_var);

    const uint32_t weight_tile_bytes = dfb_weight.get_entry_size();
    const uint32_t bias_tile_bytes = dfb_bias.get_entry_size();

    uint32_t tiles_per_batch = HtWt * C;
    uint32_t start_n = start_tile_id / tiles_per_batch;
    uint32_t start_remaining = start_tile_id % tiles_per_batch;
    uint32_t start_c = start_remaining / HtWt;
    uint32_t start_t = start_remaining % HtWt;

    // Input tile offset
    uint32_t tile_offset = start_n * n_stride + start_c * c_stride;
    uint32_t next_batch_shift = n_stride - c_stride * C;

    uint32_t num_tiles_written = 0;
    for (uint32_t n = start_n; n < N && num_tiles_written < num_tiles; ++n, start_c = 0) {
        for (uint32_t c = start_c; c < C && num_tiles_written < num_tiles; ++c, start_t = 0) {
            // read a tile from src
            dfb_src.reserve_back(onetile);
            noc.async_read(src, dfb_src, src_tile_bytes, {.page_id = tile_offset}, {.offset_bytes = 0});
            noc.async_read_barrier();
            if constexpr (batch_stat_is_fp32) {
                fill_tile_with_first_element<float>(dfb_src.get_write_ptr());
            } else {
                fill_tile_with_first_element_bfloat16(dfb_src.get_write_ptr());
            }
            dfb_src.push_back(onetile);

            // read a tile from batch variance
            dfb_batch_var.reserve_back(onetile);
            noc.async_read(
                batch_var, dfb_batch_var, batch_var_tile_bytes, {.page_id = tile_offset}, {.offset_bytes = 0});
            noc.async_read_barrier();
            if constexpr (batch_stat_is_fp32) {
                fill_tile_with_first_element<float>(dfb_batch_var.get_write_ptr());
            } else {
                fill_tile_with_first_element_bfloat16(dfb_batch_var.get_write_ptr());
            }
            dfb_batch_var.push_back(onetile);

            with_nullable_token(tensor::weight, [&](auto const&) {
                // read a tile from weight tensor
                dfb_weight.reserve_back(onetile);
                noc.async_read(weight, dfb_weight, weight_tile_bytes, {.page_id = tile_offset}, {.offset_bytes = 0});
                noc.async_read_barrier();
                if constexpr (param_is_fp32) {
                    fill_tile_with_first_element<float>(dfb_weight.get_write_ptr());
                } else {
                    fill_tile_with_first_element_bfloat16(dfb_weight.get_write_ptr());
                }
                dfb_weight.push_back(onetile);
            });

            with_nullable_token(tensor::bias, [&](auto const&) {
                // read a tile from bias tensor
                dfb_bias.reserve_back(onetile);
                noc.async_read(bias, dfb_bias, bias_tile_bytes, {.page_id = tile_offset}, {.offset_bytes = 0});
                noc.async_read_barrier();
                if constexpr (param_is_fp32) {
                    fill_tile_with_first_element<float>(dfb_bias.get_write_ptr());
                } else {
                    fill_tile_with_first_element_bfloat16(dfb_bias.get_write_ptr());
                }
                dfb_bias.push_back(onetile);
            });

            for (uint32_t t = start_t; t < HtWt && num_tiles_written < num_tiles; ++t, ++num_tiles_written) {
                // write a tile to dst
                dfb_dst.wait_front(onetile);
                noc.async_write(
                    dfb_dst, dst, dst_tile_bytes, {.offset_bytes = 0}, {.page_id = start_tile_id + num_tiles_written});
                noc.async_write_barrier();
                dfb_dst.pop_front(onetile);
            }
            tile_offset += c_stride;
        }
        tile_offset += next_batch_shift;
    }
}
