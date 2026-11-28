// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t start_tile_id = get_arg(args::start_tile_id);
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t HtWt = get_arg(args::HtWt);
    const uint32_t n_stride = get_arg(args::n_stride);
    const uint32_t c_stride = get_arg(args::c_stride);
    const uint32_t N = get_arg(args::N);
    const uint32_t C = get_arg(args::C);

    constexpr uint32_t onetile = 1;

    constexpr bool batch_stat_is_fp32 = get_arg(args::batch_stat_is_fp32) == 1;
    constexpr bool param_is_fp32 = get_arg(args::param_is_fp32) == 1;

    const Noc noc;
    DataflowBuffer dfb_src(dfb::src);              // batch_mean, read here for the compute kernel
    DataflowBuffer dfb_dst(dfb::dst);              // the buffer the compute kernel finally packs into
    DataflowBuffer dfb_batch_var(dfb::batch_var);  // batch_var, likewise read here
    DataflowBuffer dfb_weight(dfb::weight);        // affine scale; bound even when absent
    DataflowBuffer dfb_bias(dfb::bias);            // affine shift; bound even when absent

    // batch_mean
    const uint32_t src_tile_bytes = dfb_src.get_entry_size();
    const auto src = TensorAccessor(tensor::batch_mean);

    // output
    const uint32_t dst_tile_bytes = dfb_dst.get_entry_size();
    const auto dst = TensorAccessor(tensor::output);

    // batch_var
    const uint32_t batch_var_tile_bytes = dfb_batch_var.get_entry_size();
    const auto batch_var = TensorAccessor(tensor::batch_var);

    // weight
    const uint32_t weight_tile_bytes = dfb_weight.get_entry_size();
#ifdef WEIGHT_HAS_VALUE
    // An absent optional tensor is not bound, so tensor::weight does not exist on that build; the
    // accessor has to disappear at the preprocessor stage, before name lookup.
    const auto weight = TensorAccessor(tensor::weight);
#endif

    // bias
    const uint32_t bias_tile_bytes = dfb_bias.get_entry_size();
#ifdef BIAS_HAS_VALUE
    const auto bias = TensorAccessor(tensor::bias);
#endif

    const uint32_t tiles_per_batch = HtWt * C;
    const uint32_t start_n = start_tile_id / tiles_per_batch;
    const uint32_t start_remaining = start_tile_id % tiles_per_batch;
    uint32_t start_c = start_remaining / HtWt;
    uint32_t start_t = start_remaining % HtWt;

    // Input tile offset
    uint32_t tile_offset = (start_n * n_stride) + (start_c * c_stride);
    const uint32_t next_batch_shift = n_stride - (c_stride * C);

#if defined(ARCH_WORMHOLE)
    constexpr uint32_t stat_read_bytes = 32;
    constexpr uint32_t seed_bytes = 64;
    constexpr uint32_t chunk_bytes = 512;

    auto fill_seed = [&](uint32_t addr, bool is_fp32, uint32_t bytes) __attribute__((always_inline)) {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
        uint32_t v = p[0];
        if (!is_fp32) {
            v = (v & 0xFFFFu) | (v << 16);
        }
        for (uint32_t i = 0; i < seed_bytes / 4; ++i) {
            p[i] = v;
        }
        const uint32_t lim = bytes < chunk_bytes ? bytes : chunk_bytes;
        for (uint32_t o = seed_bytes; o < lim; o += seed_bytes) {
            noc.async_read(
                UnicastEndpoint{},
                CoreLocalMem<uint32_t>(addr + o),
                seed_bytes,
                {.noc_x = my_x[noc_index], .noc_y = my_y[noc_index], .addr = addr},
                {});
        }
    };
    auto fill_rest = [&](uint32_t addr, uint32_t bytes) __attribute__((always_inline)) {
        for (uint32_t o = chunk_bytes; o < bytes; o += chunk_bytes) {
            noc.async_read(
                UnicastEndpoint{},
                CoreLocalMem<uint32_t>(addr + o),
                chunk_bytes,
                {.noc_x = my_x[noc_index], .noc_y = my_y[noc_index], .addr = addr},
                {});
        }
    };
    auto read_stats = [&](uint32_t off) __attribute__((always_inline)) {
        dfb_src.reserve_back(onetile);
        noc.async_read(src, dfb_src, stat_read_bytes, {.page_id = off}, {.offset_bytes = 0});
        dfb_batch_var.reserve_back(onetile);
        noc.async_read(batch_var, dfb_batch_var, stat_read_bytes, {.page_id = off}, {.offset_bytes = 0});
#ifdef WEIGHT_HAS_VALUE
        dfb_weight.reserve_back(onetile);
        noc.async_read(weight, dfb_weight, stat_read_bytes, {.page_id = off}, {.offset_bytes = 0});
#endif
#ifdef BIAS_HAS_VALUE
        dfb_bias.reserve_back(onetile);
        noc.async_read(bias, dfb_bias, stat_read_bytes, {.page_id = off}, {.offset_bytes = 0});
#endif
        noc.async_read_barrier();
        fill_seed(dfb_src.get_write_ptr(), batch_stat_is_fp32, src_tile_bytes);
        fill_seed(dfb_batch_var.get_write_ptr(), batch_stat_is_fp32, batch_var_tile_bytes);
#ifdef WEIGHT_HAS_VALUE
        fill_seed(dfb_weight.get_write_ptr(), param_is_fp32, weight_tile_bytes);
#endif
#ifdef BIAS_HAS_VALUE
        fill_seed(dfb_bias.get_write_ptr(), param_is_fp32, bias_tile_bytes);
#endif
        noc.async_read_barrier();
        fill_rest(dfb_src.get_write_ptr(), src_tile_bytes);
        fill_rest(dfb_batch_var.get_write_ptr(), batch_var_tile_bytes);
#ifdef WEIGHT_HAS_VALUE
        fill_rest(dfb_weight.get_write_ptr(), weight_tile_bytes);
#endif
#ifdef BIAS_HAS_VALUE
        fill_rest(dfb_bias.get_write_ptr(), bias_tile_bytes);
#endif
        noc.async_read_barrier();
        dfb_src.push_back(onetile);
        dfb_batch_var.push_back(onetile);
#ifdef WEIGHT_HAS_VALUE
        dfb_weight.push_back(onetile);
#endif
#ifdef BIAS_HAS_VALUE
        dfb_bias.push_back(onetile);
#endif
    };

    uint32_t num_tiles_written = 0;
    auto write_out = [&](uint32_t count) __attribute__((always_inline)) {
        for (uint32_t k = 0; k < count; ++k, ++num_tiles_written) {
            dfb_dst.wait_front(onetile);
            noc.async_write(
                dfb_dst, dst, dst_tile_bytes, {.offset_bytes = 0}, {.page_id = start_tile_id + num_tiles_written});
            noc.async_write_barrier();
            dfb_dst.pop_front(onetile);
        }
    };

    uint32_t assigned = 0;
    uint32_t pending = 0;
    for (uint32_t n = start_n; n < N && assigned < num_tiles; ++n, start_c = 0) {
        for (uint32_t c = start_c; c < C && assigned < num_tiles; ++c, start_t = 0) {
            const uint32_t left = HtWt - start_t;
            const uint32_t count = (num_tiles - assigned) < left ? (num_tiles - assigned) : left;
            assigned += count;
            read_stats(tile_offset);
            write_out(pending);
            pending = count;
            tile_offset += c_stride;
        }
        tile_offset += next_batch_shift;
    }
    write_out(pending);
#else
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

#ifdef WEIGHT_HAS_VALUE
            {  // read a tile from weight tensor
                dfb_weight.reserve_back(onetile);
                noc.async_read(weight, dfb_weight, weight_tile_bytes, {.page_id = tile_offset}, {.offset_bytes = 0});
                noc.async_read_barrier();
                if constexpr (param_is_fp32) {
                    fill_tile_with_first_element<float>(dfb_weight.get_write_ptr());
                } else {
                    fill_tile_with_first_element_bfloat16(dfb_weight.get_write_ptr());
                }
                dfb_weight.push_back(onetile);
            }
#endif

#ifdef BIAS_HAS_VALUE
            {  // read a tile from bias tensor
                dfb_bias.reserve_back(onetile);
                noc.async_read(bias, dfb_bias, bias_tile_bytes, {.page_id = tile_offset}, {.offset_bytes = 0});
                noc.async_read_barrier();
                if constexpr (param_is_fp32) {
                    fill_tile_with_first_element<float>(dfb_bias.get_write_ptr());
                } else {
                    fill_tile_with_first_element_bfloat16(dfb_bias.get_write_ptr());
                }
                dfb_bias.push_back(onetile);
            }
#endif

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
#endif
}
